"""Training schedule and regularization options of issue #14; every default keeps the original recipe."""

import copy
import dataclasses
import json
import math
import shutil
import sqlite3
import types
from pathlib import Path

import pytest
import torch
import yaml

from dacvae_tts import training
from dacvae_tts.config import Config, ModelConfig, TrainConfig
from dacvae_tts.data import LatentDataset, collate
from dacvae_tts.model import FlowTTS, flow_loss, guidance_direction, per_example_mse, to_velocity
from dacvae_tts.training import (
    Objective,
    decay_phase_loader,
    decay_start,
    ema_key,
    ema_tracks,
    export_ema,
    load_model,
    lr_multiplier,
    schedule_multiplier,
    time_sampling_at,
    weights_key,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
NANO = dict(
    latent_dim=4, width=32, heads=2, depth=2, text_depth=1, text_attention=1, patch_size=1, positions="rope",
    qk_norm=True, prediction="edm", text_layout="joined", duration="rule", ctc_layer=1,
)
# Configuration keys added for issue #14; configurations saved before it lack them.
NEW_FIELDS = {
    "model": ("dropout",),
    "train": (
        "lr_schedule", "decay_fraction", "decay_shape", "min_lr_ratio", "decay_cache", "final_time_sampling",
        "final_time_sampling_start", "ema_decays", "model_guidance_weight",
    ),
}
TINY = {"latent_dim": 4, "width": 16, "depth": 1, "heads": 2, "text_depth": 1}


def original_lr_multiplier(step, warmup, steps):
    """The schedule before issue #14, verbatim."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(steps - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def as_before_issue_14(config):
    """A saved configuration as the code before these options wrote it."""
    old = json.loads(json.dumps(config))
    for section, keys in NEW_FIELDS.items():
        for key in keys:
            del old[section][key]
    return old


def config_file(tmp_path, name="test.yaml", model=None, **train):
    path = tmp_path / name
    fields = dict(steps=6, warmup=1, batch_size=2, accumulation=2, workers=0, precision="fp32", log_every=1,
                  checkpoint_every=3, validate_every=2)
    fields.update(train)
    path.write_text(yaml.safe_dump({"model": model or TINY, "train": fields}))
    return path


def run(config, cache, output, **options):
    """One in-process training run (train() seeds itself, so runs are reproducible); returns last.pt."""
    args = dict(
        config=str(config), cache=str(cache), output=str(output), resume=None, init_from=None, device="cpu",
        steps=None, batch_size=None, accumulation=None, workers=None, precision=None, learning_rate=None,
        optimizer=None, worker_threads=None, prefetch_factor=None, loader_start_method=None, cuda_prefetch=None,
        frame_budget=0, compile=None, no_validation=False, stop_after=None,
    )
    args.update(options)
    training.train(types.SimpleNamespace(**args))
    return torch.load(output / "last.pt", weights_only=True)


# ----------------------------------------------------------------------------------------------- schedules


def test_cosine_schedule_is_bit_identical_to_the_original():
    for warmup, steps in ((2000, 60000), (300, 10000), (0, 7), (5, 5)):
        train = TrainConfig(steps=steps, warmup=warmup)
        for step in [*range(0, steps + 3, max(steps // 997, 1)), warmup - 1, warmup, steps - 1, steps]:
            expected = original_lr_multiplier(step, warmup, steps)
            assert lr_multiplier(step, warmup, steps) == expected
            assert schedule_multiplier(step, train) == expected


@pytest.mark.parametrize("shape", ["1-sqrt", "linear"])
def test_wsd_warmup_stable_and_decay(shape):
    steps, warmup = 1000, 100
    train = TrainConfig(steps=steps, warmup=warmup, lr_schedule="wsd", decay_fraction=0.2, decay_shape=shape)
    curve = [schedule_multiplier(step, train) for step in range(steps + 5)]
    start = decay_start(steps, 0.2)
    assert start == 800
    assert curve[:warmup] == [original_lr_multiplier(s, warmup, steps) for s in range(warmup)]
    assert all(value == 1.0 for value in curve[warmup : start + 1])  # stable phase, continuous at the start
    decay = curve[start:steps]
    assert all(a >= b for a, b in zip(decay, decay[1:])) and decay[-1] > 0.1
    quarter, half = curve[start + 50], curve[start + 100]
    if shape == "1-sqrt":
        assert quarter == pytest.approx(0.1 + 0.9 * 0.5) and half == pytest.approx(0.1 + 0.9 * (1 - 0.5**0.5))
    else:
        assert half == pytest.approx(0.55)
    assert curve[steps:] == [pytest.approx(0.1)] * 5  # past the end: the floor
    to_zero = TrainConfig(steps=steps, warmup=warmup, lr_schedule="wsd", decay_shape=shape, min_lr_ratio=0.0)
    assert schedule_multiplier(steps, to_zero) == 0.0
    assert schedule_multiplier(steps - 1, to_zero) == pytest.approx(
        1 - (199 / 200) ** 0.5 if shape == "1-sqrt" else 1 / 200
    )


def test_pure_cooldown_for_branching_from_a_stable_checkpoint():
    """decay_fraction 1 without warmup: a cooldown branched from a stable-phase checkpoint with --init-from."""
    train = TrainConfig(steps=100, warmup=0, lr_schedule="wsd", decay_fraction=1.0, decay_shape="linear")
    assert schedule_multiplier(0, train) == 1.0 and schedule_multiplier(50, train) == pytest.approx(0.55)


def test_final_time_sampling_switch():
    default = TrainConfig(steps=100, time_sampling="logit_normal")
    assert {time_sampling_at(step, default) for step in range(120)} == {"logit_normal"}
    wsd = TrainConfig(steps=100, warmup=10, time_sampling="logit_normal", lr_schedule="wsd",
                      decay_fraction=0.25, final_time_sampling="uniform")
    modes = [time_sampling_at(step, wsd) for step in range(100)]
    assert modes == ["logit_normal"] * 75 + ["uniform"] * 25  # tied to the decay start
    fraction = TrainConfig(steps=100, time_sampling="logit_normal", final_time_sampling="uniform",
                           final_time_sampling_start=0.9)
    assert [time_sampling_at(step, fraction) for step in (89, 90)] == ["logit_normal", "uniform"]


def test_schedule_configuration_guards():
    bad = [
        dict(lr_schedule="step"),
        dict(lr_schedule="wsd", decay_shape="cosine"),
        dict(lr_schedule="wsd", decay_fraction=0.0),
        dict(lr_schedule="wsd", steps=100, warmup=90, decay_fraction=0.2),  # decay inside the warmup
        dict(min_lr_ratio=1.5),
        dict(decay_cache="data/hq"),  # needs wsd
        dict(final_time_sampling="beta"),
        dict(final_time_sampling="uniform"),  # "decay" start without wsd
        dict(final_time_sampling="uniform", final_time_sampling_start=1.0),
    ]
    for fields in bad:
        with pytest.raises(ValueError):
            TrainConfig(**fields)
    TrainConfig(lr_schedule="wsd", decay_cache="data/hq", final_time_sampling="uniform")


def test_old_configurations_compare_equal_for_resume():
    """Resume compares the saved configuration, completed with the new defaults, against the YAML's."""
    new = Config.load(CONFIGS / "nano_tr_w512.yaml").to_dict()
    assert Config.from_dict(as_before_issue_14(new)).to_dict() == new


def test_old_checkpoints_load_and_resume_exactly(cache, tmp_path):
    config = config_file(tmp_path, steps=4)
    full = run(config, cache, tmp_path / "full")
    assert "recommended_guidance" not in full and not any(k.startswith("ema_") for k in full)
    run(config, cache, tmp_path / "old", stop_after=2)
    path = tmp_path / "old" / "last.pt"
    saved = torch.load(path, weights_only=True)
    saved["config"] = as_before_issue_14(saved["config"])
    torch.save(saved, path)
    for ema in (True, False, 0.999, "ema"):
        load_model(path, ema=ema)
    resumed = run(config, cache, tmp_path / "old", resume=str(path))
    for key in full["model"]:
        assert torch.equal(full["model"][key], resumed["model"][key]), key
        assert torch.equal(full["ema"][key], resumed["ema"][key]), key


@pytest.fixture
def decay_cache(cache, tmp_path):
    """A 'high-quality' subset: the same shards, only two of the four training speakers."""
    path = tmp_path / "hq"
    shutil.copytree(cache, path)
    with sqlite3.connect(path / "index.sqlite") as db:
        db.execute("DELETE FROM samples WHERE split='train' AND speaker IN ('train-2','train-3')")
    return path


def test_decay_loader_checks_the_codec_and_keeps_the_main_normalization(cache, decay_cache, tmp_path):
    cfg = Config.from_dict(yaml.safe_load(config_file(tmp_path, lr_schedule="wsd").read_text()))
    main = LatentDataset(cache, "train")
    main.mean, main.std = main.mean + 1, main.std * 2  # make a mismatch with the cache's own stats visible
    _, loader = decay_phase_loader(str(decay_cache), main, cfg, 0, 1, 0, torch.device("cpu"))
    assert len(loader.dataset) == 6 and torch.equal(loader.dataset.mean, main.mean)
    assert torch.equal(loader.dataset.std, main.std)
    meta = json.loads((decay_cache / "metadata.json").read_text())
    (decay_cache / "metadata.json").write_text(json.dumps({**meta, "hop_length": 256}))
    with pytest.raises(ValueError, match="hop_length"):
        decay_phase_loader(str(decay_cache), main, cfg, 0, 1, 0, torch.device("cpu"))


def test_options_resume_exactly_across_the_decay_switch(cache, decay_cache, tmp_path, monkeypatch):
    """WSD + decay cache + uniform-t cooldown + dropout + an EMA track + model guidance, interrupted before,
    exactly at and after the decay start (update 3 of 6), must equal the uninterrupted run."""
    calls = []
    original = training.flow_loss

    def spy(*args, **kwargs):
        calls.append((kwargs["time_sampling"], kwargs["guidance_weight"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "flow_loss", spy)
    config = config_file(
        tmp_path, model={**TINY, "dropout": 0.1}, lr_schedule="wsd", decay_fraction=0.5, decay_shape="1-sqrt",
        min_lr_ratio=0.0, decay_cache=str(decay_cache), time_sampling="logit_normal",
        final_time_sampling="uniform", ema_decays=[0.2], model_guidance_weight=0.5,
    )
    full = run(config, cache, tmp_path / "full", no_validation=True)
    assert calls == [("logit_normal", 0.5)] * 6 + [("uniform", 0.5)] * 6  # two micro-batches per update
    assert full["batch_offset"] == 3  # 6 decay-phase batches from a 3-batch subset: the switch happened
    assert "ema_0.2" in full
    assert full["recommended_guidance"] == 1.0
    records = [json.loads(line) for line in (tmp_path / "full" / "train.jsonl").read_text().splitlines()]
    decay = [1.0, 1 - (1 / 3) ** 0.5, 1 - (2 / 3) ** 0.5]
    assert [r["lr"] for r in records] == pytest.approx([3e-4 * m for m in [1.0, 1.0, 1.0, *decay]])
    output = tmp_path / "resumed"
    run(config, cache, output, no_validation=True, stop_after=2)
    for stop in (3, 4, None):
        resumed = run(config, cache, output, no_validation=True, resume=str(output / "last.pt"), stop_after=stop)
    assert resumed["step"] == 6
    for key in full["model"]:
        for weights in ("model", "ema", "ema_0.2"):
            assert torch.equal(full[weights][key], resumed[weights][key]), (weights, key)


# ------------------------------------------------------------------------------------------------- dropout


def randomized(model, seed=0):
    """Zero-initialized AdaLN/output would make every branch identical; perturb them so conditions matter."""
    torch.manual_seed(seed)
    for block in model.blocks:
        last = block.ada_up if hasattr(block, "ada_up") else block.ada[-1]
        torch.nn.init.normal_(last.weight, std=0.05)
    torch.nn.init.normal_(model.output[-1].weight, std=0.05)
    return model


def nano_batch(prompts=(3, 4, 0), frames=9):
    torch.manual_seed(1)
    items = []
    for index, prompt in enumerate(prompts):
        latents = torch.randn(frames + 2 * index, 4)
        items.append(
            dict(reference=latents[:prompt], target=latents[prompt:], reference_text="",
                 text=f"Spoken words number {index}.", layout="joined")
        )
    return collate(items)


def test_zero_dropout_installs_nothing_and_draws_no_random_numbers():
    torch.manual_seed(0)
    plain = FlowTTS(ModelConfig(**NANO))
    torch.manual_seed(0)
    dropped = FlowTTS(ModelConfig(**NANO, dropout=0.1))
    # Same parameter names and values: checkpoints move freely between the two.
    assert plain.state_dict().keys() == dropped.state_dict().keys()
    assert all(torch.equal(v, dropped.state_dict()[k]) for k, v in plain.state_dict().items())
    for block in plain.blocks:
        assert not block.self_attn._forward_hooks and isinstance(block.ff[1], torch.nn.GELU)
    model = randomized(plain).train()
    batch = nano_batch()
    kwargs = {k: v for k, v in batch.items() if k != "latents"}
    time = torch.full((3,), 0.4)
    state = torch.get_rng_state()
    training_output = model(batch["latents"], time, **kwargs)
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(training_output, model.eval()(batch["latents"], time, **kwargs))


def test_dropout_is_active_only_in_training():
    model = randomized(FlowTTS(ModelConfig(**NANO, dropout=0.3)))
    reference = FlowTTS(ModelConfig(**NANO))
    reference.load_state_dict(model.state_dict())
    batch = nano_batch()
    kwargs = {k: v for k, v in batch.items() if k != "latents"}
    time = torch.full((3,), 0.4)
    model.train()
    first, second = model(batch["latents"], time, **kwargs), model(batch["latents"], time, **kwargs)
    assert not torch.allclose(first, second)
    expected = reference.eval()(batch["latents"], time, **kwargs)
    assert torch.equal(model.eval()(batch["latents"], time, **kwargs), expected)
    # The EMA is a deep copy: it keeps the hooks and is evaluated without dropout.
    assert torch.equal(copy.deepcopy(model).eval()(batch["latents"], time, **kwargs), expected)


def test_warm_start_may_switch_dropout_but_nothing_else(cache, tmp_path):
    run(config_file(tmp_path, "base.yaml", steps=3), cache, tmp_path / "base")
    warm = str(tmp_path / "base" / "last.pt")
    tuned = run(config_file(tmp_path, "tune.yaml", model={**TINY, "dropout": 0.1}, steps=2), cache,
                tmp_path / "tune", init_from=warm)
    assert tuned["step"] == 2 and all(torch.isfinite(v).all() for v in tuned["model"].values())
    other = config_file(tmp_path, "other.yaml", model={**TINY, "dropout": 0.1, "width": 32})
    with pytest.raises(ValueError, match="identical model configuration"):
        run(other, cache, tmp_path / "other", init_from=warm)


def test_regularization_configuration_guards():
    with pytest.raises(ValueError):
        ModelConfig(dropout=1.0)
    for fields in (dict(ema_decays=[0.999, 0.999]), dict(ema_decays=[1.0]), dict(ema_decays=0.999)):
        with pytest.raises(ValueError):
            TrainConfig(**fields)
    TrainConfig(ema_decays=[0.999, 0.9995])


# ---------------------------------------------------------------------------------------------- EMA tracks


def test_ema_keys_and_selection():
    assert ema_key(0.999) == "ema_0.999" and ema_key(0.9995) == "ema_0.9995"
    train = TrainConfig(ema_decay=0.9999, ema_decays=[0.9999, 0.999, 0.9995])
    assert ema_tracks(train) == {"ema_0.999": 0.999, "ema_0.9995": 0.9995}
    assert ema_tracks(TrainConfig()) == {}
    checkpoint = {"config": Config(train=train).to_dict(), "model": 0, "ema": 1, "ema_0.999": 2}
    assert weights_key(checkpoint) == "ema" and weights_key(checkpoint, False) == "model"
    assert weights_key(checkpoint, 0.9999) == "ema" and weights_key(checkpoint, "0.9999") == "ema"
    assert weights_key(checkpoint, 0.999) == weights_key(checkpoint, "ema_0.999") == "ema_0.999"
    with pytest.raises(KeyError, match="ema_0.999"):
        weights_key(checkpoint, 0.99)


def test_extra_ema_track_matches_a_run_with_that_decay(cache, tmp_path):
    """The track is updated exactly like the primary EMA: a run whose `ema_decay` is the track's decay
    reproduces it bit for bit. Tracks are saved, kept, validated, selectable and exportable. (Decays this
    short leave the warm-up min(decay, (1 + step) / (10 + step)) within six updates.)"""
    tracked = run(config_file(tmp_path, "a.yaml", ema_decay=0.2, ema_decays=[0.2, 0.3], keep_every=3),
                  cache, tmp_path / "tracked")
    single = run(config_file(tmp_path, "b.yaml", ema_decay=0.3), cache, tmp_path / "single")
    assert set(tracked) - set(single) == {"ema_0.3"}
    for key in single["model"]:
        assert torch.equal(tracked["model"][key], single["model"][key]), key
        assert torch.equal(tracked["ema_0.3"][key], single["ema"][key]), key
    assert any(not torch.equal(tracked["ema"][k], tracked["ema_0.3"][k]) for k in single["model"])
    assert "ema_0.3" in torch.load(tmp_path / "tracked" / "step-0000003.pt", weights_only=True)
    records = [json.loads(line) for line in (tmp_path / "tracked" / "train.jsonl").read_text().splitlines()]
    validations = [r for r in records if "validation_flow" in r]
    assert len(validations) == 3 and all(math.isfinite(r["ema_0_3/validation_flow"]) for r in validations)
    track, _ = load_model(tmp_path / "tracked" / "last.pt", ema=0.3)
    assert all(torch.equal(v, tracked["ema_0.3"][k]) for k, v in track.state_dict().items())
    primary, _ = load_model(tmp_path / "tracked" / "last.pt", ema=0.2)
    assert all(torch.equal(v, tracked["ema"][k]) for k, v in primary.state_dict().items())
    export_ema(tmp_path / "tracked" / "last.pt", "0.3", tmp_path / "exported.pt")
    exported, saved = load_model(tmp_path / "exported.pt")
    assert saved["exported_ema"] == "ema_0.3" and "optimizer" not in saved and "ema_0.3" not in saved
    assert all(torch.equal(v, tracked["ema_0.3"][k]) for k, v in exported.state_dict().items())


def test_ema_warmup_off_applies_each_decay_to_the_warm_started_ema(cache, tmp_path):
    """With the default warm-up, min(decay, (1 + step) / (10 + step)) is 0.1 for every track at the first
    update: the tracks are identical and the warm-started EMA gets weight 0.1. `ema_warmup: false` averages
    the checkpoint's EMA with each track's own decay from the first update on; training is unchanged."""
    assert training.ema_rate(0.999, 0) == 0.1 and training.ema_rate(0.999, 8990) == 0.999
    assert training.ema_rate(0.999, 0, warmup=False) == 0.999
    run(config_file(tmp_path, "base.yaml", steps=3), cache, tmp_path / "base")
    warm = torch.load(tmp_path / "base" / "last.pt", weights_only=True)
    tracks = dict(steps=4, ema_decay=0.9, ema_decays=[0.9, 0.5])
    options = dict(init_from=str(tmp_path / "base" / "last.pt"), stop_after=1)
    default = run(config_file(tmp_path, "default.yaml", **tracks), cache, tmp_path / "default", **options)
    exact = run(config_file(tmp_path, "exact.yaml", ema_warmup=False, **tracks), cache, tmp_path / "exact",
                **options)
    assert Config.from_dict(exact["config"]).train.ema_warmup is False and exact["step"] == 1
    for key, source in warm["ema"].items():
        assert torch.equal(default["model"][key], exact["model"][key]), key
        assert torch.equal(default["ema"][key], default["ema_0.5"][key]), key
        assert torch.equal(default["ema"][key], source.lerp(default["model"][key], 1 - 0.1)), key
        assert torch.equal(exact["ema"][key], source.lerp(exact["model"][key], 1 - 0.9)), key
        assert torch.equal(exact["ema_0.5"][key], source.lerp(exact["model"][key], 1 - 0.5)), key
    assert any(not torch.equal(exact["ema"][k], exact["ema_0.5"][k]) for k in warm["ema"])
    # From scratch the random initialization would dominate an average without warm-up.
    with pytest.raises(ValueError, match="init-from"):
        run(config_file(tmp_path, "scratch.yaml", ema_warmup=False), cache, tmp_path / "scratch")
    with pytest.raises(ValueError, match="ema_warmup"):
        TrainConfig(ema_warmup="false")
    assert Config.load(CONFIGS / "experiments" / "tr_w512_model_guidance_ft.yaml").train.ema_warmup is False


# ------------------------------------------------------------------------------------------ model guidance


def guided_setup(prediction):
    config = dict(NANO, prediction=prediction, ctc_layer=0)
    model = randomized(FlowTTS(ModelConfig(**config)))
    batch = nano_batch()
    torch.manual_seed(5)
    time = torch.tensor([0.2, 0.55, 0.9])
    noise = torch.randn_like(batch["latents"])
    return model, batch, time, noise


def branch_outputs(model, batch, time, noise):
    """Conditional and null outputs at the training state x_t, computed independently of flow_loss."""
    t = time[:, None, None]
    x1 = batch["latents"] * batch["valid"][..., None]
    xt = (1 - t) * noise * batch["valid"][..., None] + t * x1
    xt = torch.where(batch["prompt_mask"][..., None], x1, xt) * batch["valid"][..., None]
    kwargs = {k: batch[k] for k in ("prompt", "prompt_mask", "valid", "tokens", "segments")}
    b = len(time)
    with torch.no_grad():
        cond = model(xt, time, **kwargs, drop=torch.zeros(b, dtype=torch.bool))
        null_state = xt.masked_fill(batch["prompt_mask"][..., None], 0)  # the CFG null branch's state
        null = model(null_state, time, **kwargs, drop=torch.ones(b, dtype=torch.bool))
    return xt, null_state, cond, null


@pytest.mark.parametrize("prediction", ["edm", "velocity"])
def test_model_guidance_target_is_the_guided_velocity(prediction):
    model, batch, time, noise = guided_setup(prediction)
    model.eval()
    w = 0.7
    xt, null_state, cond, null = branch_outputs(model, batch, time, noise)
    mask = batch["valid"] & ~batch["prompt_mask"]
    x1 = batch["latents"]
    # Velocity-space target v + w (v_cond - v_null), mapped back to the model's output space.
    guided_velocity = (x1 - noise) + w * (
        to_velocity(model, cond, xt, time) - to_velocity(model, null, null_state, time)
    )
    t = time[:, None, None]
    scale = t.square() + (1 - t).square()
    if prediction == "edm":
        target = scale.sqrt() * (guided_velocity - (2 * t - 1) / scale * xt)
    else:
        target = guided_velocity
    expected = per_example_mse(cond, target, mask)
    loss = flow_loss(model, batch, 0, time, noise, guidance_weight=w)
    assert torch.allclose(loss, expected, rtol=1e-5, atol=1e-6)
    # ... which is the plain target shifted by w (out_cond - out_null) on the target frames.
    plain = ((1 - t) * x1 - t * noise) / scale.sqrt() if prediction == "edm" else x1 - noise
    assert torch.allclose(target[mask], (plain + w * (cond - null))[mask], atol=1e-5)
    assert not torch.allclose(loss, flow_loss(model, batch, 0, time, noise))


def test_model_guidance_leaves_cfg_dropped_rows_on_the_plain_target():
    model, batch, time, noise = guided_setup("edm")
    model.train()
    model.grad_checkpoint = True  # the recipe's setting; the no-grad null pass runs through it
    torch.manual_seed(11)
    guided = flow_loss(model, batch, 0.5, time, noise, return_details=True, guidance_weight=0.7)
    torch.manual_seed(11)
    plain = flow_loss(model, batch, 0.5, time, noise, return_details=True)
    drop = guided["drop"]
    assert torch.equal(drop, plain["drop"]) and drop.any() and not drop.all()
    assert torch.equal(guided["flow"][drop], plain["flow"][drop])
    assert not torch.allclose(guided["flow"][~drop], plain["flow"][~drop])
    torch.manual_seed(11)
    everything = flow_loss(model, batch, 1.0, time, noise, guidance_weight=0.7)
    torch.manual_seed(11)
    assert torch.equal(everything, flow_loss(model, batch, 1.0, time, noise))


def test_guidance_direction_is_detached_and_dropout_free():
    model = randomized(FlowTTS(ModelConfig(**NANO, dropout=0.3))).train()
    batch = nano_batch()
    time = torch.tensor([0.2, 0.55, 0.9])
    xt = batch["latents"] * 0.5
    drop = torch.tensor([False, True, False])
    first = guidance_direction(model, None, xt, time, batch, drop)
    second = guidance_direction(model, None, xt, time, batch, drop)
    assert model.training and not first.requires_grad
    assert torch.equal(first, second)  # eval-mode branches: no dropout noise in the direction
    assert (first[1] == 0).all() and first[0].abs().sum() > 0
    # A gradient step through the guided loss still reaches every generator parameter.
    loss = Objective(model, guidance_weight=0.5).train()(batch)["loss"].mean()
    loss.backward()
    assert all(p.grad is not None for p in model.blocks.parameters())


def test_objective_uses_the_guided_target_only_in_training():
    model, batch, _, _ = guided_setup("edm")
    for mode in ("eval", "train"):
        results = []
        for weight in (0.0, 0.7):
            objective = getattr(Objective(model, guidance_weight=weight), mode)()
            torch.manual_seed(3)
            results.append(objective(batch)["flow"])
        assert torch.equal(*results) == (mode == "eval")


def test_model_guidance_configuration_guards():
    for fields in (dict(model_guidance_weight=1.0), dict(model_guidance_weight=0.5, contrastive_weight=0.2)):
        with pytest.raises(ValueError):
            TrainConfig(**fields)
    with pytest.raises(ValueError):  # the null prediction must be trained
        Config(ModelConfig(cond_dropout=0.0), TrainConfig(model_guidance_weight=0.5))
    Config(ModelConfig(cond_dropout=0.2), TrainConfig(model_guidance_weight=0.7))


def test_model_guidance_fine_tune_is_marked_for_sampling_without_cfg(cache, tmp_path):
    run(config_file(tmp_path, "base.yaml", steps=3), cache, tmp_path / "base")
    tuned = run(config_file(tmp_path, "tune.yaml", steps=2, model_guidance_weight=0.7, ema_decays=[0.9]),
                cache, tmp_path / "tune", init_from=str(tmp_path / "base" / "last.pt"))
    assert tuned["recommended_guidance"] == 1.0 and tuned["step"] == 2 and "ema_0.9" in tuned
    assert all(torch.isfinite(v).all() for v in tuned["model"].values())


# ------------------------------------------------------------------------------------------ example configs


@pytest.mark.parametrize("name", ["tr_w512_wsd", "tr_w512_regularized", "tr_w512_model_guidance_ft"])
def test_example_configs_load(name):
    cfg = Config.load(CONFIGS / "experiments" / f"{name}.yaml")
    # Same architecture as run C, so its checkpoints can warm-start them (dropout may differ).
    assert dataclasses.replace(cfg.model, dropout=0.0) == Config.load(CONFIGS / "nano_tr_w512.yaml").model
