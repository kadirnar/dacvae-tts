"""Training schedule and regularization options of issue #14; every default keeps the original recipe."""

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
from dacvae_tts.config import Config, TrainConfig
from dacvae_tts.data import LatentDataset
from dacvae_tts.training import (
    decay_phase_loader,
    decay_start,
    load_model,
    lr_multiplier,
    schedule_multiplier,
    time_sampling_at,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
# Configuration keys added for issue #14; configurations saved before it lack them.
NEW_FIELDS = {
    "model": (),
    "train": (
        "lr_schedule", "decay_fraction", "decay_shape", "min_lr_ratio", "decay_cache", "final_time_sampling",
        "final_time_sampling_start",
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
    run(config, cache, tmp_path / "old", stop_after=2)
    path = tmp_path / "old" / "last.pt"
    saved = torch.load(path, weights_only=True)
    saved["config"] = as_before_issue_14(saved["config"])
    torch.save(saved, path)
    for ema in (True, False):
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
    """WSD + decay cache + uniform-t cooldown, interrupted before, exactly at and after the decay start
    (update 3 of 6), must equal the uninterrupted run."""
    calls = []
    original = training.flow_loss

    def spy(*args, **kwargs):
        calls.append(kwargs["time_sampling"])
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "flow_loss", spy)
    config = config_file(
        tmp_path, lr_schedule="wsd", decay_fraction=0.5, decay_shape="1-sqrt",
        min_lr_ratio=0.0, decay_cache=str(decay_cache), time_sampling="logit_normal",
        final_time_sampling="uniform",
    )
    full = run(config, cache, tmp_path / "full", no_validation=True)
    assert calls == ["logit_normal"] * 6 + ["uniform"] * 6  # two micro-batches per update
    assert full["batch_offset"] == 3  # 6 decay-phase batches from a 3-batch subset: the switch happened
    records = [json.loads(line) for line in (tmp_path / "full" / "train.jsonl").read_text().splitlines()]
    decay = [1.0, 1 - (1 / 3) ** 0.5, 1 - (2 / 3) ** 0.5]
    assert [r["lr"] for r in records] == pytest.approx([3e-4 * m for m in [1.0, 1.0, 1.0, *decay]])
    output = tmp_path / "resumed"
    run(config, cache, output, no_validation=True, stop_after=2)
    for stop in (3, 4, None):
        resumed = run(config, cache, output, no_validation=True, resume=str(output / "last.pt"), stop_after=stop)
    assert resumed["step"] == 6
    for key in full["model"]:
        for weights in ("model", "ema"):
            assert torch.equal(full[weights][key], resumed[weights][key]), (weights, key)
