"""Integration of the feature branches (#2-#16, tracking issue #17): the options work together.

Every feature is config-gated and off by default; each branch pins its own "off == main" regression
(test_speed, test_latent_negatives, test_block_options, test_pairs, test_teacher,
test_schedule_regularization). This file covers what no single branch can: options of different
branches combined in one model, objective and training run, plus the combinations that config
validation rejects on purpose.
"""

import torch

from dacvae_tts.config import ModelConfig, TrainConfig
from dacvae_tts.data import collate
from dacvae_tts.model import FlowTTS
from dacvae_tts.speed import training_loader
from dacvae_tts.training import Objective


def tiny_batch(frames=(40, 52, 33), prompts=(6, 9, 5)):
    torch.manual_seed(0)
    items = []
    for index, (length, prompt) in enumerate(zip(frames, prompts)):
        latents = torch.randn(length, 4)
        items.append(
            dict(
                reference=latents[:prompt],
                target=latents[prompt:],
                reference_text="Önceki cümle burada.",
                text=f"Söylenen sözler numara {index} burada.",
            )
        )
    return collate(items)


def test_loader_negatives_are_text_hinge_only():
    """#7 x #8: loader-side text negatives are drawn for the transcript hinge only; latent_delta
    corrupts target latents inside the step, so the loader stays the plain collate."""

    class Rows:
        costs = [3, 4, 5]
        epoch = 0

        def __len__(self):
            return 3

    train = TrainConfig(loader_negatives=True, contrastive_weight=0.2)
    assert training_loader(Rows(), train)[1].negatives
    train = TrainConfig(loader_negatives=True, contrastive_weight=0.2, contrastive_mode="latent_delta")
    items, collate_fn, _ = training_loader(Rows(), train)
    assert collate_fn is collate and isinstance(items, Rows)


def test_latent_delta_without_strict_checks_is_identical():
    """#7 x #8: strict_checks false only drops host-syncing value checks, latent negatives included."""
    cfg = ModelConfig(latent_dim=4, width=32, heads=2, depth=2, text_depth=1)
    results = []
    for strict in (True, False):
        torch.manual_seed(1)
        model = FlowTTS(cfg)
        torch.nn.init.normal_(model.output[-1].weight, std=0.05)
        model.strict_checks = strict
        objective = Objective(model, contrastive_mode="latent_delta").train()
        torch.manual_seed(2)
        results.append(objective(tiny_batch()))
    for key in ("flow", "latent_delta", "negative_random", "negative_aug"):
        assert torch.equal(results[0][key], results[1][key]), key


def test_utmos_flag_serves_protocol_and_eval_sentences():
    """#3 x #13: one `--utmos` switch. Bare it is the protocol's UTMOS22-strong (`utmos` field); with a
    model name (eval_sentences.py's former own flag) it picks that model, UTMOSv2 going to `utmosv2`."""
    import argparse

    from dacvae_tts.eval_protocol import add_protocol_args, protocol_from_args, utmos_models

    parser = argparse.ArgumentParser()
    add_protocol_args(parser)
    assert protocol_from_args(parser.parse_args([])) is None
    options = protocol_from_args(parser.parse_args(["--utmos"]))
    assert options.utmos and not options.utmosv2 and utmos_models(options) == ["utmos22"]
    options = protocol_from_args(parser.parse_args(["--utmos", "utmosv2"]))
    assert options.utmosv2 and not options.utmos and utmos_models(options) == ["utmosv2"]
    options = protocol_from_args(parser.parse_args(["--protocol-v2", "--utmos", "utmosv2"]))
    assert utmos_models(options) == ["utmos22", "utmosv2"]


class _Codec:
    """Codec stand-in: 4 latent channels, 512 samples per frame; accepts #13's decode keywords."""

    def __init__(self, checkpoint, device):
        self.latent_dim, self.sample_rate, self.hop_length = 4, 24000, 512
        self.metadata = dict(checkpoint="test-codec", sample_rate=24000, hop_length=512, latent_dim=4,
                             posterior="mean", weights_sha256="fixture", preprocessing="fixture")

    def decode(self, z, pre_tanh_gain=None, stats=None):
        if stats is not None:
            stats.update(pre_tanh_gain=1.0, pre_tanh_mode=str(pre_tanh_gain), pre_tanh_level=0.5)
        return torch.sin(z.sum(-1).cumsum(0)).repeat_interleave(512) * 0.3


def test_inference_options_combine(monkeypatch, cache, tmp_path):
    """#12 x #13: duration-diverse candidates, the articulation rule, a composite selector, two guidance
    windows and output shaping in one synthesize_many call."""
    import dacvae_tts.inference as module
    from dacvae_tts.config import Config
    from dacvae_tts.data import LatentDataset
    from dacvae_tts.inference import Synthesizer, VoiceReference

    data = LatentDataset(cache)
    config = Config(ModelConfig(latent_dim=4, width=16, depth=1, heads=2, text_depth=1, text_layout="joined",
                                duration="rule", positions="rope", prediction="edm"))
    path = tmp_path / "model.pt"
    model = FlowTTS(config.model)
    torch.save({"model": model.state_dict(), "ema": model.state_dict(), "config": config.to_dict(),
                "codec": {**data.meta, "text_normalization": "turkish-v1"}, "mean": data.mean, "std": data.std}, path)
    monkeypatch.setattr(module, "Codec", _Codec)
    tts = Synthesizer(path, device="cpu", precision="fp32")
    voice = VoiceReference(torch.randn(60, 4), "Referans cümlesi burada, biraz uzun.", "test", {})
    texts = ["Merhaba dünya.", "İkinci cümle biraz daha uzun."]

    def selector(text, audios, sample_rate):
        scores = [dict(dnsmos=float(len(audio))) for audio in audios]  # prefer the longest candidate
        return max(range(len(audios)), key=lambda k: scores[k]["dnsmos"]), scores

    results, metadata = tts.synthesize_many(
        texts, voice, candidates=3, steps=4, guidance=2.0, duration_mode="articulation",
        duration_factors=[1.0, 0.8, 1.25], selector=selector, guidance_split=0.5, apg_eta_late=0.5,
        moment_match="std", pre_tanh_gain="auto",
    )
    assert metadata["candidate_factors"] == [1.0, 0.8, 1.25] and metadata["moment_match"] == "std"
    assert metadata["late_window"]["eta"] == 0.5
    for row, best in zip(results, metadata["selected"]):
        assert [c["duration_factor"] for c in row] == [1.0, 0.8, 1.25]
        assert row[best]["duration_factor"] == 1.25 and "selection" in row[best] and "latent_moments" in row[best]
        assert all(torch.isfinite(torch.as_tensor(c["audio"])).all() for c in row)


BLOCK_OPTIONS = dict(
    long_skip=True,
    value_residual=True,
    ffn_conv_kernel=3,
    attn_gate="head",
    ffn_activation="swiglu",
    final_adaln=True,
    cond_text_pool=True,
)


def block_model(**overrides):
    """A small generator with every #9 block option on and its zero-init parts randomized, so that
    every option changes the output and receives gradients."""
    cfg = dict(latent_dim=4, width=32, heads=2, depth=3, text_depth=1, positions="rope", qk_norm=True,
               adaln_rank=8, ctc_layer=2, **BLOCK_OPTIONS)
    cfg.update(overrides)
    torch.manual_seed(0)
    model = FlowTTS(ModelConfig(**cfg))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.abs().sum():  # zero-init: ada_up, output, skip, gates, ff_conv, final_ada, text_pool
                parameter.normal_(0, 0.05)
            if name.endswith("value_mix"):
                parameter.copy_(torch.tensor([0.8, 0.3]))
    return model


def block_update(model, batch, **options):
    torch.manual_seed(1)
    for key, value in options.items():
        setattr(model, key, value)
    model.zero_grad(set_to_none=True)
    objective = Objective(model, ctc_weight=0.1).train()
    losses = objective(batch)
    (losses["loss"].mean() + objective.auxiliary(losses).mean()).backward()
    return losses["loss"].detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_block_options_under_every_checkpoint_mode():
    """#7 x #9: the speed helper's block runner carries the value-residual tensors through every
    grad_checkpoint mode; losses and all gradients equal the unchecked run."""
    batch = tiny_batch()
    model = block_model()
    reference = block_update(model, batch, grad_checkpoint=False)
    assert {"blocks.1.value_mix", "skip.1.weight", "final_ada.2.weight", "text_pool.weight"} <= set(reference[1])
    for mode in (True, "selective", 2):
        loss, grads = block_update(model, batch, grad_checkpoint=mode)
        assert torch.equal(loss, reference[0]), mode
        assert grads.keys() == reference[1].keys()
        for name in grads:
            assert torch.allclose(grads[name], reference[1][name], rtol=1e-5, atol=1e-7), (mode, name)


def test_compiled_blocks_with_value_residual_match_eager():
    """#7 x #9: `compile: blocks` compiles run_block, whose block call now also takes and returns the
    first block's values."""
    import pytest

    from dacvae_tts.speed import compile_blocks

    try:
        torch._dynamo.reset()
        torch.compile(lambda x: x * 2 + 1)(torch.ones(3))
    except Exception:
        pytest.skip("torch.compile has no working backend here")
    batch = tiny_batch()
    eager, compiled = block_model(), block_model()
    compile_blocks(compiled, "batch")
    try:
        results = [block_update(model, batch, grad_checkpoint="selective") for model in (eager, compiled)]
        assert torch.allclose(results[0][0], results[1][0], rtol=1e-4, atol=1e-5)
        for name, grad in results[0][1].items():
            assert torch.allclose(grad, results[1][1][name], rtol=1e-4, atol=1e-5), name
    finally:
        torch._dynamo.reset()


def test_dropout_with_block_options():
    """#14 x #9: model.dropout installs on the SwiGLU and conv-GELU FFNs and on value-residual/gated
    attention without new parameters; it is inactive in eval mode, active in training, and every
    checkpoint mode reproduces the unchecked step (the RNG state is replayed on recomputation)."""
    batch = tiny_batch()
    for ffn in ("swiglu", "gelu"):
        plain, dropped = block_model(ffn_activation=ffn), block_model(ffn_activation=ffn, dropout=0.3)
        assert plain.state_dict().keys() == dropped.state_dict().keys()
        dropped.load_state_dict(plain.state_dict())
        args = (batch["latents"], torch.full((3,), 0.4), batch["prompt"], batch["prompt_mask"], batch["valid"],
                batch["tokens"], batch["segments"])
        with torch.no_grad():
            assert torch.equal(plain.eval()(*args), dropped.eval()(*args))
            dropped.train()
            torch.manual_seed(3)
            first = dropped(*args)
            torch.manual_seed(4)
            assert not torch.equal(first, dropped(*args))
        reference = block_update(dropped, batch, grad_checkpoint=False)
        assert all(torch.isfinite(g).all() for g in reference[1].values())
        for mode in (True, "selective"):
            loss, grads = block_update(dropped, batch, grad_checkpoint=mode)
            assert torch.allclose(loss, reference[0], rtol=1e-6, atol=1e-7), (ffn, mode)
            for name in grads:
                assert torch.allclose(grads[name], reference[1][name], rtol=1e-5, atol=1e-6), (ffn, mode, name)


def write_silence(cache, raw=(0.125, -0.375, 0.625, -0.875)):
    """<cache>/silence.pt in the format of scripts/silence_latent.py (#11)."""
    raw = torch.tensor(raw)
    torch.save({"raw": raw, "frame": torch.full((4,), 9.0), "codec": {"checkpoint": "test-codec"}},
               cache / "silence.pt")
    return raw


def test_latent_negatives_pad_with_the_pipeline_silence(cache):
    """#8 x #11: latent negatives read the silence.pt of scripts/silence_latent.py and pad with the
    same standardized frame the data pipeline appends as tail silence."""
    from dacvae_tts.data import LatentDataset
    from dacvae_tts.negatives import load_silence

    raw = write_silence(cache)
    data = LatentDataset(cache, "train", pairing="within", layout="joined", tail_silence_prob=0.5)
    fill = load_silence(cache, data.channels, data.mean, data.std, data.meta)
    assert torch.equal(fill, data.silence) and torch.allclose(fill, (raw - data.mean) / data.std)
    assert torch.equal(load_silence(cache, 4), torch.full((4,), 9.0))  # no statistics: the stored frame


def test_padded_epoch_costs_keep_the_frame_budget(cache):
    """#7 x #11: cross-prompt epoch costs are rounded up to pad_multiple like the static costs, so the
    padded batches of every epoch stay within the frame budget."""
    import numpy as np

    from dacvae_tts.data import BucketBatchSampler, LatentDataset
    from dacvae_tts.speed import padded_costs, training_epoch_costs

    pairs = dict(cross_prompt_prob=1.0, cross_prompt_max_seconds=0.5)  # up to 23 frames of other utterances
    train = TrainConfig(pairing="within", pad_multiple=8, **pairs)
    data = LatentDataset(cache, "train", pairing="within", layout="joined", **pairs)
    epoch_costs = training_epoch_costs(data, train)
    assert training_epoch_costs(data, TrainConfig()) is None
    static = padded_costs(data.costs, 8)
    for epoch in (0, 1):
        exact, padded = data.epoch_costs(epoch), epoch_costs(epoch)
        assert np.all(padded % 8 == 0) and np.all(padded >= exact) and np.all(padded <= static)
        assert np.any(exact > data.lengths)  # cross prompts are in use
        sampler = BucketBatchSampler(static, 4, frame_budget=80, epoch_costs=epoch_costs)
        sampler.epoch = epoch
        for batch in sampler.batches():
            assert max(padded[i] for i in batch) * len(batch) <= 80


def teacher_dataset(cache, **pairs):
    """Within-pairing dataset of the fixture cache with #10's fake teacher stores (test_teacher.py)."""
    from test_teacher import build_stores

    from dacvae_tts.data import LatentDataset

    frames, speakers = build_stores(cache)
    return LatentDataset(cache, "train", pairing="within", layout="joined", teacher_features=frames,
                         speaker_embeddings=speakers, **pairs)


def test_teacher_frames_follow_cross_prompts_tail_silence_and_padding(cache):
    """#10 x #11 x #7: teacher frames are sliced like the latents of cross-utterance prompts, extended
    (and masked out of REPA) over appended tail silence, and padded with the frames by pad_lengths."""
    from dacvae_tts.speed import pad_lengths

    write_silence(cache)
    data = teacher_dataset(cache, cross_prompt_prob=1.0, cross_prompt_max_seconds=0.5, tail_silence_prob=1.0,
                           prompt_cut="quiet")
    items = [data[(0, index)] for index in range(len(data))]
    crossed = [(index, item) for index, item in enumerate(items) if "|" in item["reference_uid"]]
    assert crossed and len(crossed) < len(items)  # both cross prompts and within cuts occur
    index, item = crossed[0]
    references = data.cross_plan(0, index)
    assert torch.equal(item["teacher_reference"], torch.cat([data.row(r)["teacher"] for r in references]))
    batch = collate(items)
    lengths = [len(item["reference"]) + len(item["target"]) for item in items]
    real = [length - item["tail_silence"] for length, item in zip(lengths, items)]
    assert batch["teacher"].shape[:2] == batch["valid"].shape
    assert batch["teacher_valid"].sum(1).tolist() == real and any(item["tail_silence"] for item in items)
    padded = pad_lengths(batch, 8, 8)
    assert padded["teacher"].shape[1] % 8 == 0 and padded["teacher_valid"].shape == padded["valid"].shape
    cfg = ModelConfig(latent_dim=4, width=16, heads=2, depth=2, text_depth=1, text_layout="joined",
                      duration="rule", repa_layer=2, repa_dim=3, tla_layers="all", tla_dim=3, tla_hidden=8)
    objective = Objective(FlowTTS(cfg), repa_weight=1.0, tla_weight=0.5).train()
    losses = objective(padded)
    assert torch.isfinite(losses["repa"]).all() and torch.isfinite(losses["tla"]).all()


def test_model_guidance_composes_with_latent_negatives_not_the_text_hinge():
    """#14 x #8: the text hinge stays rejected with model guidance; latent_delta (and none) are allowed,
    and latent negatives shift their targets by the same guidance offset as the positive target."""
    import pytest

    from dacvae_tts.contracts import target_mask
    from dacvae_tts.model import flow_loss, flow_target, per_example_mse
    from dacvae_tts.negatives import random_negatives

    with pytest.raises(ValueError, match="text_hinge"):
        TrainConfig(model_guidance_weight=0.5, contrastive_weight=0.2)
    TrainConfig(model_guidance_weight=0.5, contrastive_weight=0.2, contrastive_mode="latent_delta")
    TrainConfig(model_guidance_weight=0.5, contrastive_weight=0.2, contrastive_mode="none")

    batch = tiny_batch()
    model = block_model(ffn_activation="gelu", value_residual=False, long_skip=False)
    objective = Objective(model, contrastive_mode="latent_delta", aug_weight=0.0, guidance_weight=0.5).train()
    torch.manual_seed(5)
    losses = objective(batch)
    torch.manual_seed(5)
    cached = model.conditions(batch["prompt"], batch["prompt_mask"], batch["tokens"], batch["segments"])
    details = flow_loss(model, batch, model.cfg.cond_dropout, return_details=True, cached=cached,
                        guidance_weight=0.5)
    offset, prediction = details["guidance_offset"], details["prediction"]
    assert offset.abs().sum() > 0 and torch.equal(losses["flow"], details["flow"])
    negative, usable = random_negatives(batch["latents"], batch["valid"], batch["prompt_mask"])
    mask = target_mask(batch["valid"], batch["prompt_mask"])
    target = flow_target(model, negative, details["noise"], details["times"]) + offset
    expected = per_example_mse(prediction, target, mask).masked_fill(details["drop"] | ~usable, 0)
    assert torch.allclose(losses["negative_random"], expected, rtol=1e-5, atol=1e-6)
    assert torch.allclose(losses["latent_delta"], -0.2 * expected, rtol=1e-5, atol=1e-6)


def test_decay_phase_loader_keeps_every_data_option(cache, tmp_path):
    """#14 x #7/#10/#11: the WSD decay-cache loader is built like the main one (pair options, teacher
    stores, padding, char CTC labels), normalized with the main statistics, silence frame included."""
    import shutil

    from dacvae_tts.config import Config
    from dacvae_tts.data import LatentDataset, save_stats
    from dacvae_tts.speed import TrainCollate
    from dacvae_tts.training import decay_phase_loader

    raw = write_silence(cache)
    decay = tmp_path / "decay"
    shutil.copytree(cache, decay)
    save_stats(decay / "stats.pt", 10, torch.full((4,), 5.0).double(), torch.full((4,), 40.0).double())
    cfg = Config(
        ModelConfig(latent_dim=4, width=16, heads=2, depth=2, text_depth=1, text_layout="joined", duration="rule",
                    ctc_layer=1, ctc_targets="chars"),
        TrainConfig(steps=10, warmup=1, batch_size=4, workers=0, pairing="within", cross_prompt_prob=0.5,
                    cross_prompt_max_seconds=0.5, tail_silence_prob=1.0, pad_multiple=8, text_pad_multiple=8,
                    lr_schedule="wsd", decay_cache=str(decay)),
    )
    main = LatentDataset(cache, "train", pairing="within", layout="joined", tail_silence_prob=1.0)
    sampler, loader = decay_phase_loader(str(decay), main, cfg, 0, 1, 0, torch.device("cpu"))
    data = loader.dataset
    assert torch.equal(data.mean, main.mean) and torch.equal(data.silence, (raw - main.mean) / main.std)
    assert data.cross_prompt_prob == 0.5 and sampler.epoch_costs is not None
    assert isinstance(loader.collate_fn, TrainCollate)
    batch = next(iter(loader))
    assert batch["valid"].size(1) % 8 == 0 and batch["tokens"].size(1) % 8 == 0 and "ctc_targets" in batch


def test_grpo_optimizes_a_dropout_checkpoint_on_policy(cache, tmp_path, monkeypatch):
    """#16 x #14 x #9: GRPO on a checkpoint trained with model.dropout and block options switches the
    dropout off, so the train-mode gradient pass reproduces the eval-mode rollout policy (ratio 1)."""
    import json
    import sys

    from dacvae_tts import cli, grpo
    from dacvae_tts.config import Config
    from dacvae_tts.data import LatentDataset
    from dacvae_tts.grpo import CompositeReward

    data = LatentDataset(cache)
    cfg = Config(ModelConfig(latent_dim=4, width=16, depth=2, heads=2, text_depth=1, dropout=0.3,
                             ffn_activation="swiglu", value_residual=True, attn_gate="head"), TrainConfig())
    model = FlowTTS(cfg.model)
    with torch.no_grad():  # a "trained" model: zero-init gates and output layers would hide the dropout
        for parameter in model.parameters():
            if not parameter.abs().sum():
                parameter.normal_(0, 0.1)
    state = model.state_dict()
    checkpoint = tmp_path / "dropout.pt"
    torch.save({"model": state, "ema": state, "config": cfg.to_dict(), "codec": data.meta, "mean": data.mean,
                "std": data.std}, checkpoint)
    captured = {}
    run = grpo.grpo_train
    monkeypatch.setattr(grpo, "grpo_train", lambda args: captured.setdefault("args", args))
    monkeypatch.setattr(sys, "argv", [
        "dacvae-tts", "post-train", "--mode", "grpo", "--checkpoint", str(checkpoint), "--cache", str(cache),
        "--output", str(tmp_path / "grpo"), "--device", "cpu", "--precision", "fp32", "--steps", "2",
        "--sample-steps", "4", "--guidance", "2", "--window-max", "1", "--group-size", "4",
        "--prompts-per-step", "1", "--duration-mode", "rule", "--monitor-every", "0",
        "--metric-normalization", "turkish-v2",
    ])
    cli.main()

    class LatentMean:
        def __call__(self, group):
            return [float(z.mean()) for z in group.latents]

    run(captured["args"], reward=CompositeReward({"mean": LatentMean()}, {"mean": 1.0}, min_terms=1))
    records = [json.loads(line) for line in (tmp_path / "grpo" / "grpo-log.jsonl").read_text().splitlines()]
    steps = [record for record in records if "reward" in record and record.get("optimizer_steps")]
    assert steps and all(r["clip_fraction"] == 0 and abs(r["approx_kl"]) < 1e-6 for r in steps)


# ------------------------------------------------------------------------------------------------------------
# (a) Every option off: the model, the configuration and the objective of main (8f01f08).

# The configuration fields of main 8f01f08: checkpoints written there carry exactly these keys.
MAIN_MODEL_FIELDS = (
    "latent_dim", "width", "depth", "heads", "text_depth", "patch_size", "ff_mult", "cond_dropout",
    "reference_encoder", "reference_pooling", "reference_paths", "duration_features", "positions", "qk_norm",
    "text_attention", "prediction", "text_layout", "duration", "ctc_layer", "adaln_rank",
)
MAIN_TRAIN_FIELDS = (
    "steps", "batch_size", "accumulation", "learning_rate", "warmup", "weight_decay", "optimizer", "muon_momentum",
    "ema_decay", "precision", "workers", "worker_threads", "prefetch_factor", "loader_start_method", "cuda_prefetch",
    "checkpoint_every", "validate_every", "log_every", "grad_checkpoint", "compile", "seed", "flow_reduction",
    "duration_weight", "speaker_balance", "diagnostics_every", "pairing", "prompt_fraction_min",
    "prompt_fraction_max", "prompt_dropout", "time_sampling", "batch_expansion", "keep_every", "ctc_weight",
    "contrastive_weight", "contrastive_margin", "wandb_project",
)
# Every model option added by the branches, explicitly off (#9 block options, #11, #10, #14).
MODEL_OFF = dict(
    long_skip=False, value_residual=False, ffn_conv_kernel=0, attn_gate="none", ffn_activation="gelu",
    final_adaln=False, cond_text_pool=False, ctc_targets="bytes", repa_layer=0, repa_dim=0, tla_layers=(),
    tla_dim=0, tla_hidden=256, dropout=0.0,
)
# Every training option added by the branches, explicitly off (#7, #8, #11, #10, #14).
TRAIN_OFF = dict(
    strict_checks=True, pad_multiple=1, text_pad_multiple=1, loader_negatives=False, compile_dynamic="batch",
    contrastive_mode="text_hinge", contrastive_random_weight=0.2, contrastive_aug_weight=0.2,
    contrastive_span_min=3, contrastive_span_max=125, contrastive_repeat_coverage=(0.2, 0.4),
    contrastive_skip_coverage=(0.4, 0.8), contrastive_negative_cap=0.0, cross_prompt_prob=0.0,
    cross_prompt_max_utterances=3, cross_prompt_max_seconds=12.0, long_prompt_prob=0.0,
    prompt_fraction_long_max=0.85, tail_silence_prob=0.0, tail_silence_max_seconds=0.8, prompt_cut="random",
    teacher_features="", repa_weight=0.0, repa_stop_step=0, repa_frames="all", speaker_embeddings="",
    tla_weight=0.0, tla_entropy=0.01, lr_schedule="cosine", decay_fraction=0.2, decay_shape="1-sqrt",
    min_lr_ratio=0.1, decay_cache=None, final_time_sampling=None, final_time_sampling_start="decay",
    ema_decays=[], ema_warmup=True, model_guidance_weight=0.0,
)


def test_new_options_are_exactly_the_listed_ones_and_default_off():
    """The integrated configuration is main's plus the branch options above, all of which default to off,
    so a main checkpoint's configuration loads, resumes (config equality) and warm starts unchanged."""
    import dataclasses

    from dacvae_tts.config import Config

    model_fields = {f.name for f in dataclasses.fields(ModelConfig)}
    train_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    assert model_fields == set(MAIN_MODEL_FIELDS) | set(MODEL_OFF)
    assert train_fields == set(MAIN_TRAIN_FIELDS) | set(TRAIN_OFF)
    assert ModelConfig(**MODEL_OFF) == ModelConfig() and TrainConfig(**TRAIN_OFF) == TrainConfig()
    defaults = Config().to_dict()
    old = {
        "model": {key: defaults["model"][key] for key in MAIN_MODEL_FIELDS},
        "train": {key: defaults["train"][key] for key in MAIN_TRAIN_FIELDS},
    }
    assert Config.from_dict(old) == Config() and Config.from_dict(old).to_dict() == defaults


def test_all_options_off_is_the_main_model():
    """Reuses test_block_options' pins captured on 8f01f08 (state_dict layout hash and an RNG-free forward
    fingerprint) with every option of every branch explicitly off."""
    import pytest
    from test_block_options import PREVIOUS, fingerprint, layout_hash

    for overrides, keys, layout, expected in PREVIOUS:
        model = FlowTTS(ModelConfig(**overrides, **MODEL_OFF))
        assert len(model.state_dict()) == keys and layout_hash(model) == layout
        assert fingerprint(model) == pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_all_options_off_objective_is_the_main_objective():
    """Objective built with every branch argument at its off value (as train() builds it) equals the verbatim
    main 8f01f08 Objective.forward kept by test_latent_negatives, outputs and gradients, bit for bit."""
    from test_latent_negatives import NANO, batch_of, legacy_forward, randomized

    model = randomized(FlowTTS(ModelConfig(**NANO, **MODEL_OFF))).train()
    batch = batch_of("joined")
    options = dict(
        expansion=2, ctc_weight=0.1, contrastive_weight=0.2, contrastive_margin=0.1, contrastive_mode="text_hinge",
        random_weight=0.2, aug_weight=0.2, span=(3, 125), repeat_coverage=(0.2, 0.4), skip_coverage=(0.4, 0.8),
        negative_cap=0.0, silence=None, repa_weight=0.0, repa_frames="all", tla_weight=0.0, tla_entropy=0.01,
        guidance_weight=0.0,
    )
    results = []
    for build in (lambda: legacy_forward(Objective(model, **options).train(), batch),
                  lambda: Objective(model, **options).train()(batch)):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(11)
        losses = build()
        (losses["loss"].sum() + 0.2 * losses["contrastive"].sum() + 0.1 * losses["ctc"].sum()).backward()
        results.append((losses, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
    (reference, reference_grads), (losses, grads) = results
    assert losses.keys() == reference.keys() and grads.keys() == reference_grads.keys()
    for key in reference:
        assert torch.equal(losses[key], reference[key]), key
    for key in reference_grads:
        assert torch.equal(grads[key], reference_grads[key]), key


# ------------------------------------------------------------------------------------------------------------
# (b) A broad, compatible combination of the options of every branch, through the objective and through train().

COMBINED_MODEL = dict(
    latent_dim=4, width=32, heads=2, depth=3, text_depth=1, text_attention=1, positions="rope", qk_norm=True,
    prediction="edm", text_layout="joined", duration="rule", ctc_layer=1, adaln_rank=8,
    long_skip=True, value_residual=True, ffn_conv_kernel=3, attn_gate="head", ffn_activation="swiglu",
    final_adaln=True, cond_text_pool=True,                                   # #9
    ctc_targets="chars",                                                      # #11
    repa_layer=3, repa_dim=3, tla_layers="all", tla_dim=3, tla_hidden=8,     # #10 (fake stores: 3-d)
    dropout=0.1,                                                              # #14
)
COMBINED_PAIRS = dict(cross_prompt_prob=0.5, cross_prompt_max_seconds=0.5, long_prompt_prob=0.3,
                      tail_silence_prob=0.5, prompt_cut="quiet")             # #11


def test_every_enabled_head_receives_gradients(cache):
    """One objective step with #9 block options, #10 REPA + TLA-SA, #11 char CTC and pair data, #8 latent_delta,
    #14 dropout and model guidance, and #7 padding/selective checkpointing: finite, and every parameter (the
    zero-init ones randomized, as after some training) gets a nonzero gradient."""
    from dacvae_tts.speed import TrainCollate

    write_silence(cache)
    # The fixture's 7-11 frame utterances are shorter than their 15-letter CTC targets; up to 1 s of tail silence
    # makes most rows CTC-feasible, so the character head receives a gradient.
    pairs = dict(COMBINED_PAIRS, tail_silence_prob=1.0, tail_silence_max_seconds=1.0)
    data = teacher_dataset(cache, ctc_targets="chars", **pairs)
    batch = TrainCollate(8, 8)([data[(0, index)] for index in range(6)])
    assert {"teacher", "teacher_valid", "speaker_embedding", "ctc_targets"} <= batch.keys()
    torch.manual_seed(0)
    model = FlowTTS(ModelConfig(**COMBINED_MODEL))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.abs().sum():
                parameter.normal_(0, 0.05)
            if name.endswith("value_mix"):
                parameter.copy_(torch.tensor([0.8, 0.3]))
    model.grad_checkpoint, model.strict_checks = "selective", False
    objective = Objective(model, expansion=2, ctc_weight=0.1, contrastive_mode="latent_delta", silence=data.silence,
                          repa_weight=1.0, tla_weight=0.5, guidance_weight=0.5).train()
    torch.manual_seed(1)
    losses = objective(batch)
    for key in ("flow", "ctc", "latent_delta", "negative_random", "negative_aug", "repa", "tla", "tla_entropy"):
        assert torch.isfinite(losses[key]).all() and losses[key].abs().sum() > 0, key
    total = losses["loss"].mean() + objective.auxiliary(losses).mean() + losses["latent_delta"].mean()
    total.backward()
    silent = [name for name, p in model.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert not silent, silent
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_combined_options_train_end_to_end(cache, tmp_path):
    """train() on CPU with the options of every training-side branch at once, including the WSD decay switch
    to a second cache: finite logs with every term, validation of both EMA tracks, and checkpoints that hold the
    teacher heads and the extra EMA track and reload."""
    import json
    import math
    import shutil
    import types

    import yaml
    from test_teacher import build_stores

    from dacvae_tts import training
    from dacvae_tts.training import load_model

    write_silence(cache)
    build_stores(cache)
    decay = tmp_path / "decay"
    shutil.copytree(cache, decay)  # the decay cache brings its own silence.pt and teacher stores
    train = dict(
        steps=4, warmup=1, batch_size=2, accumulation=2, workers=0, precision="fp32", log_every=1,
        checkpoint_every=2, validate_every=2, diagnostics_every=2, pairing="within", prompt_dropout=0.2,
        time_sampling="logit_normal", batch_expansion=2, ctc_weight=0.1, contrastive_weight=0.2,
        strict_checks=False, pad_multiple=8, text_pad_multiple=8, loader_negatives=True,
        grad_checkpoint="selective",                                                             # #7
        contrastive_mode="latent_delta",                                                         # #8
        **COMBINED_PAIRS,                                                                        # #11
        teacher_features="teacher/frames", repa_weight=1.0, speaker_embeddings="teacher/speakers",
        tla_weight=0.5,                                                                          # #10
        lr_schedule="wsd", decay_fraction=0.5, min_lr_ratio=0.0, decay_cache=str(decay),
        final_time_sampling="uniform", ema_decays=[0.9, 0.5],                                   # #14
    )
    config = tmp_path / "combined.yaml"
    config.write_text(yaml.safe_dump({"model": dict(COMBINED_MODEL, tla_layers="all"), "train": train}))
    names = "steps batch_size accumulation workers precision learning_rate optimizer worker_threads".split()
    names += "prefetch_factor loader_start_method cuda_prefetch compile stop_after init_from resume".split()
    args = types.SimpleNamespace(**dict.fromkeys(names), config=str(config), cache=str(cache),
                                 output=str(tmp_path / "run"), device="cpu", frame_budget=0, no_validation=False)
    training.train(args)

    records = [json.loads(line) for line in (tmp_path / "run" / "train.jsonl").read_text().splitlines()]
    steps = [r for r in records if "flow" in r]
    assert [r["step"] for r in steps] == [1, 2, 3, 4]
    for record in steps:
        for key in ("loss", "flow", "ctc", "latent_delta", "negative_random", "repa", "tla", "tla_entropy"):
            assert math.isfinite(record[key]), (record["step"], key)
        assert record["repa"] > 0 and record["tla"] > 0 and record["latent_delta"] < 0
    assert steps[-1]["lr"] < steps[1]["lr"]  # WSD decay over the last half
    validations = [r for r in records if "validation_loss" in r]
    assert [r["step"] for r in validations] == [2, 4]
    assert all("ema_0_9/validation_flow" in r and "ema_0_5/validation_flow" in r for r in validations)
    assert all(math.isfinite(r["validation_flow"]) for r in validations)
    model, saved = load_model(tmp_path / "run" / "last.pt", ema=0.9)
    assert {"ema_0.9", "ema_0.5"} <= saved.keys() and saved["step"] == 4
    assert any(k.startswith("repa.") for k in saved["model"]) and any(k.startswith("tla.") for k in saved["ema"])
    assert model.cfg.ctc_targets == "chars" and model.cfg.dropout == 0.1


def test_combined_experiment_config():
    """configs/experiments/tr_w512_combined.yaml is run C plus the evidence-backed options only: the #9 block
    options and #14's dropout/model guidance stay at their off values."""
    from pathlib import Path

    import yaml

    from dacvae_tts.config import Config

    configs = Path(__file__).resolve().parents[1] / "configs"
    combined = Config.load(configs / "experiments" / "tr_w512_combined.yaml")
    base = Config.load(configs / "nano_tr_w512.yaml")
    changed = {
        (section, key)
        for section in ("model", "train")
        for key, value in combined.to_dict()[section].items()
        if base.to_dict()[section][key] != value
    }
    assert changed == {
        ("model", key) for key in ("ctc_targets", "repa_layer", "repa_dim", "tla_layers", "tla_dim")
    } | {
        ("train", key) for key in (
            "grad_checkpoint", "compile", "strict_checks", "pad_multiple", "text_pad_multiple", "loader_negatives",
            "contrastive_mode", "cross_prompt_prob", "long_prompt_prob", "tail_silence_prob", "prompt_cut",
            "teacher_features", "repa_weight", "speaker_embeddings", "tla_weight", "lr_schedule", "min_lr_ratio",
            "final_time_sampling", "ema_decays",
        )
    }
    raw = yaml.safe_load((configs / "experiments" / "tr_w512_combined.yaml").read_text())
    assert not set(raw["model"]) & {"long_skip", "value_residual", "ffn_conv_kernel", "attn_gate", "ffn_activation",
                                    "final_adaln", "cond_text_pool", "dropout"}
    assert "model_guidance_weight" not in raw["train"] and "decay_cache" not in raw["train"]
