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
