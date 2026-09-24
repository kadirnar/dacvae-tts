"""Training throughput options (speed.py): off they are bit-identical, on they keep the objective."""

import dataclasses
import json
import os
import random
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from dacvae_tts import model as model_module
from dacvae_tts.config import Config, ModelConfig, TrainConfig
from dacvae_tts.contracts import mask_values
from dacvae_tts.data import BucketBatchSampler, collate
from dacvae_tts.model import FlowTTS, ctc_alignment_loss, flow_loss, per_example_mse
from dacvae_tts.speed import (
    NegativeSeeds,
    NonfiniteWatch,
    TrainCollate,
    block_checkpoint,
    compile_blocks,
    corrupt_rows,
    pad_lengths,
    padded_costs,
    training_loader,
)
from dacvae_tts.text import BYTE_OFFSET, corrupt_transcript
from dacvae_tts.training import Objective, load_model

SPEED_OPTIONS = {
    "grad_checkpoint",
    "compile",
    "compile_dynamic",
    "strict_checks",
    "pad_multiple",
    "text_pad_multiple",
    "loader_negatives",
}


def legacy_ctc_alignment_loss(logits, token_valid, tokens, drop):
    """`ctc_alignment_loss` before the on-device targets (Python lists), kept as the reference."""
    targets = [row[row >= BYTE_OFFSET] for row in tokens]
    lengths = torch.tensor([len(row) for row in targets], device=logits.device)
    loss = F.ctc_loss(
        logits.float().log_softmax(-1).transpose(0, 1),
        torch.cat(targets),
        token_valid.sum(1),
        lengths,
        blank=0,
        reduction="none",
        zero_infinity=True,
    )
    return (loss / lengths.clamp_min(1)).masked_fill(drop, 0)


def items(layout="joined"):
    """Three utterances long enough for a feasible CTC alignment of their transcripts."""
    shapes = [(10, 30), (8, 25), (0, 36)] if layout == "joined" else [(10, 30), (8, 25), (5, 31)]
    texts = ["alpha beta gamma", "delta epsilon", "zeta eta theta iota"]
    references = ["", "", ""] if layout == "joined" else ["Hello there.", "One two.", "Why not?"]
    return [
        dict(
            reference=torch.randn(r, 4),
            target=torch.randn(t, 4),
            reference_text=ref,
            text=text,
            layout=layout,
        )
        for (r, t), text, ref in zip(shapes, texts, references)
    ]


def build(layout="joined", patch_size=1):
    """A small model with every generator path used in the w512 recipe (and the duration head)."""
    torch.manual_seed(0)
    cfg = ModelConfig(
        latent_dim=4,
        width=32,
        depth=3,
        heads=2,
        text_depth=1,
        text_attention=1,
        patch_size=patch_size,
        positions="rope",
        qk_norm=True,
        prediction="edm",
        text_layout=layout,
        duration="rule" if layout == "joined" else "head",
        ctc_layer=2,
        adaln_rank=8,
    )
    model = FlowTTS(cfg)
    for block in model.blocks:  # the AdaLN up-projections and the output start at zero
        nn.init.normal_(block.ada_up.weight, std=0.05)
    nn.init.normal_(model.output[-1].weight, std=0.05)
    return model, collate(items(layout))


def update(model, batch, **options):
    """Losses and gradients of one training objective evaluation with fixed random draws."""
    torch.manual_seed(1)
    objective = Objective(model, 0.1, "logit_normal", 2, 0.1, 0.2, 0.1).train()
    model.zero_grad(set_to_none=True)
    for key, value in options.items():
        setattr(model, key, value)
    losses = objective(batch)
    (losses["loss"].mean() + objective.auxiliary(losses).mean()).backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    return {key: value.detach().clone() for key, value in losses.items()}, grads


def assert_identical(first, second):
    (losses_a, grads_a), (losses_b, grads_b) = first, second
    assert losses_a.keys() == losses_b.keys() and grads_a.keys() == grads_b.keys()
    for key in losses_a:
        assert torch.equal(losses_a[key], losses_b[key]), key
    for key in grads_a:
        assert torch.equal(grads_a[key], grads_b[key]), key


def training_args(config, cache, output, **overrides):
    values = dict(
        config=str(config),
        cache=str(cache),
        output=str(output),
        resume=None,
        init_from=None,
        device="cpu",
        steps=None,
        batch_size=None,
        accumulation=None,
        workers=None,
        precision=None,
        learning_rate=None,
        optimizer=None,
        worker_threads=None,
        prefetch_factor=None,
        loader_start_method=None,
        cuda_prefetch=None,
        frame_budget=0,
        compile=None,
        no_validation=True,
        stop_after=None,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def speed_config(tmp_path, **train):
    """Within-utterance recipe (joined text, CTC, contrastive negatives) without value checks."""
    path = tmp_path / "speed.yaml"
    options = dict(
        steps=4,
        warmup=1,
        batch_size=2,
        accumulation=2,
        workers=0,
        precision="fp32",
        log_every=1,
        checkpoint_every=2,
        validate_every=2,
        pairing="within",
        prompt_dropout=0.3,
        batch_expansion=2,
        ctc_weight=0.1,
        contrastive_weight=0.2,
        strict_checks=False,
    )
    options.update(train)
    model = dict(
        latent_dim=4, width=16, depth=2, heads=2, text_depth=1, text_layout="joined", duration="rule"
    )
    model.update(ctc_layer=1, positions="rope", adaln_rank=4)
    path.write_text(yaml.safe_dump({"model": model, "train": options}))
    return path


@pytest.mark.parametrize("layout", ["joined", "segments"])
def test_ctc_targets_on_device_match_python_lists(layout):
    model, batch = build(layout)
    tokens = batch["tokens"]
    logits = torch.randn(tokens.size(0), 40, 260, requires_grad=True)
    token_valid = torch.arange(40)[None] < torch.tensor([40, 33, 36])[:, None]
    drop = torch.tensor([False, True, False])
    new = ctc_alignment_loss(logits, token_valid, tokens, drop)
    (new_grad,) = torch.autograd.grad(new.sum(), logits)
    old = legacy_ctc_alignment_loss(logits, token_valid, tokens, drop)
    (old_grad,) = torch.autograd.grad(old.sum(), logits)
    assert torch.equal(new, old) and torch.equal(new_grad, old_grad)
    assert (new[~drop] > 0).all()  # a real alignment, not the zero of an infeasible one


@pytest.mark.parametrize("layout", ["joined", "segments"])
def test_default_options_are_bit_identical_to_the_previous_step(layout, monkeypatch):
    model, batch = build(layout)
    model.grad_checkpoint = True
    current = update(model, batch)
    assert current[0]["ctc"].abs().sum() > 0 and "contrastive" in current[0]
    monkeypatch.setattr(model_module, "ctc_alignment_loss", legacy_ctc_alignment_loss)
    assert_identical(current, update(model, batch))


@pytest.mark.parametrize("layout", ["joined", "segments"])
def test_strict_checks_off_keeps_numerics(layout):
    model, batch = build(layout)
    assert_identical(update(model, batch, strict_checks=True), update(model, batch, strict_checks=False))


def test_strict_checks_off_skips_only_value_checks():
    valid = torch.tensor([[True, True, False]])
    bad_reference = torch.tensor([[False, False, True]])  # reference outside the valid frames
    with pytest.raises(ValueError, match="subset"):
        mask_values(valid, bad_reference)
    assert torch.equal(mask_values(valid, bad_reference, strict=False), valid & ~bad_reference)
    with pytest.raises(ValueError, match="boolean"):
        mask_values(valid.long(), bad_reference, strict=False)  # shape/dtype checks stay
    empty = torch.zeros(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="target frame"):
        per_example_mse(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), empty)
    per_example_mse(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), empty, strict=False)
    model, batch = build()
    batch["prompt_mask"] = batch["valid"].clone()  # no target frames at all
    with pytest.raises(ValueError, match="target frame"):
        flow_loss(model, batch)
    model.strict_checks = False
    flow_loss(model, batch)


def test_nonfinite_watch_defers_and_names_the_first_update():
    watch = NonfiniteWatch(torch.device("cpu"))
    watch.note("objective", torch.tensor(1.0), 0)
    watch.note("gradient", torch.tensor(2.0), 0)
    watch.check()
    watch.note("objective", torch.tensor(float("nan")), 4)
    watch.note("objective", torch.tensor(float("inf")), 6)
    watch.note("gradient", torch.tensor(1.0), 6)
    with pytest.raises(FloatingPointError, match="objective at update 5"):
        watch.check()


def test_deferred_nonfinite_check_stops_before_the_checkpoint(cache, tmp_path, monkeypatch):
    from dacvae_tts import training

    config = speed_config(tmp_path, log_every=10, checkpoint_every=3)
    forward, calls = training.Objective.forward, {"count": 0}

    def poisoned(self, batch):
        losses = forward(self, batch)
        calls["count"] += 1
        if calls["count"] == 3:  # first micro-batch of update 2
            losses["flow"] = losses["flow"] * float("nan")
        return losses

    monkeypatch.setattr(training.Objective, "forward", poisoned)
    output = tmp_path / "poisoned"
    with pytest.raises(FloatingPointError, match="objective at update 2"):
        training.train(training_args(config, cache, output))
    assert calls["count"] == 6  # detected at the update-3 checkpoint, not at update 2
    assert not (output / "last.pt").exists()


def test_relaxed_step_has_no_python_visible_host_syncs(monkeypatch):
    """Without value checks and with loader negatives the step never asks the device for a value.

    Counted: truth tests, item/tolist/int/float conversions, copies to the CPU and boolean-mask
    indexing (which needs the number of selected elements). What remains is inside F.ctc_loss, which
    copies the length tensors to the host.
    """
    model, _ = build()
    batch = TrainCollate(negatives=True)(
        [dict(item, negative_seed=f"test:{row}") for row, item in enumerate(items())]
    )
    calls = []
    original_getitem = torch.Tensor.__getitem__

    def counted(name, function):
        def wrapper(self, *args, **kwargs):
            calls.append(name)
            return function(self, *args, **kwargs)

        return wrapper

    def getitem(self, index):
        parts = index if isinstance(index, tuple) else (index,)
        if any(isinstance(p, torch.Tensor) and p.dtype == torch.bool for p in parts):
            calls.append("boolean index")
        return original_getitem(self, index)

    def run(strict):
        calls.clear()
        with monkeypatch.context() as patch:
            for name in ("__bool__", "item", "tolist", "__int__", "__float__", "cpu"):
                patch.setattr(torch.Tensor, name, counted(name, getattr(torch.Tensor, name)))
            patch.setattr(torch.Tensor, "__getitem__", getitem)
            update(model, batch, strict_checks=strict, grad_checkpoint=True)
        return list(calls)

    assert run(strict=True)  # the value checks do synchronize
    assert run(strict=False) == []


@pytest.mark.parametrize("patch_size", [1, 2])
def test_length_padding_keeps_losses(patch_size):
    model, batch = build("segments", patch_size)
    padded = pad_lengths(batch, 16, 8)
    assert padded["latents"].size(1) % 16 == 0 and padded["tokens"].size(1) % 8 == 0
    assert padded["latents"].size(1) > batch["latents"].size(1)
    assert padded["tokens"].size(1) > batch["tokens"].size(1)
    extra = slice(batch["latents"].size(1), None)
    assert not padded["valid"][:, extra].any() and not padded["prompt_mask"][:, extra].any()
    assert (padded["tokens"][:, batch["tokens"].size(1) :] == 0).all()
    time = torch.tensor([0.2, 0.5, 0.9])
    noise = torch.randn_like(batch["latents"])
    padded_noise = F.pad(noise, (0, 0, 0, padded["latents"].size(1) - noise.size(1)))
    model.train()
    first = flow_loss(model, batch, 0, time, noise, return_details=True)
    second = flow_loss(model, padded, 0, time, padded_noise, return_details=True)
    for key in ("flow", "ctc", "prediction_rms"):
        assert torch.allclose(first[key], second[key], rtol=1e-5, atol=1e-6), key
    assert (first["ctc"] > 0).all() or patch_size > 1
    assert torch.equal(first["frames"], second["frames"])
    duration = model.predict_duration(
        batch["prompt"], batch["prompt_mask"], batch["tokens"], batch["segments"]
    )
    padded_duration = model.predict_duration(
        padded["prompt"], padded["prompt_mask"], padded["tokens"], padded["segments"]
    )
    assert torch.allclose(duration, padded_duration, rtol=1e-5, atol=1e-6)


def test_train_collate_pads_and_draws_deterministic_negatives():
    rows = [dict(item, negative_seed=f"negatives:42:0:{i}") for i, item in enumerate(items())]
    first, second = TrainCollate(16, 8, True)(rows), TrainCollate(16, 8, True)(rows)
    plain = collate(rows)
    assert first.keys() == second.keys() >= {"negative_tokens", "negative_segments", "negative_usable"}
    for key in first:
        assert torch.equal(first[key], second[key]), key
    assert first["negative_tokens"].size(1) % 8 == 0 and first["negative_usable"].all()
    for i, row in enumerate(rows):
        expected, expected_segments = corrupt_transcript(
            plain["tokens"][i], plain["segments"][i], random.Random(row["negative_seed"])
        )
        assert torch.equal(first["negative_tokens"][i, : len(expected)], expected)
        assert torch.equal(first["negative_segments"][i, : len(expected)], expected_segments)
        assert (first["negative_tokens"][i, len(expected) :] == 0).all()
    reseeded = TrainCollate(16, 8, True)([dict(r, negative_seed=r["negative_seed"] + "x") for r in rows])
    assert not torch.equal(reseeded["negative_tokens"], first["negative_tokens"])
    # The objective takes them from the batch instead of corrupting the transcripts itself.
    objective = Objective(build()[0], contrastive_weight=0.2)
    objective.rng = None  # the fallback path would fail without its generator
    negatives = objective.negatives(first)
    assert negatives[0] is first["negative_tokens"] and negatives[2] is first["negative_usable"]


def test_corrupt_rows_matches_the_objective_fallback():
    _, batch = build()
    objective = Objective(build()[0], contrastive_weight=0.2)
    expected = objective.negatives(batch)
    shared = random.Random(0)  # Objective's generator, shared by all rows
    result = corrupt_rows(batch["tokens"], batch["segments"], [shared] * len(batch["tokens"]))
    for a, b in zip(expected, result):
        assert torch.equal(a, b)


def test_negative_seeds_are_unique_per_epoch_and_row():
    class Rows:
        epoch, costs = 7, np.array([3, 4, 5])

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return {"index": index}

    wrapped = NegativeSeeds(Rows(), 42)
    assert len(wrapped) == 3
    assert wrapped[(3, 1)] == {"index": (3, 1), "negative_seed": "negatives:42:3:1"}
    assert wrapped[2]["negative_seed"] == "negatives:42:7:2"
    train = TrainConfig()
    assert training_loader(Rows(), train)[1] is collate  # defaults: the plain collate
    train.loader_negatives = True
    assert training_loader(Rows(), train)[1] is collate  # no contrastive term: nothing to draw
    train.contrastive_weight, train.pad_multiple = 0.2, 4
    loader_items, collate_fn, costs = training_loader(Rows(), train)
    assert isinstance(loader_items, NegativeSeeds) and collate_fn.negatives
    assert costs.tolist() == [4, 4, 8]


def test_padded_costs_bound_the_padded_batch_frames():
    rng = np.random.default_rng(0)
    costs = rng.integers(5, 60, 500)
    assert padded_costs(costs, 1) is costs
    padded = padded_costs(costs, 16)
    assert (padded % 16 == 0).all() and (padded >= costs).all() and (padded - costs < 16).all()
    sampler = BucketBatchSampler(padded, 32, frame_budget=256, bucket_size=128)
    for batch in sampler.batches():
        length = max(costs[i] for i in batch)
        assert len(batch) * (-(-length // 16) * 16) <= 256


def test_pad_multiples_are_validated():
    TrainConfig(pad_multiple=64, text_pad_multiple=32)
    for options in (dict(pad_multiple=0), dict(text_pad_multiple=True), dict(pad_multiple=2.0)):
        with pytest.raises(ValueError, match="pad_multiple"):
            TrainConfig(**options)


def test_speed_options_train_and_resume_exactly(cache, tmp_path):
    """Loader negatives are seeded per item, so unlike the objective's shared generator they survive
    an interruption; spawned workers exercise the picklable collate and dataset wrapper."""
    config = speed_config(tmp_path, workers=2, pad_multiple=8, text_pad_multiple=8, loader_negatives=True)
    base = ["-m", "dacvae_tts", "train", "--config", str(config), "--cache", str(cache), "--device", "cpu"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    full, resumed = tmp_path / "full", tmp_path / "resumed"

    def run(args):
        subprocess.run(
            [sys.executable, *args], check=True, capture_output=True, text=True, timeout=300, env=env
        )

    run([*base, "--output", str(full)])
    run([*base, "--output", str(resumed), "--stop-after", "2"])
    run([*base, "--output", str(resumed), "--resume", str(resumed / "last.pt")])
    _, a = load_model(full / "last.pt")
    _, b = load_model(resumed / "last.pt")
    assert a["step"] == b["step"] == 4
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key
        assert torch.equal(a["ema"][key], b["ema"][key]), key
    records = [json.loads(line) for line in (full / "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records if "flow" in r] == [1, 2, 3, 4]
    assert all(r["contrastive"] >= 0 and np.isfinite(r["loss"]) for r in records if "flow" in r)
    assert any(r.get("validation_loss") is not None for r in records)


@pytest.mark.parametrize("mode", [True, "selective", 2, 3])
def test_checkpoint_modes_give_identical_gradients(mode):
    model, batch = build()
    assert_identical(update(model, batch, grad_checkpoint=False), update(model, batch, grad_checkpoint=mode))


def test_block_checkpoint_modes():
    assert [block_checkpoint(2, n) for n in (1, 2, 3, 4)] == [None, "full", None, "full"]
    assert block_checkpoint(True, 1) == "full" and block_checkpoint("selective", 5) == "selective"
    assert block_checkpoint(False, 1) is None and block_checkpoint(True, 1, training=False) is None


def test_checkpoint_and_compile_options_are_validated():
    for value in (False, True, "selective", 1, 2, 12):
        TrainConfig(grad_checkpoint=value)
    for value in ("full", 0, -1, 1.5, "2"):
        with pytest.raises(ValueError, match="grad_checkpoint"):
            TrainConfig(grad_checkpoint=value)
    TrainConfig(compile="blocks", compile_dynamic="auto")
    with pytest.raises(ValueError, match="compile_dynamic"):
        TrainConfig(compile_dynamic="static")
    with pytest.raises(ValueError, match="compile"):
        TrainConfig(compile="block")


def test_throughput_defaults_load_old_checkpoints():
    defaults = TrainConfig()
    assert (defaults.strict_checks, defaults.pad_multiple, defaults.text_pad_multiple) == (True, 1, 1)
    assert not defaults.loader_negatives and defaults.grad_checkpoint is False and defaults.compile is False
    # Checkpoints written before these options existed load with the defaults.
    old = {k: v for k, v in dataclasses.asdict(defaults).items() if k not in SPEED_OPTIONS}
    assert Config.from_dict({"model": {}, "train": old}).train == defaults


def compiler_available():
    try:
        torch._dynamo.reset()
        return torch.equal(torch.compile(lambda x: x * 2 + 1)(torch.ones(3)), torch.full((3,), 3.0))
    except Exception:
        return False
    finally:
        torch._dynamo.reset()


def test_compiled_blocks_match_eager_and_share_graphs_across_batch_sizes():
    if not compiler_available():
        pytest.skip("torch.compile has no working backend here")
    from torch._dynamo.utils import counters

    eager, _ = build()
    compiled, _ = build()
    for model in (eager, compiled):
        model.grad_checkpoint = "selective"
    torch._dynamo.reset()
    counters.clear()
    compile_blocks(compiled, "batch")
    try:
        rows = items()
        for batch in (pad_lengths(collate(rows), 64, 32), pad_lengths(collate(rows[:2]), 64, 32)):
            time, noise = torch.rand(len(batch["latents"])), torch.randn_like(batch["latents"])
            results = []
            for model in (eager, compiled):
                model.zero_grad(set_to_none=True)
                details = flow_loss(model, batch, 0, time, noise, return_details=True)
                (details["flow"].mean() + details["ctc"].mean()).backward()
                results.append((details["flow"], [p.grad for p in model.parameters() if p.grad is not None]))
            assert torch.allclose(results[0][0], results[1][0], rtol=1e-4, atol=1e-5)
            for a, b in zip(results[0][1], results[1][1]):
                assert torch.allclose(a, b, rtol=1e-4, atol=1e-5)
        # All three blocks and both batch sizes (same padded lengths) share one compiled graph.
        assert counters["stats"]["unique_graphs"] == 1
    finally:
        torch._dynamo.reset()


def test_blocks_compiler_failure_falls_back_to_eager(cache, tmp_path, monkeypatch, capsys):
    import torch._inductor.exc as inductor

    from dacvae_tts import training

    calls = {"count": 0}

    def fake_compile(function, dynamic=True):
        def wrapped(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:  # the second block call hits a "rare shape"
                raise inductor.InductorError(AssertionError("synthetic"), None)
            return function(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(torch, "compile", fake_compile)
    options = dict(pad_multiple=8, text_pad_multiple=8, loader_negatives=True, grad_checkpoint="selective")
    config = speed_config(tmp_path, compile="blocks", log_every=1, **options)
    training.train(training_args(config, cache, tmp_path / "fallback"))
    _, saved = load_model(tmp_path / "fallback" / "last.pt")
    assert saved["step"] == 4 and calls["count"] == 2  # compiled blocks abandoned after the failure
    assert Config.from_dict(saved["config"]).train.compile == "blocks"
    assert any("activation checkpointing" in line for line in capsys.readouterr().out.splitlines())
