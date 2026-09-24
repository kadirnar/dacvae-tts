"""Training throughput options (speed.py): off they are bit-identical, on they keep the objective."""

import types

import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from dacvae_tts import model as model_module
from dacvae_tts.config import ModelConfig
from dacvae_tts.contracts import mask_values
from dacvae_tts.data import collate
from dacvae_tts.model import FlowTTS, ctc_alignment_loss, flow_loss, per_example_mse
from dacvae_tts.speed import NonfiniteWatch
from dacvae_tts.text import BYTE_OFFSET
from dacvae_tts.training import Objective


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
