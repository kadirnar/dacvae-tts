import copy

import pytest
import torch
from torch import nn

from dacvae_tts.config import ModelConfig
from dacvae_tts.contracts import normalization_stats
from dacvae_tts.data import collate, save_stats
from dacvae_tts.model import FlowTTS, flow_loss, per_example_mse, reduce_flow, sample, sinusoidal, time_grid
from dacvae_tts.training import Objective


def fixture(ref=3, target=4, extra=0, **options):
    cfg = ModelConfig(latent_dim=4, width=16, depth=2, heads=2, text_depth=1, **options)
    model = FlowTTS(cfg)
    batch = collate(
        [
            dict(
                reference=torch.randn(ref, 4),
                target=torch.randn(target, 4),
                reference_text="Reference words.",
                text="Target sentence.",
            ),
            dict(
                reference=torch.randn(ref + extra, 4),
                target=torch.randn(target + 2, 4),
                reference_text="Another reference.",
                text="Second target.",
            ),
        ]
    )
    for block in model.blocks:
        nn.init.normal_(block.ada[-1].weight, std=0.05)
    nn.init.normal_(model.output[-1].weight, std=0.05)
    return model, batch


@pytest.mark.parametrize("ref,target", [(3, 3), (3, 4), (4, 3), (4, 4)])
@pytest.mark.parametrize("packing", [1, 2, 3])
def test_partial_pack_poison_invariance(ref, target, packing):
    model, batch = fixture(ref, target, patch_size=packing)
    kwargs = {k: v for k, v in batch.items() if k != "latents"}
    t = torch.tensor([0.2, 0.8])
    clean = model(batch["latents"], t, **kwargs)
    poisoned = batch["latents"].masked_fill(~batch["valid"][..., None], float("nan"))
    kwargs["prompt"] = kwargs["prompt"].masked_fill(~batch["prompt_mask"][..., None], float("nan"))
    result = model(poisoned, t, **kwargs)
    assert torch.equal(clean, result)
    assert torch.isfinite(result).all()
    assert model.input.in_features == packing * 9
    assert model.output[-1].out_features == packing * 4


def test_baseline_dimensions_and_summary():
    model = FlowTTS(ModelConfig())
    assert model.input.in_features == 514
    assert model.output[-1].out_features == 256
    assert model.reference_summary(torch.randn(2, 7, 128), torch.ones(2, 7, dtype=torch.bool)).shape == (
        2,
        256,
    )


def test_bad_shapes_and_empty_targets_fail():
    model, batch = fixture()
    kwargs = {k: v for k, v in batch.items() if k != "latents"}
    with pytest.raises(ValueError, match="C=4"):
        model(torch.randn(2, 9, 5), torch.rand(2), **kwargs)
    with pytest.raises(ValueError, match="Flow time"):
        model(batch["latents"], torch.rand(2, 1), **kwargs)
    with pytest.raises(ValueError, match="target frame"):
        per_example_mse(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.bool))
    batch["prompt_mask"] = batch["valid"].clone()
    with pytest.raises(ValueError, match="target frame"):
        flow_loss(model, batch)


def test_hand_computed_reductions_and_poisoned_loss():
    prediction = torch.tensor([[[2.0], [float("nan")], [float("inf")]], [[1.0], [1.0], [1.0]]])
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool)
    losses = per_example_mse(prediction, torch.zeros_like(prediction), mask)
    assert torch.equal(losses, torch.tensor([4.0, 1.0]))
    assert reduce_flow(losses, mask.sum(1), "utterance") == 2.5
    assert reduce_flow(losses, mask.sum(1), "frame") == 1.75


class Field(nn.Module):
    def __init__(self, conditioned=0.0, null=0.0):
        super().__init__()
        self.conditioned, self.null, self.calls = conditioned, null, 0

    def conditions(self, prompt, mask, tokens, segments, speaker=None, context=None, context_mask=None,
                   quality=None):
        return (torch.ones(tokens.shape + (4,)), tokens.ne(0), torch.zeros(prompt.size(0), 4))

    def forward(self, x, t, prompt, prompt_mask, valid, tokens, segments, cached=None):
        self.calls += 1
        # Guided sampling evaluates the conditioned and null branches in one batch.
        conditioned = cached[0].flatten(1).any(1)[:, None, None]
        return torch.where(conditioned, torch.full_like(x, self.conditioned), torch.full_like(x, self.null))


@pytest.mark.parametrize("value", [0.0, 2.0])
@pytest.mark.parametrize("grid", [[0.0, 0.01, 0.4, 1.0], [0.0, 0.3, 0.7, 1.0]])
def test_analytic_euler_and_fixed_reference(value, grid):
    _, batch = fixture()
    kwargs = {k: v for k, v in batch.items() if k != "latents"}
    stats = {}
    field = Field(value, value)
    output, times, states = sample(
        field,
        **kwargs,
        steps=3,
        times=grid,
        guidance=1,
        initial_noise=torch.zeros_like(batch["latents"]),
        return_trajectory=True,
        stats=stats,
    )
    mask = batch["valid"] & ~batch["prompt_mask"]
    assert torch.allclose(output[mask], torch.full_like(output[mask], value))
    for state in states:
        assert torch.equal(state[batch["prompt_mask"]], batch["prompt"][batch["prompt_mask"]])
        assert (state[~batch["valid"]] == 0).all()
    assert field.calls == stats["forward_calls"] == stats["branch_evaluations"] == 3


def test_guidance_equation_and_work_count():
    _, batch = fixture()
    kwargs = {k: v for k, v in batch.items() if k != "latents"}
    field = Field(3.0, 1.0)
    stats = {}
    result = sample(
        field, **kwargs, steps=2, guidance=1.5, initial_noise=torch.zeros_like(batch["latents"]), stats=stats
    )
    mask = batch["valid"] & ~batch["prompt_mask"]
    assert torch.allclose(result[mask], torch.full_like(result[mask], 4.0))
    assert field.calls == stats["forward_calls"] == 2 and stats["branch_evaluations"] == 4


@pytest.mark.parametrize(
    "grid", [[0.1, 0.5, 1.0], [0.0, 0.5, 0.9], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0], [0.0, float("nan"), 1.0]]
)
def test_reject_bad_grids(grid):
    with pytest.raises(ValueError):
        time_grid(2, -1, "cpu", grid)


def test_null_payload_invariance_and_train_inference_equivalence():
    model, batch = fixture()
    kwargs = {k: v for k, v in batch.items() if k != "latents"}
    drop = torch.ones(2, dtype=torch.bool)
    t = torch.tensor([0.3, 0.6])
    x = batch["latents"].clone()
    first = model(x, t, **kwargs, drop=drop)
    changed = copy.deepcopy(kwargs)
    changed["prompt"] = torch.randn_like(changed["prompt"]) * 100
    changed["tokens"] = torch.where(
        changed["tokens"] >= 4, (changed["tokens"] + 3).clamp_max(259), changed["tokens"]
    )
    x = x.masked_fill(batch["prompt_mask"][..., None], 500.0)
    second = model(x, t, **changed, drop=drop)
    assert torch.equal(first, second)
    null_kwargs = {
        **kwargs,
        "prompt": torch.zeros_like(kwargs["prompt"]),
        "prompt_mask": torch.zeros_like(kwargs["prompt_mask"]),
    }
    cache = model.conditions(
        null_kwargs["prompt"], null_kwargs["prompt_mask"], kwargs["tokens"], kwargs["segments"], drop
    )
    direct = model(x.masked_fill(batch["prompt_mask"][..., None], 0), t, **null_kwargs, cached=cache)
    assert torch.equal(first, direct)


@pytest.mark.parametrize(
    "encoder,pooling",
    [("mlp", "mean"), ("temporal", "mean"), ("temporal", "attention"), ("temporal", "mean_std")],
)
def test_reference_summary_masking_and_no_reference(encoder, pooling):
    model, _ = fixture(reference_encoder=encoder, reference_pooling=pooling)
    x = torch.randn(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    first = model.reference_summary(x, mask)
    x = torch.cat([x, torch.full((1, 4, 4), float("nan"))], 1)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    assert torch.allclose(first, model.reference_summary(x, mask), atol=1e-6)
    assert torch.equal(model.reference_summary(x, torch.zeros_like(mask)), torch.zeros(1, 16))


def test_positions_order_and_long_lengths():
    short = sinusoidal(torch.arange(10), 16)
    long = sinusoidal(torch.arange(10000), 16)
    assert torch.equal(short, long[:10])
    assert not torch.equal(short[0], short[1])
    assert torch.isfinite(long).all()
    model, batch = fixture()
    kwargs = {k: v for k, v in batch.items() if k != "latents"}
    assert not torch.allclose(
        model(batch["latents"], torch.zeros(2), **kwargs), model(batch["latents"], torch.ones(2), **kwargs)
    )


def test_zero_initialization_eventual_updates():
    _, batch = fixture()
    model = FlowTTS(ModelConfig(latent_dim=4, width=16, depth=2, heads=2, text_depth=1))
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    assert model.output[-1].weight.count_nonzero() == 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for _ in range(8):
        optimizer.zero_grad()
        loss = Objective(model)(batch)["loss"].mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    for group in ("text", "ref", "time", "blocks", "input", "output", "duration"):
        assert any(
            not torch.equal(p, initial[name])
            for name, p in model.named_parameters()
            if name.startswith(group + ".")
        )


def test_small_variance_statistics(tmp_path):
    path = tmp_path / "stats.pt"
    save_stats(path, 10, torch.ones(4, dtype=torch.float64) * 20, torch.ones(4, dtype=torch.float64) * 40)
    obj = torch.load(path, weights_only=True)
    normalization_stats(obj["mean"], obj["std"], 4)
    assert torch.equal(obj["std"], torch.ones(4) * 0.001)
    with pytest.raises(ValueError):
        normalization_stats(torch.zeros(4), torch.zeros(4))


def test_interpolation_clean_prefix_and_velocity_target():
    _, batch = fixture()
    noise = torch.full_like(batch["latents"], 2)
    times = torch.tensor([0.0, 0.75])

    class RecordingField(nn.Module):
        def forward(self, x, time, prompt, prompt_mask, valid, tokens, segments, drop):
            self.x = x.detach().clone()
            return batch["latents"] - noise

    field = RecordingField()
    loss = flow_loss(field, batch, dropout=0, noise=noise, time=times)
    assert torch.equal(loss, torch.zeros(2))
    mask = batch["valid"] & ~batch["prompt_mask"]
    expected = (1 - times[:, None, None]) * noise + times[:, None, None] * batch["latents"]
    assert torch.equal(field.x[mask], expected[mask])
    assert torch.equal(field.x[batch["prompt_mask"]], batch["latents"][batch["prompt_mask"]])
    assert field.x[~batch["valid"]].count_nonzero() == 0


def test_audio_forward_beyond_training_length():
    model, batch = fixture(ref=27, target=1001, extra=1)
    kwargs = {k: v for k, v in batch.items() if k != "latents"}
    result = model(batch["latents"], torch.tensor([0.2, 0.9]), **kwargs)
    assert result.shape == batch["latents"].shape
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("pooling", ["mean", "attention", "mean_std"])
def test_optional_reference_and_duration_features_train(pooling):
    model, batch = fixture(
        reference_encoder="temporal", reference_pooling=pooling, duration_features="text_stats"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(3):
        optimizer.zero_grad()
        loss = Objective(model)(batch)["loss"].mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
