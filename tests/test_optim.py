import copy

import pytest
import torch

from dacvae_tts.config import ModelConfig, TrainConfig
from dacvae_tts.model import FlowTTS
from dacvae_tts.optim import Muon, build_optimizer, muon_parts, orthogonalize, partition
from dacvae_tts.training import Objective
from tests.test_model import model_and_batch


def test_newton_schulz_drives_singular_values_towards_one():
    torch.manual_seed(0)
    for shape in ((3, 32, 16), (2, 16, 48), (4, 24, 24)):
        matrices = torch.randn(shape)
        result = orthogonalize(matrices)
        assert result.shape == matrices.shape and result.dtype == matrices.dtype
        singular = torch.linalg.svdvals(result)
        assert singular.max() < 1.3
        if shape[1] != shape[2]:
            assert singular.min() > 0.5
        else:
            # Random square matrices have near-zero singular values that five steps cannot fully lift.
            assert (singular > 0.5).float().mean() > 0.9
        # The iteration keeps the singular vectors: U^T result V stays (almost) diagonal.
        u, _, vh = torch.linalg.svd(matrices, full_matrices=False)
        core = u.mT @ result @ vh.mT
        assert (core - torch.diag_embed(core.diagonal(dim1=-2, dim2=-1))).abs().max() < 1e-3
    with pytest.raises(ValueError):
        orthogonalize(torch.randn(4, 4))


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"reference_encoder": "temporal", "reference_pooling": "attention"},
        {"reference_pooling": "mean_std"},
    ],
)
def test_partition_assigns_every_parameter_once(overrides):
    model = FlowTTS(ModelConfig(latent_dim=4, width=32, heads=2, depth=2, text_depth=1, **overrides))
    matrices, parts, others = partition(model)
    names = {id(p): name for name, p in model.named_parameters()}
    assert sorted(names) == sorted(id(p) for p in (*matrices, *others))
    muon = {names[id(p)]: count for p, count in zip(matrices, parts, strict=True)}
    adamw = {names[id(p)] for p in others}
    assert muon["blocks.0.self_attn.q.weight"] == 1 and muon["blocks.1.ff.2.weight"] == 1
    assert muon["blocks.0.cross_attn.kv.weight"] == 2 and muon["blocks.1.ada.1.weight"] == 9
    assert muon["text.mlps.0.1.weight"] == 1 and muon["time.2.weight"] == 1
    # Embeddings, boundary projections, heads, filters, gains and biases are not hidden matrices.
    for name in (
        "text.embedding.weight",
        "text.segment.weight",
        "text.convs.0.weight",
        "text.norm.weight",
        "input.weight",
        "output.1.weight",
        "duration.2.weight",
        "blocks.0.self_attn.q.bias",
    ):
        assert name in adamw, name
    assert all(p.ndim == 2 and min(p.shape) > 1 for p in matrices)
    assert ("ref.input.weight" if "reference_encoder" in overrides else "ref.0.weight") in adamw
    if overrides.get("reference_pooling") == "attention":
        assert "ref_pool.attention.weight" in adamw  # [1,D] is a vector, not a matrix


def test_fused_rows_are_orthogonalized_separately():
    torch.manual_seed(0)
    fused = torch.nn.Parameter(torch.zeros(16, 8))
    assert muon_parts("blocks.0.self_attn.kv.weight", fused) == 2
    optimizer = Muon([fused], [2], [], lr=1.0, weight_decay=0.0)
    fused.grad = torch.randn(16, 8) * torch.tensor([1.0] * 8 + [100.0] * 8)[:, None]
    optimizer.step()
    update = -fused.detach() / (0.2 * 8**0.5)
    for block in update.chunk(2):
        singular = torch.linalg.svdvals(block)
        assert singular.min() > 0.5 and singular.max() < 1.3
    # The two halves had very different gradient scales; joint orthogonalization would mix them.
    joint = torch.linalg.svdvals(orthogonalize(fused.grad[None])[0].chunk(2)[0])
    assert joint.min() < 0.5


def test_adamw_group_matches_torch_adamw():
    torch.manual_seed(0)
    ours = torch.nn.Parameter(torch.randn(7))
    theirs = torch.nn.Parameter(ours.detach().clone())
    optimizer = Muon([], [], [ours], lr=1e-2, weight_decay=0.1)
    reference = torch.optim.AdamW([theirs], lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95))
    for _ in range(5):
        gradient = torch.randn(7)
        ours.grad, theirs.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        reference.step()
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_training_reduces_loss_and_state_round_trips():
    torch.manual_seed(0)
    model, batch = model_and_batch()
    optimizer = build_optimizer(model, "muon", 3e-3, 0.01)
    assert isinstance(optimizer, Muon) and [g["muon"] for g in optimizer.param_groups] == [True, False]

    def update(module, opt, seed):
        torch.manual_seed(seed)
        opt.zero_grad(set_to_none=True)
        # eval(): no condition dropout, so the fixed seed fully determines noise and flow time.
        loss = Objective(module).eval()(batch)["loss"].mean()
        loss.backward()
        opt.step()
        return loss.item()

    losses = [update(model, optimizer, 123) for _ in range(30)]
    assert losses[-1] < losses[0] * 0.7
    assert all(torch.isfinite(p).all() for p in model.parameters())

    clone = copy.deepcopy(model)
    restored = build_optimizer(clone, "muon", 3e-3, 0.01)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    update(model, optimizer, 7)
    update(clone, restored, 7)
    for left, right in zip(model.parameters(), clone.parameters(), strict=True):
        assert torch.equal(left, right)


def test_configuration_selects_the_optimizer():
    model, _ = model_and_batch()
    assert TrainConfig().optimizer == "muon"
    assert isinstance(build_optimizer(model, "adamw", 1e-3, 0.01), torch.optim.AdamW)
    with pytest.raises(ValueError):
        build_optimizer(model, "sgd", 1e-3, 0.01)
    with pytest.raises(ValueError):
        TrainConfig(optimizer="sgd")
