"""Opt-in DiT block options (issue #9): each one trains, ignores padding and, off, changes nothing."""

import hashlib

import pytest
import torch

from dacvae_tts.config import Config, ModelConfig
from dacvae_tts.model import FlowTTS
from dacvae_tts.optim import partition
from dacvae_tts.training import Objective
from tests.test_nano import NANO, nano_batch

BASE = dict(NANO, adaln_rank=8)
OPTIONS = {
    "long_skip": dict(long_skip=True),
}
# Options whose new parameters start at zero (or identity): the model starts as the baseline function.
ZERO_INIT = ["long_skip"]
ALL = {key: value for option in OPTIONS.values() for key, value in option.items()}

# Captured from the architecture before these options existed (8f01f08): state_dict length and layout hash,
# and an RNG-free forward fingerprint, so a default model keeps its keys, shapes and computation.
PLAIN = dict(latent_dim=4, width=32, heads=2, depth=2, text_depth=1)  # absolute positions, full adaLN, P=2
PREVIOUS = [
    (BASE, 94, "2a5acd8deff41289", (-6.339404, 7.867113)),
    (PLAIN, 66, "ef917c3d8b03d440", (0.432159, 5.663713)),
]


def layout_hash(model):
    rows = "\n".join(f"{key}:{tuple(value.shape)}" for key, value in model.state_dict().items())
    return hashlib.sha1(rows.encode()).hexdigest()[:16]


def deterministic(model):
    """RNG-free weights, so the fingerprint depends on the computation only (not on the platform's RNG)."""
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            wave = torch.sin(torch.arange(parameter.numel(), dtype=torch.float64) * 0.7 + index)
            parameter.copy_((0.3 * wave / parameter.shape[-1] ** 0.5).reshape(parameter.shape))
    return model


def fingerprint(model):
    batch = nano_batch(prompts=(3, 4))
    shape = batch["latents"].shape
    frames = torch.arange(shape[1] * shape[2], dtype=torch.float64).reshape(shape[1:])
    batch["latents"] = torch.cos(frames * 0.3).float().expand_as(batch["latents"]) * batch["valid"][..., None]
    batch["prompt"] = batch["latents"] * batch["prompt_mask"][..., None]
    kwargs = {k: v for k, v in batch.items() if k != "latents"}
    with torch.no_grad():
        out = deterministic(model).eval()(batch["latents"], torch.linspace(0.2, 0.8, 2), **kwargs).double()
    return float(out.sum()), float(out.abs().sum())


def perturbed(model, seed=0):
    """Noise on every tensor that holds zeros (zero-init gates, skips, modulations): exercises all paths."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.all():
                parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return model


def inputs(batch):
    return {k: v for k, v in batch.items() if k != "latents"}


def new_parameters(model):
    previous = FlowTTS(ModelConfig(**BASE)).state_dict()
    return {name: p for name, p in model.named_parameters() if name not in previous}


@pytest.mark.parametrize("overrides, keys, layout, expected", PREVIOUS)
def test_options_off_keep_the_previous_model(overrides, keys, layout, expected):
    model = FlowTTS(ModelConfig(**overrides))
    assert len(model.state_dict()) == keys and layout_hash(model) == layout
    assert fingerprint(model) == pytest.approx(expected, rel=1e-5, abs=1e-5)
    # Explicitly disabled options are the default: an old checkpoint loads strictly.
    off = {"long_skip": False}
    FlowTTS(ModelConfig(**overrides, **off)).load_state_dict(model.state_dict(), strict=True)


@pytest.mark.parametrize("name", ZERO_INIT)
def test_zero_initialized_option_starts_as_the_baseline(name):
    base = perturbed(FlowTTS(ModelConfig(**BASE))).eval()
    model = FlowTTS(ModelConfig(**BASE, **OPTIONS[name])).eval()
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and sorted(missing) == sorted(new_parameters(model))
    with pytest.raises(RuntimeError):
        model.load_state_dict(base.state_dict())  # new parameters: an old checkpoint is not silently complete
    batch = nano_batch(prompts=(3, 4))
    time, drop = torch.tensor([0.3, 0.7]), torch.tensor([False, True])
    expected = base(batch["latents"], time, **inputs(batch), drop=drop)
    assert torch.equal(model(batch["latents"], time, **inputs(batch), drop=drop), expected)


@pytest.mark.parametrize("name", [*OPTIONS, "all"])
def test_option_trains_with_checkpointing_and_ignores_padding(name):
    overrides = ALL if name == "all" else OPTIONS[name]
    model = perturbed(FlowTTS(ModelConfig(**BASE, **overrides))).train()
    batch = nano_batch(frames=64)  # CTC needs at least as many frames as transcript bytes
    gradients = []
    for checkpointing in (False, True):
        model.grad_checkpoint = checkpointing
        model.zero_grad(set_to_none=True)
        torch.manual_seed(0)
        losses = Objective(model, expansion=2, ctc_weight=0.1).train()(batch)
        (losses["loss"].mean() + 0.1 * losses["ctc"].mean()).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        gradients.append([p.grad.clone() for p in model.parameters()])
    for plain, recomputed in zip(*gradients, strict=True):
        assert torch.allclose(plain, recomputed, atol=1e-6)
    assert all(p.grad.abs().sum() > 0 for p in new_parameters(model).values())

    model.eval()
    batch = nano_batch()  # the first example has two padded frames
    time = torch.tensor([0.3, 0.7])
    full = model(batch["latents"], time, **inputs(batch))
    length, text_length = int(batch["valid"][0].sum()), int(batch["tokens"][0].ne(0).sum())
    text = ("tokens", "segments")
    short = {k: v[:1, :text_length] if k in text else v[:1, :length] for k, v in inputs(batch).items()}
    alone = model(batch["latents"][:1, :length], time[:1], **short)
    assert torch.allclose(full[0, :length], alone[0], atol=1e-5)


def test_muon_partition_covers_the_new_parameters():
    model = FlowTTS(ModelConfig(**BASE, **ALL))
    matrices, parts, others = partition(model)
    names = {id(p): name for name, p in model.named_parameters()}
    assert sorted(names) == sorted(id(p) for p in (*matrices, *others))
    muon = {names[id(p)]: count for p, count in zip(matrices, parts, strict=True)}
    adamw = {names[id(p)] for p in others}
    assert muon["skip.1.weight"] == 1 and {"skip.0.weight", "skip.1.bias"} <= adamw  # hidden [D,2D] map
    assert muon["blocks.0.ff.0.weight"] == 1 and muon["blocks.1.self_attn.kv.weight"] == 2


def test_long_skip_fuses_the_input_embedding_before_the_output_head():
    model = FlowTTS(ModelConfig(**BASE, long_skip=True))
    assert model.skip[1].weight.shape == (32, 64) and not model.skip[1].weight.any()
    extra = sum(p.numel() for p in new_parameters(model).values())
    assert extra == 2 * 64 + 64 * 32 + 32  # LayerNorm over [h_0, h_L] + zero-init Linear


def test_example_configs_change_one_model_option_of_the_w512_recipe():
    base = Config.load("configs/nano_tr_w512.yaml").to_dict()
    for name, overrides in OPTIONS.items():
        experiment = Config.load(f"configs/experiments/tr_w512_{name}.yaml").to_dict()
        assert experiment["train"] == base["train"], name
        changed = {k: v for k, v in experiment["model"].items() if base["model"][k] != v}
        assert changed == overrides, name
    replicate = Config.load("configs/experiments/tr_w512_seed43.yaml").to_dict()
    assert replicate["model"] == base["model"] and replicate["train"] == {**base["train"], "seed": 43}
