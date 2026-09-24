"""Opt-in DiT block options (issue #9): each one trains, ignores padding and, off, changes nothing."""

import hashlib

import pytest
import torch

from dacvae_tts.config import Config, ModelConfig
from dacvae_tts.model import Attention, Block, FlowTTS, sample
from dacvae_tts.optim import partition
from dacvae_tts.text import BYTE_OFFSET
from dacvae_tts.training import Objective
from tests.test_model import model_and_batch
from tests.test_nano import NANO, nano_batch

BASE = dict(NANO, adaln_rank=8)
OPTIONS = {
    "long_skip": dict(long_skip=True),
    "value_residual": dict(value_residual=True),
    "ffn_conv": dict(ffn_conv_kernel=5),
    "attn_gate": dict(attn_gate="head"),
    "swiglu": dict(ffn_activation="swiglu"),
    "final_adaln": dict(final_adaln=True),
    "cond_text_pool": dict(cond_text_pool=True),
}
# Every option explicitly disabled: must be exactly the previous model.
OFF = dict(
    long_skip=False, value_residual=False, ffn_conv_kernel=0, attn_gate="none", ffn_activation="gelu",
    final_adaln=False, cond_text_pool=False,
)
# Options whose new parameters start at zero (or identity): the model starts as the baseline function.
ZERO_INIT = ["long_skip", "value_residual", "ffn_conv", "attn_gate", "final_adaln", "cond_text_pool"]
ALL = {key: value for option in OPTIONS.values() for key, value in option.items()}
CASES = {**OPTIONS, "all": ALL, "ffn_conv_patch2": dict(ffn_conv_kernel=3, patch_size=2)}

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
    assert OFF.keys() == ALL.keys() and all(getattr(ModelConfig(), k) == v for k, v in OFF.items())
    FlowTTS(ModelConfig(**overrides, **OFF)).load_state_dict(model.state_dict(), strict=True)


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


@pytest.mark.parametrize("name", ZERO_INIT)
def test_option_keeps_the_baseline_initialization_for_the_same_seed(name):
    """New modules draw no random numbers before baseline ones: A/B arms with one seed start from the same
    weights (and, reseeded after construction, see the same noise), so they differ by the option alone."""
    torch.manual_seed(0)
    base = FlowTTS(ModelConfig(**BASE)).state_dict()
    torch.manual_seed(0)
    model = FlowTTS(ModelConfig(**BASE, **OPTIONS[name])).state_dict()
    assert all(torch.equal(value, model[key]) for key, value in base.items())


@pytest.mark.parametrize("name", list(CASES))
def test_option_trains_with_checkpointing_and_ignores_padding(name):
    model = perturbed(FlowTTS(ModelConfig(**{**BASE, **CASES[name]}))).train()
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


def split(model):
    """Muon part counts and AdamW names; every trainable parameter lands in exactly one group."""
    matrices, parts, others = partition(model)
    names = {id(p): name for name, p in model.named_parameters()}
    assert sorted(names) == sorted(id(p) for p in (*matrices, *others))
    muon = {names[id(p)]: count for p, count in zip(matrices, parts, strict=True)}
    return muon, {names[id(p)] for p in others}


def test_muon_partition_covers_the_new_parameters():
    muon, adamw = split(FlowTTS(ModelConfig(**BASE, **ALL)))
    assert muon["skip.1.weight"] == 1 and {"skip.0.weight", "skip.1.bias"} <= adamw  # hidden [D,2D] map
    assert "blocks.0.value_mix" in adamw  # two scalars
    assert {"blocks.0.ff_conv.weight", "blocks.1.ff_conv.bias"} <= adamw  # [H,1,k] filters
    assert {"blocks.0.self_attn.gate.weight", "blocks.1.cross_attn.gate.weight"} <= adamw  # [heads,D] heads
    assert muon["blocks.0.ff.0.proj.weight"] == 2 and muon["blocks.0.ff.1.weight"] == 1  # SwiGLU gate | value
    assert muon["blocks.1.self_attn.kv.weight"] == 2
    assert muon["final_ada.0.weight"] == 1 and muon["final_ada.2.weight"] == 2  # rank-r down | shift, scale
    assert muon["text_pool.weight"] == 1  # hidden D x D map
    full, _ = split(FlowTTS(ModelConfig(**NANO, final_adaln=True)))  # adaln_rank 0: one D -> 2D map
    assert full["final_ada.1.weight"] == 2
    gelu, _ = split(FlowTTS(ModelConfig(**{**BASE, **ALL, "ffn_activation": "gelu"})))
    assert gelu["blocks.0.ff.0.weight"] == 1 and gelu["blocks.0.ff.2.weight"] == 1  # GELU: not a fused matrix


def test_long_skip_fuses_the_input_embedding_before_the_output_head():
    model = FlowTTS(ModelConfig(**BASE, long_skip=True))
    assert model.skip[1].weight.shape == (32, 64) and not model.skip[1].weight.any()
    extra = sum(p.numel() for p in new_parameters(model).values())
    assert extra == 2 * 64 + 64 * 32 + 32  # LayerNorm over [h_0, h_L] + zero-init Linear


def test_value_residual_feeds_the_first_block_values_to_later_blocks():
    model = FlowTTS(ModelConfig(**BASE, value_residual=True))
    assert all(block.value_mix.tolist() == [1.0, 0.0] for block in model.blocks)  # identity at init
    assert sum(p.numel() for p in new_parameters(model).values()) == 2 * len(model.blocks)
    model = perturbed(model).eval()
    batch = nano_batch(prompts=(3, 4))
    time = torch.tensor([0.3, 0.7])
    second = model.blocks[1].self_attn.kv.weight
    with torch.no_grad():
        model.blocks[1].value_mix.copy_(torch.tensor([0.0, 1.0]))  # block 2 uses only the values of block 1
        before = model(batch["latents"], time, **inputs(batch))
        second[32:] += torch.randn_like(second[32:])  # block 2's own value projection no longer matters
        assert torch.allclose(model(batch["latents"], time, **inputs(batch)), before, atol=1e-6)
        model.blocks[1].value_mix.copy_(torch.tensor([1.0, 0.0]))
        assert not torch.allclose(model(batch["latents"], time, **inputs(batch)), before, atol=1e-3)


def test_ffn_conv_is_depthwise_and_never_mixes_in_padded_frames():
    model = FlowTTS(ModelConfig(**BASE, ffn_conv_kernel=5))
    hidden = model.blocks[0].ff[-1].in_features
    assert model.blocks[0].ff_conv.weight.shape == (hidden, 1, 5)
    assert sum(p.numel() for p in new_parameters(model).values()) == len(model.blocks) * hidden * (5 + 1)
    block = perturbed(model).blocks[0]
    torch.manual_seed(0)
    h = torch.randn(2, 9, 32)
    valid = torch.ones(2, 9, dtype=torch.bool)
    valid[0, 6:] = False
    garbage = h.masked_fill(~valid[..., None], 1e3)
    out = block.conv_feed_forward(h, valid)
    assert torch.allclose(block.conv_feed_forward(garbage, valid)[valid], out[valid])
    # Sanity: the convolution does reach neighbouring frames, so the mask above is what protects them.
    unmasked = block.conv_feed_forward(garbage, torch.ones_like(valid))
    assert not torch.allclose(unmasked[0, 4:6], out[0, 4:6]) and torch.allclose(unmasked[0, :3], out[0, :3])
    for kernel in (-1, 4):
        with pytest.raises(ValueError):
            ModelConfig(**BASE, ffn_conv_kernel=kernel)


def test_head_gate_scales_each_attention_head():
    model = FlowTTS(ModelConfig(**BASE, attn_gate="head"))
    assert model.blocks[0].cross_attn.gate.weight.shape == (2, 32)
    assert model.text.blocks[0].attention.gate is None  # generator blocks only
    assert sum(p.numel() for p in new_parameters(model).values()) == len(model.blocks) * 2 * (32 * 2 + 2)
    torch.manual_seed(0)
    gated, plain = Attention(32, 2, gate=True), Attention(32, 2)
    gated.load_state_dict(plain.state_dict(), strict=False)
    x, valid = torch.randn(2, 5, 32), torch.ones(2, 5, dtype=torch.bool)
    assert torch.equal(gated(x, x, valid), plain(x, x, valid))  # 2 sigmoid(0) = 1
    with torch.no_grad():
        gated.gate.bias.copy_(torch.tensor([-40.0, 0.0]))  # close head 0, keep head 1 at exactly 1
        closed = gated(x, x, valid)
        plain.out.weight[:, :16] = 0  # the same as dropping head 0's output
        assert torch.allclose(closed, plain(x, x, valid), atol=1e-6)
    with pytest.raises(ValueError):
        ModelConfig(**BASE, attn_gate="elementwise")


def test_swiglu_keeps_the_feed_forward_parameter_count():
    w512 = Config.load("configs/nano_tr_w512.yaml").model
    gelu, swiglu = Block(w512), Block(ModelConfig(**{**w512.__dict__, "ffn_activation": "swiglu"}))
    assert swiglu.ff[0].proj.weight.shape == (2 * 1024, 512) and swiglu.ff[1].weight.shape == (512, 1024)
    count = lambda module: sum(p.numel() for p in module.parameters())  # noqa: E731
    assert count(swiglu.ff) - count(gelu.ff) == 512  # 2/3 of the GELU width: only the extra bias half differs
    x = torch.randn(3, 512)
    gate, value = swiglu.ff[0].proj(x).chunk(2, -1)
    assert torch.allclose(swiglu.ff[0](x), torch.nn.functional.silu(gate) * value)
    with pytest.raises(ValueError):
        ModelConfig(**BASE, ffn_activation="relu")


def test_final_adaln_modulates_the_output_norm_from_the_condition():
    model = FlowTTS(ModelConfig(**BASE, final_adaln=True))
    assert sum(p.numel() for p in new_parameters(model).values()) == (32 * 8 + 8) + (8 * 64 + 64)
    full = FlowTTS(ModelConfig(**NANO, final_adaln=True))
    assert sum(p.numel() for n, p in full.named_parameters() if n.startswith("final_ada")) == 32 * 64 + 64
    model = perturbed(model).eval()
    with torch.no_grad():
        model.final_ada[-1].weight.zero_()
        model.final_ada[-1].bias.copy_(torch.cat([torch.zeros(32), -torch.ones(32)]))  # shift 0, scale -1
    batch = nano_batch(prompts=(3, 4))
    velocity = model(batch["latents"], torch.tensor([0.3, 0.7]), **inputs(batch))
    # 1 + scale = 0 removes the normalized features: only the output head's bias is left on every frame.
    frames = int(batch["valid"].sum())
    assert torch.allclose(velocity[batch["valid"]], model.output[1].bias.expand(frames, -1))


def test_cond_text_pool_averages_the_target_bytes_and_vanishes_when_dropped():
    _, batch = model_and_batch()  # segments layout: reference and target transcripts
    model = perturbed(FlowTTS(ModelConfig(**PLAIN, cond_text_pool=True))).eval()
    assert model.text_pool.weight.shape == (32, 32) and model.text_pool.bias is None
    seen = []
    model.text_pool.register_forward_hook(lambda module, args, output: seen.append(args[0]))
    model(batch["latents"], torch.tensor([0.3, 0.7]), **inputs(batch), drop=torch.tensor([False, True]))
    text = model.conditions(batch["prompt"], batch["prompt_mask"], batch["tokens"], batch["segments"])[0]
    target = (batch["segments"] == 1) & (batch["tokens"] >= BYTE_OFFSET)
    assert target.sum() < batch["tokens"].ne(0).sum()  # reference transcript and special tokens excluded
    expected = (text * target[..., None]).sum(1) / target.sum(1, keepdim=True)
    assert torch.allclose(seen[0][0], expected[0], atol=1e-6)
    assert not seen[0][1].any()  # dropped text: the pooled term is exactly zero


def test_condition_dropout_removes_the_transcript_from_every_option():
    """CFG's null branch: with the condition dropped no option (pooled text, gates, skips, final adaLN) may
    carry the transcript, and forward's dropout agrees with the sampler's cached zero conditions."""
    model = perturbed(FlowTTS(ModelConfig(**BASE, **ALL))).eval()
    batch = nano_batch(prompts=(3, 4))
    x, kwargs = batch["latents"], inputs(batch)
    time, drop = torch.tensor([0.3, 0.7]), torch.ones(2, dtype=torch.bool)
    tokens = kwargs["tokens"]
    other = dict(kwargs, tokens=tokens.masked_fill(tokens >= BYTE_OFFSET, BYTE_OFFSET + ord("x")))  # same length
    null = model(x, time, **kwargs, drop=drop)
    assert torch.equal(model(x, time, **other, drop=drop), null)
    assert not torch.allclose(model(x, time, **other), model(x, time, **kwargs), atol=1e-4)
    cond = model.conditions(kwargs["prompt"], kwargs["prompt_mask"], kwargs["tokens"], kwargs["segments"])
    zeros = (torch.zeros_like(cond[0]), cond[1], torch.zeros_like(cond[2]))
    blank = x.masked_fill(batch["prompt_mask"][..., None], 0)
    stripped = dict(kwargs, prompt=torch.zeros_like(x), prompt_mask=torch.zeros_like(batch["prompt_mask"]))
    assert torch.allclose(model(blank, time, **stripped, cached=zeros), null, atol=1e-6)
    guided = sample(model, **kwargs, steps=2, guidance=2.0, seed=1)
    assert torch.isfinite(guided).all() and torch.equal(guided[batch["prompt_mask"]], x[batch["prompt_mask"]])


def test_example_configs_change_one_model_option_of_the_w512_recipe():
    base = Config.load("configs/nano_tr_w512.yaml").to_dict()
    for name, overrides in OPTIONS.items():
        experiment = Config.load(f"configs/experiments/tr_w512_{name}.yaml").to_dict()
        assert experiment["train"] == base["train"], name
        changed = {k: v for k, v in experiment["model"].items() if base["model"][k] != v}
        assert changed == overrides, name
    replicate = Config.load("configs/experiments/tr_w512_seed43.yaml").to_dict()
    assert replicate["model"] == base["model"] and replicate["train"] == {**base["train"], "seed": 43}
