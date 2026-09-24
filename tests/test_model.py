import warnings
from pathlib import Path

import pytest
import torch

from dacvae_tts.config import Config, ModelConfig
from dacvae_tts.data import collate
from dacvae_tts.model import FlowTTS, flow_loss, sample
from dacvae_tts.training import Objective


def model_and_batch():
    model = FlowTTS(ModelConfig(latent_dim=4, width=32, heads=2, depth=2, text_depth=1))
    batch = collate(
        [
            {
                "reference": torch.randn(3, 4),
                "target": torch.randn(6, 4),
                "text": "Target words.",
                "reference_text": "Reference.",
            },
            {
                "reference": torch.randn(4, 4),
                "target": torch.randn(3, 4),
                "text": "Hi!",
                "reference_text": "Prompt.",
            },
        ]
    )
    return model, batch


def test_backward_all_parameters_and_checkpointing():
    model, batch = model_and_batch()
    model.grad_checkpoint = True
    losses = Objective(model)(batch)
    losses["loss"].mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.output[-1].weight.grad.abs().sum() > 0


def test_padding_invariance_and_odd_patch_boundary():
    model, batch = model_and_batch()
    # Exercise all attention paths (AdaLN/output start at zero by design).
    for block in model.blocks:
        torch.nn.init.normal_(block.ada[-1].weight, std=0.02)
    torch.nn.init.normal_(model.output[-1].weight, std=0.02)
    model.eval()
    kwargs = {k: v for k, v in batch.items() if k != "latents"}
    x = batch["latents"]
    time = torch.full((2,), 0.5)
    pred = model(x, time, **kwargs)
    short = {
        k: v[1:2, :7] if k not in ("tokens", "segments") else v[1:2, : int(batch["tokens"][1].ne(0).sum())]
        for k, v in kwargs.items()
    }
    single = model(x[1:2, :7], time[1:2], **short)
    assert torch.allclose(pred[1, :7], single[0], atol=2e-6)


def test_sampler_preserves_prompt_and_is_reproducible():
    model, batch = model_and_batch()
    kwargs = {k: v for k, v in batch.items() if k != "latents"}
    first = sample(model.eval(), **kwargs, steps=3, seed=10)
    second = sample(model, **kwargs, steps=3, seed=10)
    assert torch.equal(first, second)
    assert torch.equal(first[batch["prompt_mask"]], batch["prompt"][batch["prompt_mask"]])
    assert (first[~batch["valid"]] == 0).all()


def test_loss_ignores_padded_target_values():
    model, batch = model_and_batch()
    x = batch["latents"]
    noise, time = torch.randn_like(x), torch.ones(2) * 0.5
    original = flow_loss(model, batch, 0, time, noise)
    batch["latents"] = x.masked_fill(~batch["valid"][..., None], 1e4)
    changed = flow_loss(model, batch, 0, time, noise)
    assert torch.equal(original, changed)


def test_tiny_overfit_reduces_fixed_flow_loss():
    model, batch = model_and_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    noise, time = torch.randn_like(batch["latents"]), torch.full((2,), 0.5)
    initial = flow_loss(model, batch, 0, time, noise).mean().item()
    for _ in range(40):
        optimizer.zero_grad()
        loss = flow_loss(model, batch, 0, time, noise).mean()
        loss.backward()
        optimizer.step()
    assert loss.item() < initial * 0.3


def test_byte_ctc_on_packed_frames_warns_but_loads():
    # 12.5 packed frames/s against 16-19 bytes/s: zero_infinity used to zero most CTC rows without a word.
    with pytest.warns(UserWarning, match="ctc_targets: chars or patch_size: 1"):
        ModelConfig(ctc_layer=4, patch_size=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ModelConfig(ctc_layer=4, patch_size=1)
        ModelConfig(ctc_layer=4, patch_size=2, ctc_targets="chars")
        ModelConfig(ctc_layer=0, patch_size=2)
    # Every shipped model config still loads; none of them trains byte CTC on packed frames.
    configs = Path(__file__).resolve().parents[1] / "configs"
    paths = sorted(configs.glob("*.yaml")) + sorted((configs / "experiments").glob("*.yaml"))
    loaded = {}
    for path in paths:
        if path.name != "evaluation.yaml":  # the evaluation-run settings, not a model config
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                loaded[path.name] = Config.load(path)
    assert {name for name, cfg in loaded.items() if cfg.model.patch_size == 2} >= {"small.yaml", "tiny.yaml"}
    assert len(loaded) == len(paths) - 1
