"""Opt-in nano recipe: joined text, within-utterance prompts, rotary positions, EDM target, CTC head."""

import pytest
import torch

from dacvae_tts.config import Config, ModelConfig, TrainConfig
from dacvae_tts.data import LatentDataset, collate
from dacvae_tts.model import FlowTTS, flow_loss, sample, sample_time, to_velocity
from dacvae_tts.text import BOS, BYTE_OFFSET, EOS, tokenize, tokenize_bytes
from dacvae_tts.training import Objective

NANO = dict(
    latent_dim=4,
    width=32,
    heads=2,
    depth=2,
    text_depth=1,
    text_attention=1,
    patch_size=1,
    positions="rope",
    qk_norm=True,
    prediction="edm",
    text_layout="joined",
    duration="rule",
    ctc_layer=1,
)


def nano_batch(prompts=(3, 0), frames=9):
    torch.manual_seed(0)
    items = []
    for index, prompt in enumerate(prompts):
        latents = torch.randn(frames + 2 * index, 4)
        items.append(
            dict(
                reference=latents[:prompt],
                target=latents[prompt:],
                reference_text="",
                text=f"Spoken words number {index}.",
                layout="joined",
            )
        )
    return collate(items)


def randomized(model):
    for block in model.blocks:
        last = block.ada_up if hasattr(block, "ada_up") else block.ada[-1]
        torch.nn.init.normal_(last.weight, std=0.05)
    if getattr(model, "ada_shared", None) is not None:
        torch.nn.init.normal_(model.ada_shared[-1].weight, std=0.05)
    torch.nn.init.normal_(model.output[-1].weight, std=0.05)
    return model


def test_joined_layout_has_no_boundary_token():
    tokens, segments = tokenize_bytes(b"ab", b"cd", "joined")
    assert tokens.tolist() == [BOS, *(BYTE_OFFSET + v for v in b"ab cd"), EOS]
    assert segments.tolist() == [1] * 7
    assert torch.equal(tokenize("", "cd", layout="joined")[0], tokenize_bytes(b"", b"cd", "joined")[0])
    assert tokenize_bytes(b"ab", b"cd")[0].tolist()[3] == 2  # the segment layout keeps its SEP token
    with pytest.raises(ValueError):
        tokenize_bytes(b"", b"cd", "unknown")


def test_configuration_guards():
    with pytest.raises(ValueError):
        ModelConfig(text_layout="joined")  # the duration head needs separate transcripts
    with pytest.raises(ValueError):
        ModelConfig(positions="rope", width=30, heads=2)  # odd head width
    with pytest.raises(ValueError):
        Config(ModelConfig(), TrainConfig(pairing="within"))
    Config(ModelConfig(text_layout="joined", duration="rule"), TrainConfig(pairing="within"))


def test_within_pairing_cuts_the_prompt_from_the_same_utterance(cache):
    data = LatentDataset(cache, "train", pairing="within", layout="joined", prompt_dropout=0.0)
    assert (data.costs == data.lengths).all()
    item = data[(0, 0)]
    whole = data.row(0)["latents"]
    assert torch.equal(torch.cat([item["reference"], item["target"]]), whole)
    assert len(item["target"]) >= 1 and item["uid"] == item["reference_uid"]
    assert torch.equal(data[(0, 0)]["reference"], item["reference"])  # deterministic per (epoch, index)
    dropped = LatentDataset(cache, "train", pairing="within", layout="joined", prompt_dropout=1.0)
    assert all(len(dropped[(0, i)]["reference"]) == 0 for i in range(len(dropped)))
    batch = collate([dropped[(0, 0)], data[(0, 1)]])
    assert not batch["prompt_mask"][0].any() and batch["valid"][0].any()
    with pytest.raises(ValueError):
        LatentDataset(cache, "train", pairing="within")  # needs the joined layout


def test_nano_model_trains_and_ignores_padding():
    model = randomized(FlowTTS(ModelConfig(**NANO)))
    batch = nano_batch()
    losses = Objective(model, time_sampling="logit_normal", expansion=3, ctc_weight=0.1).train()(batch)
    assert losses["flow"].shape == (6,) and losses["duration"].shape == (2,) and losses["ctc"].shape == (6,)
    assert not losses["duration"].any()
    (losses["loss"].mean() + 0.1 * losses["ctc"].mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    model.eval()
    kwargs = {k: v for k, v in batch.items() if k != "latents"}
    time = torch.tensor([0.3, 0.7])
    full = model(batch["latents"], time, **kwargs)
    length = int(batch["valid"][0].sum())
    text_length = int(batch["tokens"][0].ne(0).sum())
    short = {
        k: (v[:1, :text_length] if k in ("tokens", "segments") else v[:1, :length]) for k, v in kwargs.items()
    }
    alone = model(batch["latents"][:1, :length], time[:1], **short)
    assert torch.allclose(full[0, :length], alone[0], atol=1e-5)  # length-aware positions ignore padding


def test_edm_target_recovers_the_true_velocity():
    torch.manual_seed(0)
    model = FlowTTS(ModelConfig(**NANO))
    x1, noise = torch.randn(3, 5, 4), torch.randn(3, 5, 4)
    time = torch.tensor([0.0, 0.4, 0.999])
    t = time[:, None, None]
    xt = (1 - t) * noise + t * x1
    ideal = ((1 - t) * x1 - t * noise) / (t.square() + (1 - t).square()).sqrt()
    assert torch.allclose(to_velocity(model, ideal, xt, time), x1 - noise, atol=1e-4)
    assert abs(float(ideal.std()) - 1) < 0.25  # unit-variance target at every t
    plain = FlowTTS(ModelConfig(latent_dim=4, width=32, heads=2, depth=1, text_depth=1))
    assert to_velocity(plain, ideal, xt, time) is ideal


def test_stratified_logit_normal_time():
    torch.manual_seed(0)
    time = sample_time(4096, torch.device("cpu"), "logit_normal")
    assert 0 < time.min() and time.max() < 1 and abs(float(time.mean()) - 0.5) < 0.02
    assert float(((time > 0.25) & (time < 0.75)).float().mean()) > 0.6  # mass in the middle
    with pytest.raises(ValueError):
        sample_time(4, torch.device("cpu"), "other")


def test_batched_guidance_matches_two_separate_passes():
    model = randomized(FlowTTS(ModelConfig(**NANO))).eval()
    batch = nano_batch(prompts=(3, 4))
    kwargs = {k: v for k, v in batch.items() if k != "latents"}
    noise = torch.randn_like(batch["latents"])
    guided = sample(model, **kwargs, steps=3, guidance=2.0, initial_noise=noise)
    conditioned = sample(model, **kwargs, steps=1, guidance=1.0, initial_noise=noise, times=[0.0, 1.0])
    assert torch.isfinite(guided).all() and not torch.allclose(guided, conditioned)
    assert torch.equal(guided[batch["prompt_mask"]], batch["prompt"][batch["prompt_mask"]])
    # One Euler step by hand: u + g (v - u) from two independent forward passes.
    cond = model.conditions(kwargs["prompt"], kwargs["prompt_mask"], kwargs["tokens"], kwargs["segments"])
    null = (torch.zeros_like(cond[0]), cond[1], torch.zeros_like(cond[2]))
    start = torch.where(batch["prompt_mask"][..., None], batch["prompt"], noise) * batch["valid"][..., None]
    zero = torch.zeros(2)
    v = to_velocity(model, model(start, zero, **kwargs, cached=cond), start, zero)
    blank = start.masked_fill(batch["prompt_mask"][..., None], 0)
    stripped = dict(
        kwargs, prompt=torch.zeros_like(kwargs["prompt"]), prompt_mask=torch.zeros_like(kwargs["prompt_mask"])
    )
    u = to_velocity(model, model(blank, zero, **stripped, cached=null), blank, zero)
    target = batch["valid"] & ~batch["prompt_mask"]
    manual = start + (u + 2.0 * (v - u))
    one_step = sample(model, **kwargs, steps=1, guidance=2.0, initial_noise=noise, times=[0.0, 1.0])
    assert torch.allclose(one_step[target], manual[target], atol=1e-5)


def test_ctc_excludes_examples_without_text():
    model = randomized(FlowTTS(ModelConfig(**NANO))).train()
    batch = nano_batch(frames=64)  # CTC needs at least as many frames as transcript bytes
    details = flow_loss(model, batch, dropout=1.0, return_details=True)
    assert not details["ctc"].any()  # every example lost its text: nothing to align
    details = flow_loss(model, batch, dropout=0.0, return_details=True)
    assert (details["ctc"] > 0).all()


def test_corrupt_transcript_skips_or_repeats_one_word():
    import random

    from dacvae_tts.text import SPACE, corrupt_transcript

    tokens, segments = tokenize_bytes(b"", b"one two three", "joined")
    words = lambda t: bytes(int(v) - BYTE_OFFSET for v in t[1:-1]).decode()  # noqa: E731
    seen = set()
    for seed in range(40):
        corrupted = corrupt_transcript(tokens, segments, random.Random(seed))
        assert corrupted is not None
        new_tokens, new_segments = corrupted
        assert new_tokens.shape == new_segments.shape and (new_segments == 1).all()
        assert new_tokens[0] == BOS and new_tokens[-1] == EOS
        text = words(new_tokens)
        assert text != "one two three" and "  " not in text and SPACE not in (new_tokens[1], new_tokens[-2])
        seen.add(text)
    assert {"two three", "one three", "one two"} & seen and {"one one two three", "one two two three"} & seen
    assert corrupt_transcript(*tokenize_bytes(b"", b"single", "joined"), random.Random(0)) is None
    # Segment layout: only the target transcript changes.
    tokens, segments = tokenize_bytes(b"ref words", b"aa bb")
    new_tokens, new_segments = corrupt_transcript(tokens, segments, random.Random(1))
    assert torch.equal(new_tokens[: int((segments == 0).sum())], tokens[: int((segments == 0).sum())])


def test_contrastive_and_low_rank_adaln():
    model = randomized(FlowTTS(ModelConfig(**NANO, adaln_rank=8))).train()
    assert model.ada_shared is not None and not hasattr(model.blocks[0], "ada")
    batch = nano_batch(frames=64)
    objective = Objective(model, expansion=2, ctc_weight=0.1, contrastive_weight=0.2).train()
    losses = objective(batch)
    assert losses["contrastive"].shape == (4,) and (losses["contrastive"] >= 0).all()
    auxiliary = objective.auxiliary(losses)
    (losses["loss"].mean() + auxiliary.mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.ada_shared[-1].weight.grad.abs().sum() > 0
    # No contrastive term outside training or when disabled.
    assert "contrastive" not in Objective(model, contrastive_weight=0.2).eval()(batch)
    assert "contrastive" not in Objective(model).train()(batch)
