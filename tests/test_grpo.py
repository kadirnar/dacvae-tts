"""Flow-GRPO post-training (dacvae_tts.grpo): SDE policy math, group rewards, clipped update, CLI, monitor, oracle."""

import math
import os
import random

import numpy as np
import pytest
import torch

from dacvae_tts.config import Config, ModelConfig, TrainConfig
from dacvae_tts.data import LatentDataset
from dacvae_tts.grpo import (
    CompositeReward,
    Judges,
    PromptSource,
    build_terms,
    choose_window,
    gaussian_kl,
    group_advantages,
    parse_spec,
    ppo_clip_loss,
    sde_sample,
    sde_sigma,
    sde_step,
    transition_log_prob,
)
from dacvae_tts.model import FlowTTS, sample
from dacvae_tts.text import tokenize

NANO = dict(
    latent_dim=4, width=32, heads=2, depth=2, text_depth=1, text_attention=1, patch_size=1, positions="rope",
    qk_norm=True, prediction="edm", text_layout="joined", duration="rule", ctc_layer=1,
)


def masks():
    valid = torch.tensor([[1] * 7, [1] * 5 + [0] * 2], dtype=torch.bool)
    prompt_mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    return valid, prompt_mask, valid & ~prompt_mask


def trained_looking_model():
    torch.manual_seed(3)
    model = FlowTTS(ModelConfig(**NANO))
    # Zero-initialized AdaLN/output would make every branch identical; perturb them so guidance matters.
    for block in model.blocks:
        torch.nn.init.normal_(block.ada[-1].weight, std=0.05)
    torch.nn.init.normal_(model.output[-1].weight, std=0.05)
    return model.eval()


def padded_condition():
    """Two prompts of different reference/target lengths (padding exercised)."""
    reference_frames, target_frames = (5, 3), (7, 4)
    total = [r + t for r, t in zip(reference_frames, target_frames)]
    width = max(total)
    prompt, prompt_mask = torch.zeros(2, width, 4), torch.zeros(2, width, dtype=torch.bool)
    for row, frames in enumerate(reference_frames):
        prompt[row, :frames] = torch.randn(frames, 4)
        prompt_mask[row, :frames] = True
    texts = ("Bir.", "İki üç.")
    encoded = [tokenize("Referans cümlesi.", text, version="turkish-v1", layout="joined") for text in texts]
    pad = torch.nn.utils.rnn.pad_sequence
    return dict(
        prompt=prompt,
        prompt_mask=prompt_mask,
        valid=torch.arange(width)[None] < torch.tensor(total)[:, None],
        tokens=pad([t for t, _ in encoded], batch_first=True),
        segments=pad([s for _, s in encoded], batch_first=True),
    )


def test_transition_log_prob_matches_normal_over_target_frames():
    _, _, target = masks()
    mean, value, std = torch.randn(2, 7, 3), torch.randn(2, 7, 3), 0.37
    density = torch.distributions.Normal(mean, std).log_prob(value)
    expected = torch.stack([density[row][target[row]].sum() for row in range(2)])
    total = transition_log_prob(value, mean, std, target, "sum")
    assert torch.allclose(total, expected, atol=1e-4)
    assert torch.allclose(transition_log_prob(value, mean, std, target), expected / (target.sum(1) * 3), atol=1e-6)
    # Prompt and padding frames carry no action: arbitrary values there change nothing.
    garbage = value.masked_fill(~target[..., None], 1e6)
    assert torch.equal(transition_log_prob(garbage, mean, std, target, "sum"), total)


def test_gaussian_kl_is_the_closed_form():
    _, _, target = masks()
    mean, reference, std = torch.randn(2, 7, 3), torch.randn(2, 7, 3), 0.2
    normal = torch.distributions.Normal
    kl = torch.distributions.kl_divergence(normal(mean, std), normal(reference, std))
    expected = torch.stack([kl[row][target[row]].sum() for row in range(2)])
    assert torch.allclose(gaussian_kl(mean, reference, std, target, "sum"), expected, rtol=1e-5)
    assert torch.equal(gaussian_kl(mean, mean, std, target), torch.zeros(2))


def test_sde_step_drift_and_its_ode_limit():
    _, _, target = masks()
    x, v = torch.randn(2, 7, 3), torch.randn(2, 7, 3)
    t0, t1 = torch.tensor(0.3), torch.tensor(0.45)
    mean, std = sde_step(x, v, t0, t1, 0.0, target)
    assert std == 0 and torch.equal(mean, x + (t1 - t0) * v.masked_fill(~target[..., None], 0))
    # The drift correction is (sigma^2/2) * score with the clean-estimate form of the score.
    sigma = 0.6
    x1_hat = x + (1 - t0) * v
    score = -(x - t0 * x1_hat) / (1 - t0) ** 2
    mean, std = sde_step(x, v, t0, t1, sigma, target)
    expected = x + (t1 - t0) * (v + sigma**2 / 2 * score).masked_fill(~target[..., None], 0)
    assert torch.allclose(mean, expected, atol=1e-6) and math.isclose(std, sigma * math.sqrt(0.15), rel_tol=1e-6)
    near, _ = sde_step(x, v, t0, t1, 1e-4, target)
    assert torch.allclose(near, x + (t1 - t0) * v.masked_fill(~target[..., None], 0), atol=1e-6)
    assert sde_sigma(0.25, "flow", 0.7) == pytest.approx(0.7 * math.sqrt(3))
    assert sde_sigma(0.0, "flow", 0.7, t_min=0.25) == pytest.approx(0.7 * math.sqrt(3))
    assert sde_sigma(0.9, "constant", 0.5) == 0.5


@pytest.mark.parametrize("guidance", [1.0, 2.5])
def test_sde_sampler_is_the_deployed_sampler_without_noise(guidance):
    model, condition = trained_looking_model(), padded_condition()
    noise = torch.randn(condition["prompt"].shape)
    reference = sample(model, **condition, steps=5, guidance=guidance, initial_noise=noise, guidance_until=0.7)
    ode, transitions = sde_sample(model, condition, 5, guidance, -1.0, (), initial_noise=noise, guidance_until=0.7)
    assert torch.equal(ode, reference) and transitions == []
    near, transitions = sde_sample(model, condition, 5, guidance, -1.0, (1, 2), 1e-7, initial_noise=noise,
                                   guidance_until=0.7)
    assert torch.allclose(near, reference, atol=1e-5) and [t["step"] for t in transitions] == [1, 2]
    noisy, transitions = sde_sample(model, condition, 5, guidance, -1.0, (1, 2), 0.5, initial_noise=noise,
                                    generator=torch.Generator().manual_seed(0), guidance_until=0.7)
    assert not torch.allclose(noisy, reference, atol=1e-3)
    assert torch.equal(noisy[condition["prompt_mask"]], condition["prompt"][condition["prompt_mask"]])
    assert (noisy[~condition["valid"]] == 0).all()
    assert torch.equal(transitions[0]["next"], transitions[1]["state"])
    assert all(torch.isfinite(t["log_prob"]).all() and t["log_prob"].shape == (2,) for t in transitions)


def gaussian_velocity(x, t, m, s):
    """Exact velocity of the linear path for data N(m, s^2): (E[x1 | x_t] - x) / (1 - t)."""
    variance = (1 - t) ** 2 + t**2 * s**2
    x1 = m + t * s**2 / variance * (x - t * m)
    return (x1 - x) / (1 - t)


@pytest.mark.parametrize("schedule,level", [("constant", 0.8), ("flow", 0.7)])
def test_sde_keeps_the_ode_marginals_on_gaussian_data(schedule, level):
    torch.manual_seed(0)
    m, s, count, steps = 1.5, 0.5, 40000, 200
    times = torch.linspace(0, 1, steps + 1)
    mask = torch.ones(count, 1, dtype=torch.bool)

    def run(correct):
        x = torch.randn(count, 1, 1)
        for t0, t1 in zip(times[:-1], times[1:]):
            v = gaussian_velocity(x, t0, m, s)
            sigma = sde_sigma(t0, schedule, level, float(times[1])) if t0 < 0.8 else 0.0
            if correct:
                mean, std = sde_step(x, v, t0, t1, sigma, mask)
            else:  # noise without the score correction
                mean, std = x + (t1 - t0) * v, sigma * math.sqrt(float(t1 - t0))
            x = mean + std * torch.randn_like(x)
        return x

    x = run(True)
    assert abs(x.mean().item() - m) < 0.02 and abs(x.std().item() - s) < 0.02
    assert abs(run(False).std().item() - s) > 0.1  # the test can tell a wrong drift apart


def test_window_choice_stays_in_the_early_part_of_the_grid():
    rng = random.Random(0)
    windows = {choose_window(16, 2, 0.5, rng) for _ in range(200)}
    assert windows == {tuple(range(start, start + 2)) for start in range(7)}
    assert choose_window(4, 4, 1.0, rng) == (0, 1, 2, 3)
    with pytest.raises(ValueError):
        choose_window(16, 3, 0.1, rng)


def test_group_advantages_standardize_each_term_with_floors_and_signs():
    raw = {"cer": np.array([0.0, 0.0, 0.2, 0.0]), "sim": np.array([0.70, 0.71, 0.70, 0.705])}
    weights, signs, floors = {"cer": 1.0, "sim": 0.5}, {"cer": -1}, {"cer": 0.01, "sim": 0.01}
    z_cer = -(raw["cer"] - raw["cer"].mean()) / raw["cer"].std()
    z_sim = (raw["sim"] - raw["sim"].mean()) / 0.01  # spread 0.004 is below the floor
    weighted = group_advantages(raw, weights, signs, floors, "weighted")
    assert np.allclose(weighted, (z_cer + 0.5 * z_sim) / 1.5)
    advantages = group_advantages(raw, weights, signs, floors)
    assert abs(advantages.mean()) < 1e-9 and abs(advantages.std() - 1) < 1e-3
    assert np.argmin(advantages) == 2  # the sample with transcription errors is the one pushed down
    flat = {"cer": np.zeros(4), "sim": np.full(4, 0.7)}
    assert not group_advantages(flat, weights, signs, floors).any()


def test_composite_reward_requires_two_terms_and_handles_failed_scores():
    class Error:
        sign = -1

        def __call__(self, group):
            return [0.1, float("nan"), 0.0, 0.3]

    def quality(group):
        return [3.0, 3.1, 2.9, 3.0]

    with pytest.raises(ValueError, match="at least 2"):
        CompositeReward({"cer": Error()}, {"cer": 1.0})
    with pytest.raises(ValueError, match="without an implementation"):
        CompositeReward({"cer": Error()}, {"cer": 1.0, "utmos": 0.4})
    reward = CompositeReward({"cer": Error(), "mos": quality}, {"cer": 1.0, "mos": 0.4, "unused": 0.0})
    raw = reward.score([None] * 4)
    assert raw["cer"][1] == 0.3  # a failed judgement counts as the worst sample (highest error)
    assert reward.signs == {"cer": -1, "mos": 1} and set(raw) == {"cer", "mos"}
    assert reward.advantages(raw).shape == (4,)
    assert CompositeReward({"mos": quality}, {"mos": 1.0}, min_terms=1).weights == {"mos": 1.0}


def test_ppo_clip_loss_gradient_signs():
    clip = 0.1
    cases = [  # (ratio, advantage, expected d loss / d log pi)
        (1.0, 1.0, -1.0),  # on-policy the gradient is the policy gradient -A
        (1.0, -1.0, 1.0),
        (0.8, 1.0, -0.8),  # a disfavoured good sample is still pushed up
        (1.2, -1.0, 1.2),  # a favoured bad sample is still pushed down
        (1.2, 1.0, 0.0),  # beyond 1 + clip in the rewarded direction: no further incentive
        (0.8, -1.0, 0.0),
    ]
    for ratio, advantage, expected in cases:
        log_ratio = torch.tensor([math.log(ratio)], requires_grad=True)
        ppo_clip_loss(log_ratio, torch.tensor([advantage]), clip).sum().backward()
        assert log_ratio.grad.item() == pytest.approx(expected, abs=1e-6), (ratio, advantage)


def test_reward_spec_and_term_factory():
    assert parse_spec("cer=1, sim=0.5,os.path:basename=0.2") == {"cer": 1.0, "sim": 0.5, "os.path:basename": 0.2}
    assert parse_spec("") == {}
    for bad in ("cer", "=1", "cer=nan"):
        with pytest.raises(ValueError):
            parse_spec(bad)
    judges = Judges("cpu")
    assert build_terms(["os.path:basename"], judges, "tiny", "sv")["os.path:basename"] is os.path.basename
    with pytest.raises(ValueError, match="dnsmos-model"):
        build_terms(["dnsmos"], judges, "tiny", "sv")
    with pytest.raises(ValueError, match="Unknown reward term"):
        build_terms(["pesq"], judges, "tiny", "sv")


def toy_checkpoint(cache, path):
    data = LatentDataset(cache)
    cfg = Config(ModelConfig(latent_dim=4, width=16, depth=1, heads=2, text_depth=1), TrainConfig())
    model = FlowTTS(cfg.model)
    state = model.state_dict()
    torch.save({"model": state, "ema": state, "config": cfg.to_dict(), "codec": data.meta, "mean": data.mean,
                "std": data.std}, path)
    return path


def test_prompts_pair_two_recordings_of_one_speaker_with_the_rule_length(cache):
    saved = torch.load(toy_checkpoint(cache, cache.parent / "base.pt"), weights_only=True)
    source = PromptSource(cache, "val", saved, duration_mode="rule")
    rng = random.Random(0)
    for index in range(len(source.data)):
        item = source.item(index, rng)
        assert item["uid"] == source.data.row(index)["uid"]  # the target is row `index`
        assert item["reference"].shape != item["target"].shape or not torch.equal(item["reference"], item["target"])
    prompts = source.fixed(5, seed=3)
    assert [p.uid for p in prompts] == [p.uid for p in source.fixed(5, seed=3)] and len(prompts) == 5
    for prompt in prompts:
        frames = int(prompt.target.sum())
        reference_frames = int(prompt.condition["prompt_mask"].sum())
        # Same text length in bytes for every cache sentence: the rule keeps the prompt's frames per byte.
        assert frames == reference_frames == len(prompt.reference)
