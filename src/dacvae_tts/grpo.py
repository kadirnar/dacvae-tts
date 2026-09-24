"""Online RL post-training: Flow-GRPO with a composite, group-standardized reward (WER/CER + SIM + DNSMOS + UTMOS).

Why. Most of the quality gap (DNSMOS 2.89 vs the 3.27 codec ceiling) is generation error, and fine-tuning on the
high-quality subset did not raise DNSMOS. For flow-matching TTS the strongest training-side lever with evidence is
online RL on several metrics at once: FlowTTS-GRPO (arXiv:2606.23190, F5-TTS; reward WER + SIM + DNSMOS P.835 with
lambda 0.4, group 10, 16 steps, SDE sigma 0.5 on a 2-step window, 1289 updates on 8 GPUs) moved DNSMOS 3.154 -> 3.408,
WER 1.88 -> 1.73 and SIM 0.753 -> 0.790; F5R-TTS (arXiv:2504.02407) reports WER -29.5 % relative, SIM +4.6 %. The
offline `posttrain` preference loss is a flow-error surrogate, not a policy gradient.

Policy (Flow-GRPO, arXiv:2505.05470, re-derived for this repository's time: t=0 noise, t=1 data). The linear path
x_t = (1-t) eps + t x1 has velocity v = E[x1 - eps | x_t], clean estimate x1_hat = x + (1-t) v, noise estimate
eps_hat = x - t v, and score grad log p_t(x) = -E[eps | x_t] / (1-t) = -(x - t v) / (1-t) = -(x - t x1_hat) / (1-t)^2.
For any sigma_t the SDE

    dx = [v + (sigma_t^2 / 2) grad log p_t(x)] dt + sigma_t dW

keeps the marginals of the ODE dx = v dt: its Fokker-Planck equation dp/dt = -div(p v) - (sigma^2/2) div(grad p)
+ (sigma^2/2) lap p = -div(p v) is the ODE's continuity equation (checked numerically on Gaussian data, where v is
exact, in tests/test_grpo.py). Euler-Maruyama from t0 to t1 = t0 + dt makes one sampler step a Gaussian policy

    x_t1 ~ N(mu, s^2 I),   mu = x + dt [v - sigma^2 (x - t0 v) / (2 (1 - t0))],   s = sigma sqrt(dt),

with an exact log-density that is differentiable in the network through v; sigma -> 0 is the Euler step of
`model.sample` (tested bit-exactly). `sde_sigma` gives a constant sigma (FlowTTS-GRPO: 0.5, the default) or Flow-GRPO's
sigma_t = a sqrt((1-t)/t) (their a sqrt(t/(1-t)) in reversed time: large near the noise, zero at the data; at t=0 the
first grid time stands in for t, as their `sigma_max` substitution does).

Only a window of `sde_window` consecutive steps is stochastic; the other steps are the deterministic Euler ODE of
inference (FlowTTS-GRPO's 2-step window, Flow-GRPO-Fast/MixGRPO). The window start is drawn uniformly per rollout step
among the steps ending in the first `window_max` of the grid, where the trajectory is still undecided. By default the G
samples of a group share their initial noise (DanceGRPO, arXiv:2505.07818), so every reward difference inside a group
comes from the window noise the policy gradient credits; `--no-shared-noise` draws independent noise per sample
(larger spread, noisier credit). The policy is the sampler that is deployed: classifier-free guidance enters mu
through the guided velocity v_null + g (v_cond - v_null) of the joint conditional/null forward that `sample` uses,
the guidance interval included, and prompt frames are overwritten with the clean prompt after every step. Log-densities
and the KL run over target frames only (valid & ~prompt_mask); prompt and padding frames carry no action.

Objective (GRPO, DeepSeekMath arXiv:2402.03300, with the Flow-GRPO clipped surrogate). Per prompt (reference latents +
transcript, target text, target length from the deployed duration rule) G trajectories are sampled and scored. Each
reward term k is standardized inside the group, z_k = sign_k (r_k - mean) / max(std, floor_k), where the floor (about
one unit of measurement noise: CER 0.01, WER 0.02, SIM 0.01, MOS 0.05) stops a term whose samples barely differ from
turning noise into a full-size signal; the weighted sum R = sum_k w_k z_k is standardized again,
A = (R - mean) / max(std + 1e-4, c) (`--advantage standardized`), or kept as the weighted mean of the z-scores
(`--advantage weighted`, no second division, Dr. GRPO arXiv:2503.20783's argument against std normalization). The
composite floor c = min_k |w_k| is the spread R gets when a single term moves by exactly its floor: a group whose terms
all stay under their floors keeps its small advantages instead of being rescaled back to unit variance, while a group
with real spread (std >= c) is standardized as before. Groups whose advantages are all zero are skipped. The loss over
window steps k and samples i is

    L = mean_{i,k} [ -min(rho A_i, clip(rho, 1-eps, 1+eps) A_i) + beta KL(N(mu_theta, s^2) || N(mu_ref, s^2)) ],

rho = exp(log pi_theta - log pi_old), KL = |mu_theta - mu_ref|^2 / (2 s^2) in closed form against a frozen copy of the
starting model. Log-densities and the KL are averaged over target elements (`--logprob-reduction mean`, Flow-GRPO's
implementation): a sum over ~30k elements makes rho explode, and the mean keeps eps independent of utterance length;
eps 1e-4 and beta 0.04 are Flow-GRPO's values for this reduction. Gradients pass only through the window forwards,
recomputed with grad. Old log-densities and reference means are recomputed without grad right before the updates with
exactly the micro-batching of the gradient passes, so bf16 kernel choices cannot make rho != 1 on-policy; the gap to
the rollout's own log-density is logged (`logp_mismatch`). With the defaults (one PPO epoch, one optimizer step per
rollout) the update is exactly on-policy and the clip is inactive; `--ppo-epochs`/`--updates-per-rollout` > 1 reuse the
rollout off-policy under the clip. AdamW is the default optimizer here: Muon rescales every update of a matrix to unit
singular values, which turns the zero-mean noise of a policy gradient into full-size steps.

Rewards (`--reward-weights`, lazily loaded frozen judges): `cer`/`wer` from a batched transformers Whisper
(`--asr-model`, turbo by default) with the repository's metric normalization (turkish-v1 for --language tr), capped at
1; `sim`, cosine of speaker embeddings between each sample and the codec-decoded prompt (SIM-r); `dnsmos`, official
DNSMOS OVRL (`--dnsmos-model` sig_bak_ovr.onnx); `utmos`, UTMOS22-strong (SpeechMOS torch.hub; English-trained, a
proxy for Turkish); `distillmos` (optional `pip install distillmos`); `module:attr` imports any callable
`term(group) -> list[float]` (attribute `sign = -1` when lower is better). Reward hacking is real: with a SCOREQ-only
reward held-out UTMOS collapsed 4.51 -> 1.23 while an equal-weight WER + Distill-MOS + UTMOSv2 ensemble held
(arXiv:2609.13150), so at least two reward terms are required (`--allow-single-reward` overrides) and a held-out
monitor scores the EMA model with the deployed ODE sampler (`--monitor-steps`, default 32) using judges outside the
reward: Whisper large-v3 instead of turbo, WavLM-SV (the repository's SIM metric) instead of the reward's UniSpeech-SAT
SV, and optionally Distill-MOS. The existing `posttrain.reward` (0.125 per DNSMOS point vs 2 per WER unit) is left
unchanged; this composite is separate.

Commands (GPU machine; oracle first: best-of-N under the same reward is the ceiling reweighting can reach):

  python scripts/oracle_best_of_n.py --checkpoint runs/tr-w512-clean/step-0060000.pt --cache data/tr55/clean \\
      --output outputs/oracle-c60k --split val --limit 64 --candidates 8 --sample-steps 32 --guidance 5 \\
      --dnsmos-model models/sig_bak_ovr.onnx
  # the same with --sampler sde --sample-steps 16 shows the within-group reward spread GRPO will see

  dacvae-tts post-train --mode grpo --checkpoint runs/tr-w512-clean/step-0060000.pt --cache data/tr55/clean \\
      --output runs/tr-grpo --dnsmos-model models/sig_bak_ovr.onnx --steps 1000 --group-size 8 \\
      --prompts-per-step 4 --sample-steps 16 --guidance 5 --learning-rate 1e-5 --save-every 100

Watch `grpo-log.jsonl`: per-term reward means rise while `monitor` (held-out, other judges) follows; `reward_std`
(group spread), `degenerate_groups` and `floored_groups` (composite spread under c, degenerate ones included) say
whether sigma/window give a usable signal; `kl_ref` grows slowly; `logp_mismatch` stays near zero; listen to
`monitor/step-*/` audio. Stop when held-out metrics fall while the reward rises (hacking).
"""

import copy
import importlib
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .codec import backend_options, check_compatibility
from .contracts import mask_values, sanitize
from .data import LatentDataset, collate
from .duration import DurationPredictor, auto_mode, clamp_scale, rule_frames
from .model import guided_update, sample, set_dropout, time_grid, to_velocity
from .optim import build_optimizer
from .text import normalize
from .training import atomic_save, autocast, distributed_device, load_model

CONDITION_KEYS = ("prompt", "prompt_mask", "valid", "tokens", "segments")
# Smallest group spread that counts as signal, about one unit of each metric's measurement noise.
REWARD_FLOORS = {"cer": 0.01, "wer": 0.02, "sim": 0.01, "dnsmos": 0.05, "utmos": 0.05, "distillmos": 0.05}
UTMOS_ENTRY = "utmos22_strong"


# --------------------------------------------------------------------------------------------------------------------
# SDE policy: Euler-Maruyama step, Gaussian log-density, closed-form KL, clipped surrogate


def sde_sigma(t0, schedule="constant", level=0.5, t_min=None):
    """Diffusion coefficient sigma_t of a window step starting at t0 (t=0 noise)."""
    if not math.isfinite(level) or level < 0:
        raise ValueError("SDE sigma must be finite and nonnegative")
    if schedule == "constant":
        return float(level)
    if schedule != "flow":
        raise ValueError("sigma schedule must be constant or flow")
    t = max(float(t0), float(t_min or 0.0))
    if not 0 < t < 1:
        raise ValueError("The flow sigma schedule needs 0 < t < 1 (pass the first grid time as t_min)")
    return level * math.sqrt((1 - t) / t)


def sde_step(x, v, t0, t1, sigma, mask):
    """Mean and std of the marginal-preserving SDE step t0 -> t1: mu = x + dt [v + sigma^2/2 * score]."""
    dt = t1 - t0
    drift = v
    if sigma > 0:
        # score = -(x - t0 v) / (1 - t0); sigma = 0 leaves the Euler step of `sample` bit for bit.
        drift = v - (sigma**2 / (2 * (1 - t0))) * (x - t0 * v)
    mean = x + dt * sanitize(drift, mask)
    return mean, sigma * math.sqrt(float(dt))


def transition_log_prob(value, mean, std, mask, reduction="mean"):
    """log N(value; mean, std^2 I) over the target elements of [B,L,C] -> [B] (sum, or mean per element)."""
    if std <= 0:
        raise ValueError("A transition log-density needs std > 0")
    residual = sanitize(value.float() - mean.float(), mask)
    count = (mask.sum(1) * value.size(-1)).float()
    total = -residual.square().sum((1, 2)) / (2 * std**2) - count * (math.log(std) + 0.5 * math.log(2 * math.pi))
    return reduce_elements(total, count, reduction)


def gaussian_kl(mean, reference_mean, std, mask, reduction="mean"):
    """KL(N(mean, s^2 I) || N(reference_mean, s^2 I)) = |mean - reference_mean|^2 / (2 s^2) over target elements."""
    difference = sanitize(mean.float() - reference_mean.float(), mask)
    count = (mask.sum(1) * mean.size(-1)).float()
    return reduce_elements(difference.square().sum((1, 2)) / (2 * std**2), count, reduction)


def reduce_elements(total, count, reduction):
    if reduction == "mean":
        return total / count.clamp_min(1)
    if reduction != "sum":
        raise ValueError("reduction must be mean or sum")
    return total


def ppo_clip_loss(log_ratio, advantages, clip):
    """Per-sample -min(rho A, clip(rho) A): no incentive to move rho beyond 1 +- clip in the rewarded direction."""
    ratio = log_ratio.exp()
    return -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages)


def choose_window(steps, size, window_max, rng):
    """`size` consecutive step indices, start uniform among windows ending in the first `window_max` of the grid."""
    last = math.floor(window_max * steps + 1e-9) - size
    if size < 1 or last < 0:
        raise ValueError("The SDE window must fit into the first window-max fraction of the sampling steps")
    start = rng.randint(0, last)
    return tuple(range(start, start + size))


def is_guided(t0, guidance, guidance_from=0.0, guidance_until=1.0):
    return guidance != 1 and guidance_from <= float(t0) < guidance_until


def policy_velocity(model, x, t, condition, guidance, guided, cond=None, mask=None):
    """Velocity the sampler follows at x: `sample`'s joint conditional/null forward with plain CFG when guided."""
    prompt, prompt_mask, valid, tokens, segments = (condition[k] for k in CONDITION_KEYS)
    if cond is None:
        cond = model.conditions(prompt, prompt_mask, tokens, segments)
    if not guided:
        return to_velocity(model, model(x, t, prompt, prompt_mask, valid, tokens, segments, cached=cond), x, t)
    zeros = torch.zeros_like
    both, times = torch.cat([x, x.masked_fill(prompt_mask[..., None], 0)]), t.repeat(2)
    output = model(
        both,
        times,
        torch.cat([prompt, zeros(prompt)]),
        torch.cat([prompt_mask, zeros(prompt_mask)]),
        valid.repeat(2, 1),
        tokens.repeat(2, 1),
        segments.repeat(2, 1),
        cached=(torch.cat([cond[0], zeros(cond[0])]), cond[1].repeat(2, 1), torch.cat([cond[2], zeros(cond[2])])),
    )
    v, u = to_velocity(model, output, both, times).chunk(2)
    return guided_update(v, u, x, t, mask, guidance)


@torch.no_grad()
def sde_sample(
    model,
    condition,
    steps=16,
    guidance=1.5,
    sway=-1.0,
    window=(),
    sigma=0.5,
    schedule="constant",
    generator=None,
    initial_noise=None,
    guidance_from=0.0,
    guidance_until=1.0,
    reduction="mean",
):
    """`model.sample` with the steps in `window` made stochastic; returns (x, window transitions).

    Each transition holds the step index, t0/t1 (0-d tensors of the grid), sigma, the state before and after the step
    and the rollout's log-density. An empty window (or sigma 0) with the same initial noise is exactly `sample`.
    """
    prompt, prompt_mask, valid, tokens, segments = (condition[k] for k in CONDITION_KEYS)
    mask = mask_values(valid, prompt_mask)
    prompt = sanitize(prompt, prompt_mask)
    x = (
        torch.randn(prompt.shape, device=prompt.device, dtype=prompt.dtype, generator=generator)
        if initial_noise is None
        else initial_noise.clone()
    )
    if x.shape != prompt.shape:
        raise ValueError("initial_noise must match the prompt [B,L,C]")
    x = sanitize(torch.where(prompt_mask[..., None], prompt, x), valid)
    times = time_grid(steps, sway, x.device)
    cond = model.conditions(prompt, prompt_mask, tokens, segments)
    transitions = []
    for index, (t0, t1) in enumerate(zip(times[:-1], times[1:])):
        t = t0.expand(x.size(0))
        guided = is_guided(t0, guidance, guidance_from, guidance_until)
        v = policy_velocity(model, x, t, condition, guidance, guided, cond, mask)
        level = sde_sigma(t0, schedule, sigma, float(times[1])) if index in window else 0.0
        mean, std = sde_step(x, v, t0, t1, level, mask)
        state = x
        if level > 0:
            noise = torch.randn(x.shape, device=x.device, dtype=torch.float32, generator=generator)
            x = mean.float() + std * sanitize(noise, mask)
        else:
            x = mean
        x = sanitize(torch.where(prompt_mask[..., None], prompt, x), valid)
        if level > 0:
            transitions.append(
                {
                    "step": index,
                    "t0": t0,
                    "t1": t1,
                    "sigma": level,
                    "state": state,
                    "next": x,
                    "log_prob": transition_log_prob(x, mean, std, mask, reduction),
                }
            )
    return x, transitions


# --------------------------------------------------------------------------------------------------------------------
# Prompts, groups and audio decoding


@dataclass
class Prompt:
    """One synthesis request: condition tensors [1,...] on the policy device plus the strings the judges need."""

    condition: dict
    text: str
    reference_text: str
    reference: torch.Tensor  # [Lref, C] normalized prompt latents (CPU)
    uid: str = ""
    speaker: str = ""

    @property
    def target(self):
        return self.condition["valid"][0] & ~self.condition["prompt_mask"][0]


def repeat_rows(condition, rows):
    return {key: value.expand(rows, *value.shape[1:]) for key, value in condition.items()}


class Group:
    """The G samples of one prompt as reward terms see them; decoded audio and ASR results are memoized."""

    def __init__(self, prompt, latents, decoder=None):
        self.prompt, self.latents, self.decoder = prompt, list(latents), decoder
        self._memo = {}

    def __len__(self):
        return len(self.latents)

    @property
    def text(self):
        return self.prompt.text

    @property
    def reference_text(self):
        return self.prompt.reference_text

    def memo(self, key, compute):
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]

    def audio(self):
        """16 kHz float32 waveforms of the samples (the rate of every judge)."""
        if self.decoder is None:
            raise ValueError("This reward needs audio; build the group with a LatentDecoder")
        return self.memo("audio", lambda: [self.decoder(z) for z in self.latents])

    def reference_audio(self):
        if self.decoder is None:
            raise ValueError("This reward needs audio; build the group with a LatentDecoder")
        return self.memo("reference_audio", lambda: self.decoder(self.prompt.reference))


class LatentDecoder:
    """Normalized latents -> waveform through the frozen codec, loaded on first use (latent-only rewards never pay)."""

    def __init__(self, saved, device, codec_options=None, rate=16000):
        self.info, self.device, self.rate = saved["codec"], torch.device(device), rate
        self.mean, self.std = saved["mean"].to(self.device), saved["std"].to(self.device)
        self.options, self._codec = dict(codec_options or {}), None

    @property
    def codec(self):
        if self._codec is None:
            from .codec import Codec

            options = dict(self.options)
            if self.info.get("loudness_lufs") is not None:
                options["loudness"] = self.info["loudness_lufs"]
            self._codec = Codec(self.info["checkpoint"], self.device, **options)
            check_compatibility(self._codec.metadata, self.info)
        return self._codec

    def full(self, latents):
        """Waveform at the codec's own rate (for listening checks)."""
        return self.codec.decode(latents.float().to(self.device) * self.std + self.mean)

    def __call__(self, latents):
        from scipy.signal import resample_poly

        audio = self.full(latents).numpy()
        rate = self.codec.sample_rate
        factor = math.gcd(rate, self.rate)
        return resample_poly(audio, self.rate // factor, rate // factor).astype(np.float32)


def target_frames(reference_frames, reference_text, text, version, mode="auto", scale=1.0, predictor=None):
    """Target length of the deployed duration rules (mirrors `Synthesizer.target_frames`)."""
    reference_text, text = normalize(reference_text, version), normalize(text, version)
    if mode == "auto":
        mode = auto_mode(reference_frames, reference_text)
    if mode == "syllable":
        frames = rule_frames(reference_frames, reference_text, text, "syllables")
    elif mode == "predictor":
        frames = (predictor or DurationPredictor.load()).predict(reference_frames, reference_text, text)
    elif mode in ("rule", "clamp"):
        frames = rule_frames(reference_frames, reference_text, text, "bytes")
        if mode == "clamp":
            frames *= clamp_scale(reference_frames, reference_text)
    else:
        raise ValueError("duration mode must be gt, rule, clamp, syllable, predictor or auto")
    return max(round(frames * scale), 1)


class PromptSource:
    """Cross-utterance prompts of one cache split: another recording of the same speaker is the voice prompt, the
    target text is new and its length follows the deployed duration rule (`gt` keeps the recorded length).

    Partners are chosen here rather than by LatentDataset's cross pairing, which refuses splits with single-recording
    speakers; caches merged with --keep-singletons for within-utterance training keep them (as in monitor.py, only
    speakers with two or more recordings give prompts).
    """

    def __init__(self, cache, split, saved, seed=42, duration_mode="auto", duration_scale=1.0, max_frames=1000,
                 device="cpu"):
        self.data = LatentDataset(cache, split, seed, pairing="within", layout="joined")  # row access only
        check_compatibility(self.data.meta, saved["codec"])
        if not torch.equal(self.data.mean, saved["mean"].cpu()) or not torch.equal(self.data.std, saved["std"].cpu()):
            raise ValueError("Prompt cache and checkpoint use different latent statistics")
        self.indices = np.flatnonzero(self.data.group_end - self.data.group_start >= 2)
        if not len(self.indices):
            raise ValueError(f"The {split} split has no speaker with two recordings to pair a prompt with")
        self.layout = saved["config"]["model"]["text_layout"]
        self.version = saved["codec"].get("text_normalization", "unicode-v1")
        self.mode, self.scale, self.max_frames = duration_mode, duration_scale, max_frames
        self.device = torch.device(device)
        self.predictor = DurationPredictor.load() if duration_mode == "predictor" else None

    def item(self, index, rng):
        """Row `index` as the target and another recording of its speaker as the prompt (LatentDataset's layout)."""
        start, end = int(self.data.group_start[index]), int(self.data.group_end[index])
        partner = rng.randrange(start, end - 1)
        partner += partner >= index
        target, reference = self.data.row(index), self.data.row(partner)
        return {
            "target": target["latents"],
            "reference": reference["latents"],
            "text": target["text"],
            "reference_text": reference["text"],
            "text_bytes": target["text_bytes"],
            "reference_text_bytes": reference["text_bytes"],
            "token_ids": target["token_ids"],
            "reference_token_ids": reference["token_ids"],
            "uid": target["uid"],
            "speaker": target["speaker"],
            "text_normalization": self.data.meta.get("text_normalization", "unicode-v1"),
            "layout": self.layout,
        }

    def prompt(self, index, rng):
        """The prompt of dataset row `index`, or None when it is unusable (too long, empty after normalization)."""
        item = self.item(index, rng)
        try:
            frames = (
                len(item["target"])
                if self.mode == "gt"
                else target_frames(len(item["reference"]), item["reference_text"], item["text"], self.version,
                                   self.mode, self.scale, self.predictor)
            )
        except ValueError:
            return None
        if len(item["reference"]) + frames > self.max_frames:
            return None
        condition = collate([{**item, "target": torch.zeros(frames, item["reference"].size(1))}])
        condition.pop("latents")
        return Prompt(
            {key: value.to(self.device) for key, value in condition.items()},
            item["text"],
            item["reference_text"],
            item["reference"],
            item["uid"],
            item["speaker"],
        )

    def draw(self, rng, attempts=100):
        for _ in range(attempts):
            prompt = self.prompt(int(rng.choice(self.indices)), rng)
            if prompt is not None:
                return prompt
        raise ValueError("No usable prompt in 100 draws; raise --max-frames or check the cache texts")

    def fixed(self, count, seed):
        """A deterministic prompt set (held-out monitor, oracle)."""
        rng = random.Random(seed)
        order = [int(i) for i in self.indices]
        rng.shuffle(order)
        prompts = []
        for index in order:
            if len(prompts) == count:
                break
            prompt = self.prompt(index, rng)
            if prompt is not None:
                prompts.append(prompt)
        return prompts


# --------------------------------------------------------------------------------------------------------------------
# Rewards: judges, terms, group standardization


def parse_spec(spec):
    """'cer=1,sim=0.5,pkg.mod:attr=0.2' -> {'cer': 1.0, 'sim': 0.5, 'pkg.mod:attr': 0.2}."""
    result = {}
    for part in filter(None, (piece.strip() for piece in (spec or "").split(","))):
        name, sep, value = part.rpartition("=")
        if not sep or not name.strip():
            raise ValueError(f"Expected name=value in {spec!r}, got {part!r}")
        result[name.strip()] = float(value)
        if not math.isfinite(result[name.strip()]):
            raise ValueError(f"Nonfinite value in {part!r}")
    return result


def standardize(values, floor=0.0, eps=1e-8):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(float(values.std()), floor, eps)


def composite_floor(weights):
    """Smallest composite spread counted as signal: a term moving by exactly its floor has z-spread 1 and spreads
    R = sum_k w_k z_k by |w_k|, so the lightest active term sets the bar (z-score units, scale-free in the weights)."""
    return min(abs(weight) for weight in weights.values() if weight)


def group_advantages(raw, weights, signs=None, floors=None, mode="standardized", eps=1e-4, stats=None):
    """Per-term z-scores inside the group (sign-aware, spread floored), weighted sum, then the advantage.

    `standardized` divides the centred composite by max(std + eps, composite_floor): judge jitter below every term's
    floor stays small instead of being rescaled to unit variance, and groups with real spread are unchanged. `stats`
    (a dict) receives the composite spread, the floor and whether the group fell under it.
    """
    signs, floors = signs or {}, floors or {}
    active = [name for name, weight in weights.items() if weight]
    if not active:
        raise ValueError("No active reward term")
    if mode not in ("standardized", "weighted"):
        raise ValueError("advantage must be standardized or weighted")
    composite = sum(weights[k] * signs.get(k, 1) * standardize(raw[k], floors.get(k, 0.0)) for k in active)
    spread, floor = float(composite.std()), composite_floor({k: weights[k] for k in active})
    if stats is not None:
        stats.update(composite_std=spread, composite_floor=floor, under_floor=spread + eps < floor)
    if mode == "weighted":
        return composite / sum(abs(weights[k]) for k in active)
    return np.zeros_like(composite) if spread == 0 else (composite - composite.mean()) / max(spread + eps, floor)


class CompositeReward:
    """Weighted, group-standardized reward terms. A term maps a Group to one raw score per sample; `sign = -1` marks
    lower-is-better metrics (error rates). Raw scores are what gets logged."""

    def __init__(self, terms, weights, floors=None, advantage="standardized", min_terms=2, eps=1e-4):
        self.weights = {name: float(weight) for name, weight in weights.items() if weight}
        if len(self.weights) < min_terms:
            raise ValueError(
                f"{len(self.weights)} active reward term(s); at least {min_terms} are required because single-metric "
                "rewards get hacked (arXiv:2609.13150: held-out UTMOS 4.51 -> 1.23); --allow-single-reward overrides"
            )
        missing = set(self.weights) - set(terms)
        if missing:
            raise ValueError(f"Reward terms without an implementation: {sorted(missing)}")
        self.terms = {name: terms[name] for name in self.weights}
        self.signs = {name: getattr(term, "sign", 1) for name, term in self.terms.items()}
        self.floors = {**REWARD_FLOORS, **(floors or {})}
        self.advantage, self.eps = advantage, eps

    def score(self, group):
        raw = {}
        for name, term in self.terms.items():
            values = np.asarray(term(group), dtype=np.float64)
            if values.shape != (len(group),):
                raise ValueError(f"Reward term {name} returned {values.shape}, expected ({len(group)},)")
            bad = ~np.isfinite(values)
            if bad.all():
                raise ValueError(f"Reward term {name} returned no finite score")
            # A failed judgement counts as the group's worst sample instead of stopping the run.
            values[bad] = values[~bad].min() if self.signs[name] > 0 else values[~bad].max()
            raw[name] = values
        return raw

    def advantages(self, raw, stats=None):
        return group_advantages(raw, self.weights, self.signs, self.floors, self.advantage, self.eps, stats)


class Judges:
    """Frozen reward/monitor models, loaded once per (kind, name) and shared (one DNSMOS session serves both)."""

    def __init__(self, device):
        self.device, self.models = torch.device(device), {}

    def get(self, key, factory):
        if key not in self.models:
            self.models[key] = factory()
        return self.models[key]


class WhisperASR:
    """Batched transformers Whisper, greedy decoding; names without '/' map to openai/whisper-<name>."""

    def __init__(self, name, device, language):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        name = name if "/" in name else f"openai/whisper-{name}"
        dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(name)
        self.model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        self.device, self.dtype, self.language, self.name = torch.device(device), dtype, language, name

    @torch.inference_mode()
    def transcribe(self, audios):
        features = self.processor(list(audios), sampling_rate=16000, return_tensors="pt").input_features
        ids = self.model.generate(features.to(self.device, self.dtype), language=self.language, task="transcribe",
                                  num_beams=1, max_new_tokens=220)
        return [text.strip() for text in self.processor.batch_decode(ids, skip_special_tokens=True)]


class ErrorRateTerm:
    """CER or WER against the target text (capped); cer and wer of one ASR share a single transcription."""

    sign = -1

    def __init__(self, asr, kind, normalization, cap=1.0):
        self.asr, self.kind, self.normalization, self.cap = asr, kind, normalization, cap

    def __call__(self, group):
        from .metrics import error_counts

        hypotheses = group.memo(("asr", self.asr.name), lambda: self.asr.transcribe(group.audio()))
        try:
            return [min(error_counts(group.text, h, self.normalization)[self.kind], self.cap) for h in hypotheses]
        except ValueError:  # empty reference after normalization: no signal
            return [0.0] * len(hypotheses)


class SpeakerEmbedder:
    def __init__(self, name, device):
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

        self.extractor = AutoFeatureExtractor.from_pretrained(name)
        self.model = AutoModelForAudioXVector.from_pretrained(name).to(device).eval()
        self.device, self.name = torch.device(device), name

    @torch.inference_mode()
    def __call__(self, audios):
        inputs = self.extractor(list(audios), sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        return nn.functional.normalize(self.model(**inputs).embeddings.float(), dim=-1).cpu()


class SimilarityTerm:
    """Cosine similarity of speaker embeddings between each sample and the codec-decoded voice prompt (SIM-r)."""

    def __init__(self, embedder):
        self.embedder = embedder

    def __call__(self, group):
        reference = group.memo(("speaker", self.embedder.name), lambda: self.embedder([group.reference_audio()]))
        return (self.embedder(group.audio()) @ reference.T).squeeze(-1).tolist()


class DNSMOSTerm:
    def __init__(self, path):
        from .metrics import DNSMOS

        self.model = DNSMOS(path)

    def __call__(self, group):
        audios = group.audio()
        # Single-threaded ONNX sessions: score the clips of a group in parallel threads.
        with ThreadPoolExecutor(max_workers=min(8, len(audios))) as pool:
            return [score["dnsmos_ovrl"] for score in pool.map(self.model, audios)]


class UTMOSTerm:
    """UTMOS22 strong learner through SpeechMOS (torch.hub, downloaded on first use)."""

    def __init__(self, repo, device):
        self.model = torch.hub.load(repo, UTMOS_ENTRY, trust_repo=True).to(device).eval()
        self.device = torch.device(device)

    @torch.inference_mode()
    def __call__(self, group):
        return [float(self.model(torch.from_numpy(a)[None].to(self.device), 16000)) for a in group.audio()]


class DistillMOSTerm:
    """Distill-MOS (Microsoft, `pip install distillmos`), 16 kHz input."""

    def __init__(self, device):
        try:
            import distillmos
        except ImportError as exc:
            raise ImportError("The distillmos term needs `pip install distillmos`") from exc
        self.model = distillmos.ConvTransformerSQAModel().to(device).eval()
        self.device = torch.device(device)

    @torch.inference_mode()
    def __call__(self, group):
        return [float(self.model(torch.from_numpy(a)[None].to(self.device))) for a in group.audio()]


def build_terms(names, judges, asr_model, speaker_model, dnsmos_model=None, utmos_repo="tarepan/SpeechMOS:v1.2.0",
                language="tr", normalization=None):
    """Reward/monitor terms by name; frozen models come from (and stay in) `judges`."""
    normalization = normalization or ("turkish-v1" if language == "tr" else "english-unicode-v2")
    device, terms = judges.device, {}
    for name in names:
        if ":" in name:
            module, attribute = name.split(":", 1)
            terms[name] = getattr(importlib.import_module(module), attribute)
        elif name in ("cer", "wer"):
            asr = judges.get(("asr", asr_model), lambda: WhisperASR(asr_model, device, language))
            terms[name] = ErrorRateTerm(asr, name, normalization)
        elif name == "sim":
            embedder = judges.get(("sv", speaker_model), lambda: SpeakerEmbedder(speaker_model, device))
            terms[name] = SimilarityTerm(embedder)
        elif name == "dnsmos":
            if not dnsmos_model:
                raise ValueError("The dnsmos term needs --dnsmos-model (official sig_bak_ovr.onnx)")
            terms[name] = judges.get(("dnsmos", str(dnsmos_model)), lambda: DNSMOSTerm(dnsmos_model))
        elif name == "utmos":
            terms[name] = judges.get(("utmos", utmos_repo), lambda: UTMOSTerm(utmos_repo, device))
        elif name == "distillmos":
            terms[name] = judges.get(("distillmos",), lambda: DistillMOSTerm(device))
        else:
            raise ValueError(f"Unknown reward term {name!r}; built in: cer, wer, sim, dnsmos, utmos, distillmos")
    return terms


def reward_from_args(args, judges):
    weights = parse_spec(args.reward_weights)
    terms = build_terms([k for k, w in weights.items() if w], judges, args.asr_model, args.speaker_model,
                        args.dnsmos_model, args.utmos_repo, args.language, args.metric_normalization)
    return CompositeReward(terms, weights, parse_spec(args.reward_floors), args.advantage,
                           1 if args.allow_single_reward else 2)


# --------------------------------------------------------------------------------------------------------------------
# Rollouts and the clipped policy-gradient update


@dataclass
class Rollout:
    prompt: Prompt
    latents: torch.Tensor  # [G, T, C] generated target latents (CPU)
    transitions: list
    raw: dict = field(default_factory=dict)
    advantages: torch.Tensor = None


def rollout(model, prompt, size, window, args, generator=None, shared_noise=True):
    """G samples of one prompt with the SDE window; shared initial noise keeps the group on one ODE prefix."""
    condition = repeat_rows(prompt.condition, size)
    shape, rows = condition["prompt"].shape, 1 if shared_noise else size
    noise = torch.randn((rows, *shape[1:]), device=condition["prompt"].device, generator=generator)
    noise = noise.expand(shape).contiguous()
    x, transitions = sde_sample(
        model,
        condition,
        args.sample_steps,
        args.guidance,
        args.sway,
        window,
        args.sde_sigma,
        args.sigma_schedule,
        generator,
        noise,
        args.guidance_from,
        args.guidance_until,
        getattr(args, "logprob_reduction", "mean"),
    )
    return Rollout(prompt, x[:, prompt.target].float().cpu(), transitions)


def transition_mean(model, prompt, rows, transition, args):
    """Mean and std of one recorded window step for the sample rows `rows`, recomputed with `model`."""
    condition = repeat_rows(prompt.condition, len(rows))
    mask = mask_values(condition["valid"], condition["prompt_mask"])
    x, t0 = transition["state"][rows], transition["t0"]
    guided = is_guided(t0, args.guidance, args.guidance_from, args.guidance_until)
    v = policy_velocity(model, x, t0.expand(len(rows)), condition, args.guidance, guided, mask=mask)
    mean, std = sde_step(x, v, t0, transition["t1"], transition["sigma"], mask)
    return mean, std, mask


def grpo_update(policy, reference, optimizer, ema, rollouts, args, device):
    """Recompute old log-densities (and reference means), then ppo_epochs x updates_per_rollout clipped steps."""
    batches = [
        (r, rows)
        for r in rollouts
        if bool((r.advantages != 0).any())
        for rows in torch.arange(len(r.latents), device=device).split(args.micro_batch)
    ]
    stats = {"optimizer_steps": 0, "active_groups": len({id(r) for r, _ in batches})}
    if not batches:
        return stats
    reduction = args.logprob_reduction
    policy.eval()
    old, reference_means, mismatch = {}, {}, []
    with torch.no_grad(), autocast(device, args.precision):
        for b, (r, rows) in enumerate(batches):
            for k, transition in enumerate(r.transitions):
                mean, std, mask = transition_mean(policy, r.prompt, rows, transition, args)
                old[b, k] = transition_log_prob(transition["next"][rows], mean, std, mask, reduction)
                mismatch.append((old[b, k] - transition["log_prob"][rows]).abs().max().item())
                if reference is not None:
                    reference_means[b, k] = transition_mean(reference, r.prompt, rows, transition, args)[0]
    parts = np.array_split(np.arange(len(batches)), min(args.updates_per_rollout, len(batches)))
    chunks = [[int(b) for b in part] for part in parts]
    totals = {"loss": 0.0, "clip_fraction": 0.0, "approx_kl": 0.0, "kl_ref": 0.0, "terms": 0}
    grads = []
    policy.train()
    for _ in range(args.ppo_epochs):
        for chunk in chunks:
            optimizer.zero_grad(set_to_none=True)
            count = sum(len(batches[b][1]) * len(batches[b][0].transitions) for b in chunk)
            for b in chunk:
                r, rows = batches[b]
                advantages = r.advantages[rows]
                for k, transition in enumerate(r.transitions):
                    with autocast(device, args.precision):
                        mean, std, mask = transition_mean(policy, r.prompt, rows, transition, args)
                    log_ratio = transition_log_prob(transition["next"][rows], mean, std, mask, reduction) - old[b, k]
                    loss = ppo_clip_loss(log_ratio, advantages, args.clip)
                    if reference is not None:
                        kl = gaussian_kl(mean, reference_means[b, k], std, mask, reduction)
                        loss = loss + args.kl * kl
                        totals["kl_ref"] += kl.detach().sum().item()
                    (loss.sum() / count).backward()
                    with torch.no_grad():
                        ratio = log_ratio.exp()
                        totals["loss"] += loss.detach().sum().item()
                        totals["clip_fraction"] += ((ratio - 1).abs() > args.clip).float().sum().item()
                        totals["approx_kl"] += ((ratio - 1) - log_ratio).sum().item()  # k3 estimator, >= 0
                        totals["terms"] += len(rows)
            grad = nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            if not torch.isfinite(grad):
                raise FloatingPointError("Nonfinite GRPO gradient")
            optimizer.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()), list(policy.parameters()), 0.01)
            grads.append(grad.item())
            stats["optimizer_steps"] += 1
    policy.eval()
    terms = max(totals.pop("terms"), 1)
    stats.update({key: value / terms for key, value in totals.items()})
    stats.update(grad_norm=float(np.mean(grads)), logp_mismatch=max(mismatch))
    return stats


class Monitor:
    """Held-out check of a model with the deployed ODE sampler and judges that are not in the reward."""

    def __init__(self, prompts, terms, decoder, args, directory=None):
        self.prompts, self.terms, self.decoder, self.args, self.directory = prompts, terms, decoder, args, directory

    def __call__(self, model, step, device):
        values = {name: [] for name in self.terms}
        for i, prompt in enumerate(self.prompts):
            with autocast(device, self.args.precision):
                x = sample(model, **prompt.condition, steps=self.args.monitor_steps, guidance=self.args.guidance,
                           seed=self.args.seed + i, sway=self.args.sway, guidance_until=self.args.guidance_until,
                           guidance_from=self.args.guidance_from)
            latents = x[0, prompt.target].float().cpu()
            group = Group(prompt, [latents], self.decoder)
            for name, term in self.terms.items():
                values[name].append(float(np.asarray(term(group), dtype=np.float64)[0]))
            if self.directory is not None and i < self.args.monitor_audio:
                import soundfile as sf

                path = Path(self.directory) / f"step-{step:06d}" / f"{i:03d}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, self.decoder.full(latents).numpy(), self.decoder.codec.sample_rate, subtype="FLOAT")
        return {name: float(np.mean(v)) for name, v in values.items()}


def validate(args, training=True):
    if not 0 <= args.guidance_from < args.guidance_until <= 1 or not -1 <= args.sway <= 0 or args.guidance < 0:
        raise ValueError("Need 0 <= guidance-from < guidance-until <= 1, sway in [-1,0] and guidance >= 0")
    if args.sde_sigma <= 0 or not 0 < args.window_max <= 1 or args.sde_window > args.sample_steps:
        raise ValueError("Need sde-sigma > 0, window-max in (0,1] and sde-window <= sample-steps")
    choose_window(args.sample_steps, args.sde_window, args.window_max, random.Random(0))
    if training and (args.group_size < 2 or args.clip <= 0 or args.kl < 0 or args.max_grad_norm <= 0):
        raise ValueError("Need group-size >= 2, clip > 0, kl >= 0 and max-grad-norm > 0")


def save_checkpoint(path, saved, policy, ema, step, args):
    result = {key: saved[key] for key in ("config", "codec", "mean", "std")}
    result.update(
        {
            "model": policy.state_dict(),
            "ema": ema.state_dict(),
            "stage": "grpo",
            "step": step,
            "parent": str(Path(args.checkpoint).resolve()),
            "posttrain_args": {key: value for key, value in vars(args).items() if key != "func"},
            "recommended_guidance": args.guidance,
        }
    )
    atomic_save(result, path)


def grpo_train(args, reward=None, monitor=None):
    """`post-train --mode grpo`. `reward`/`monitor` may be injected (tests use latent-space toy rewards)."""
    device, _, world = distributed_device(args.device)
    if world > 1:
        raise ValueError("GRPO post-training runs in one process; put the judges on a second GPU with --reward-device")
    validate(args)
    torch.manual_seed(args.seed)
    policy, saved = load_model(args.checkpoint, device)
    policy.grad_checkpoint = args.grad_checkpoint
    # A checkpoint trained with model.dropout (#14) would drop units in the train-mode gradient passes but not
    # in the eval-mode rollouts and old log-densities, so the on-policy ratio would drift from 1: the policy is
    # optimized without dropout (parameter-free, so its weights and checkpoints are unaffected).
    set_dropout(policy, 0.0)
    reference = copy.deepcopy(policy).requires_grad_(False) if args.kl > 0 else None
    ema = copy.deepcopy(policy).requires_grad_(False)
    source = PromptSource(args.cache, "train", saved, args.seed, args.duration_mode, args.duration_scale,
                          args.max_frames, device)
    reward_device = torch.device(args.reward_device or device)
    decoder = LatentDecoder(saved, reward_device, backend_options(args))
    judges = Judges(reward_device)
    reward = reward or reward_from_args(args, judges)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if monitor is None and args.monitor_every > 0 and args.monitor_prompts > 0:
        prompts = PromptSource(args.cache, "val", saved, args.seed, args.duration_mode, args.duration_scale,
                               args.max_frames, device).fixed(args.monitor_prompts, args.seed)
        names = [name.strip() for name in args.monitor_terms.split(",") if name.strip()]
        terms = build_terms(names, judges, args.monitor_asr_model, args.monitor_speaker_model, args.dnsmos_model,
                            args.utmos_repo, args.language, args.metric_normalization)
        monitor = Monitor(prompts, terms, decoder, args, output / "monitor")
    optimizer = build_optimizer(policy, args.optimizer or "adamw", args.learning_rate, 0.01,
                                saved["config"]["train"].get("muon_momentum", 0.95), fused=device.type == "cuda")
    rng = random.Random(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with open(output / "grpo-log.jsonl", "a") as log:

        def emit(record):
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()

        if monitor is not None:
            emit({"step": 0, "monitor": monitor(ema, 0, device)})
        for step in range(1, args.steps + 1):
            started = time.perf_counter()
            window = choose_window(args.sample_steps, args.sde_window, args.window_max, rng)
            rollouts, floored = [], 0
            policy.eval()
            for _ in range(args.prompts_per_step):
                prompt = source.draw(rng)
                with autocast(device, args.precision):
                    r = rollout(policy, prompt, args.group_size, window, args, generator, args.shared_noise)
                r.raw = reward.score(Group(prompt, r.latents, decoder))
                group_stats = {}
                advantages = reward.advantages(r.raw, group_stats)
                r.advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
                floored += group_stats["under_floor"]
                rollouts.append(r)
            sampled = time.perf_counter()
            stats = grpo_update(policy, reference, optimizer, ema, rollouts, args, device)
            names = rollouts[0].raw
            emit(
                {
                    "step": step,
                    "window": list(window),
                    "reward": {k: float(np.mean([r.raw[k].mean() for r in rollouts])) for k in names},
                    "reward_std": {k: float(np.mean([r.raw[k].std() for r in rollouts])) for k in names},
                    "degenerate_groups": sum(not bool((r.advantages != 0).any()) for r in rollouts),
                    "floored_groups": floored,
                    **stats,
                    "rollout_seconds": sampled - started,
                    "update_seconds": time.perf_counter() - sampled,
                }
            )
            if monitor is not None and step % args.monitor_every == 0:
                emit({"step": step, "monitor": monitor(ema, step, device)})
            if step % args.save_every == 0 or step == args.steps:
                save_checkpoint(output / f"grpo-{step:06d}.pt", saved, policy, ema, step, args)


# --------------------------------------------------------------------------------------------------------------------
# Oracle: best-of-N under the same composite reward


def oracle_best_of_n(args, reward=None):
    """Best-of-N with the GRPO reward: how far reweighting the model's own samples can move each metric.

    `--sampler ode` draws the N candidates with the deployed sampler (different initial noise): the achievable
    best-of-N. `--sampler sde` uses the GRPO rollout policy (window, sigma, shared noise) and reports the group spread
    the policy gradient will work with. Writes oracle.jsonl (every candidate) and summary.json.
    """
    validate(args, training=False)
    device = torch.device(args.device)
    model, saved = load_model(args.checkpoint, device)
    source = PromptSource(args.cache, args.split, saved, args.seed, args.duration_mode, args.duration_scale,
                          args.max_frames, device)
    prompts = source.fixed(args.limit, args.seed)
    reward_device = torch.device(args.reward_device or device)
    decoder = LatentDecoder(saved, reward_device, backend_options(args))
    reward = reward or reward_from_args(args, Judges(reward_device))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    names = list(reward.weights)
    picks = {key: {k: [] for k in names} for key in ("first", "mean", "best", "term_best", "group_std")}
    degenerate = floored = 0
    sde = args.sampler == "sde"
    with open(output / "oracle.jsonl", "x") as stream:
        for i, prompt in enumerate(prompts):
            window = choose_window(args.sample_steps, args.sde_window, args.window_max, rng) if sde else ()
            with autocast(device, args.precision):
                r = rollout(model, prompt, args.candidates, window, args, generator, sde and args.shared_noise)
            group = Group(prompt, r.latents, decoder)
            raw = reward.score(group)
            group_stats = {}
            advantages = reward.advantages(raw, group_stats)
            degenerate += not np.any(advantages)
            floored += group_stats["under_floor"]
            best = int(np.argmax(advantages))
            for k in names:
                values, sign = raw[k], reward.signs[k]
                picks["first"][k].append(values[0])
                picks["mean"][k].append(values.mean())
                picks["best"][k].append(values[best])
                picks["term_best"][k].append(values.max() if sign > 0 else values.min())
                picks["group_std"][k].append(values.std())
            stream.write(json.dumps({"index": i, "uid": prompt.uid, "speaker": prompt.speaker, "text": prompt.text,
                                     "window": list(window), "selected": best, "advantages": advantages.tolist(),
                                     "raw": {k: v.tolist() for k, v in raw.items()}}, ensure_ascii=False) + "\n")
            if i < args.save_audio:
                import soundfile as sf

                rate = decoder.codec.sample_rate
                clips = {"prompt": prompt.reference, "first": r.latents[0], "best": r.latents[best]}
                for name, latents in clips.items():
                    sf.write(output / f"{i:04d}-{name}.wav", decoder.full(latents).numpy(), rate, subtype="FLOAT")
    summary = {
        "prompts": len(prompts),
        "candidates": args.candidates,
        "sampler": args.sampler,
        "weights": reward.weights,
        "degenerate_groups": degenerate,
        "floored_groups": floored,
        **{key: {k: float(np.mean(v)) for k, v in table.items()} for key, table in picks.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary
