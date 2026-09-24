"""Latent negatives for contrastive flow matching that need no extra generator pass.

RobustSpeechFlow (arXiv:2605.22083, a 60M SupertonicTTS) pairs the *correct* transcript with
corrupted target latents and subtracts their flow losses: L_pos - λ_rand L_rand - λ_aug L_aug with
λ = 0.2 each, a subtraction rather than a hinge (Seed-TTS WER 1.44 -> 1.38, SIM unchanged; random
negatives alone 1.41). ΔFM (arXiv:2506.05350) is the same objective, |F - F+|² - λ|F - F-|², where F-
is the regression target of the negative sample under the positive's noise and time. Only the target
changes, so the positive pass's prediction is reused: no second text encoding and no second generator
forward/backward, unlike the transcript hinge (`corrupt_transcript`), which costs ~20-25% per step.

Boundedness: every distance is a per-example mean over the same target frames and is reduced exactly
like the positive flow term, so per frame the objective is (1 - Σλ)|F|² + linear terms, a convex
quadratic in the prediction whenever Σλ < 1. Its minimum, F+ + Σλ(F+ - F-)/(1 - Σλ), lies a finite
step past the true target, away from the negatives, so the plain subtraction cannot run away. The
optional cap (`negative_distance`) additionally stops the push once a prediction is far enough away.

Rows follow the collate layout [prompt | target | padding], but only the target mask is used: target
frames are addressed by their rank among the row's target frames, and prompt and padding frames come
back unchanged, so negative and positive targets differ on target frames only.
"""

from pathlib import Path

import torch

from .contracts import target_mask
from .model import flow_target, per_example_mse


def target_frames(valid, prompt_mask):
    """Target mask [B,L], the frame index of every target frame in order (leading columns of [B,L]),
    each frame's rank among its row's target frames [B,L] and the target length [B]."""
    target = target_mask(valid, prompt_mask)
    order = torch.sort((~target).long(), dim=1, stable=True).indices
    return target, order, target.long().cumsum(1) - 1, target.sum(1)


def take(latents, order, rank):
    """latents[b, order[b, rank[b, j]]] for every row b and column j: [B,J,C]."""
    index = order.gather(1, rank)
    return latents.gather(1, index[..., None].expand(-1, -1, latents.size(-1)))


def fill_rows(latents, order, length, fill=None):
    """Tail padding [B,C]. `fill` [C] or [B,C] (e.g. the cache's standardized silence latent) wins;
    otherwise each row's own last target frame. Trimmed utterances usually end in a pause or breath,
    so that frame is the closest in-row stand-in for silence; a voiced last frame instead holds the
    final sound, which is still a wrong target but a less realistic one than the paper's silence.
    Skip negatives pad 40-80% of the target at the paper's coverage, so a real silence latent matters."""
    if fill is not None:
        return torch.broadcast_to(fill.to(latents), (latents.size(0), latents.size(-1)))
    return take(latents, order, (length - 1).clamp_min(0)[:, None])[:, 0]


MAX_EDITS = 8  # vectorized edit rounds; the coverage budget usually ends the edits after 1-4


def augmented_negatives(
    latents,
    valid,
    prompt_mask,
    span=(3, 125),
    repeat_coverage=(0.2, 0.4),
    skip_coverage=(0.4, 0.8),
    fill=None,
    generator=None,
    edits=MAX_EDITS,
):
    """RobustSpeechFlow's length-preserving failure-mode negatives inside every row's target, and a
    mask [B] of the rows that received one.

    Per row (paper, Sec. 3.3): repeat or skip with probability 1/2, a coverage budget κ·n frames with
    κ ~ U(repeat_coverage) or U(skip_coverage), then edits with span lengths ~ U{span} frames until the
    budget is used. Repeat overwrites a span [k, k+l) with the original content of another span
    [s, s+l), s != k (a repeat that also skips what it covers). Skip removes a span of the remaining
    content, shifts the rest forward and pads the tail with `fill` (the paper's precomputed silence
    latent). Defaults are the paper's: 0.1-5 s spans (3-125 frames at 25 fps), κ 0.2-0.4 / 0.4-0.8.
    Spans are frame ranges because word boundaries are unknown without an alignment.

    Deviation for vectorization: at most `edits` rounds, and a span longer than the remaining budget
    is clipped to it (the paper stops before it), so rows whose budget holds one minimum span always
    get a negative; rows with a budget below span[0] frames come back unchanged and unusable.
    """
    target, order, rank, length = target_frames(valid, prompt_mask)
    rows, frames = rank.shape
    low, high = span
    draws = torch.rand(rows, 2 + 3 * edits, device=latents.device, generator=generator)
    skip = draws[:, 0] < 0.5
    lower = torch.where(skip, skip_coverage[0], repeat_coverage[0])
    upper = torch.where(skip, skip_coverage[1], repeat_coverage[1])
    remaining = ((lower + draws[:, 1] * (upper - lower)) * length).long()
    usable = remaining >= low
    positions = torch.arange(frames, device=latents.device).expand(rows, -1)
    # source[b, r]: the original target rank that negative target frame r plays; -1 plays `fill`.
    source, content, count = positions, length, length[:, None]
    for edit in range(edits):
        u = draws[:, 2 + 3 * edit : 5 + 3 * edit]
        size = (low + (u[:, 0] * (high - low + 1)).long()).minimum(remaining)
        active = size >= low  # budget left for one more minimum span
        size = size.clamp_min(1)
        # Skip: start inside the content that is not tail padding yet, shift the rest forward.
        start = (u[:, 1] * (content - size + 1)).long().minimum(content - size).clamp_min(0)[:, None]
        shifted = source.gather(1, (positions + size[:, None]).clamp(max=frames - 1))
        skipped = torch.where(
            positions < start, source, torch.where(positions < count - size[:, None], shifted, -1)
        )
        # Repeat: overwrite [k, k+l) with original [s, s+l), s drawn from the other n-l start positions.
        slots = (length - size).clamp_min(1)
        k = (u[:, 1] * (slots + 1)).long().minimum(slots)[:, None]
        s = (k[:, 0] + 1 + (u[:, 2] * slots).long().minimum(slots - 1)) % (slots + 1)
        inside = (positions >= k) & (positions < k + size[:, None])
        repeated = torch.where(inside, s[:, None] + positions - k, source)
        source = torch.where(active[:, None], torch.where(skip[:, None], skipped, repeated), source)
        remaining = remaining - torch.where(active, size, 0)
        content = content - torch.where(active & skip, size, 0)
    played = source.gather(1, rank.clamp_min(0))
    picked = take(latents, order, played.clamp(0, frames - 1).minimum(count - 1))
    tail = fill_rows(latents, order, length, fill)
    negative = torch.where((played < 0)[..., None], tail[:, None], picked)
    return torch.where((target & usable[:, None])[..., None], negative, latents), usable


def random_negatives(latents, valid, prompt_mask, copies=1, fill=None):
    """Every row's target replaced by the target of the row `copies` places earlier, and a mask [B].

    Batch expansion repeats each utterance `copies` times in a row, so this shift always reaches
    another utterance (bucketing keeps its length close). The partner is cropped, or padded at the end
    with `fill` (default: the partner's own last target frame), to this row's target length; the
    onsets stay aligned. With a single utterance in the batch nothing is usable.
    """
    target, order, rank, length = target_frames(valid, prompt_mask)
    rows = latents.size(0)
    partner = torch.arange(rows, device=latents.device).roll(copies)
    other = length[partner][:, None]
    r = rank.clamp_min(0)
    picked = take(latents[partner], order[partner], r.minimum(other - 1).clamp_min(0))
    tail = fill_rows(latents[partner], order[partner], length[partner], fill)
    negative = torch.where((r >= other)[..., None], tail[:, None], picked)
    usable = torch.full((rows,), rows > copies, device=latents.device)
    return torch.where((target & usable[:, None])[..., None], negative, latents), usable


def negative_distance(model, prediction, negative_latents, noise, time, mask, positive=None, cap=0.0):
    """Per-example MSE [B] between the prediction and the flow target of `negative_latents` under the
    positive pass's noise and time, in the model's own parameterization (the ΔFM negative term).

    The negative target is detached, so gradients reach the generator only through its prediction.
    cap > 0 clamps the distance at cap times the (detached) distance between the positive target
    `positive` and the negative one: an example stops being pushed once its prediction sits cap times
    farther from the negative than the true target does (cap 1: only while it is closer than that).
    """
    negative = flow_target(model, negative_latents, noise, time).detach()
    distance = per_example_mse(prediction, negative, mask)
    if cap > 0:
        distance = torch.minimum(distance, cap * per_example_mse(positive.detach(), negative, mask))
    return distance


def delta_sums(losses, weights):
    """Log sums [5]: weighted latent_delta, then per distance its weighted sum and covered weight.

    Distances are zero exactly where no negative was applied (CFG-dropped or unusable rows)."""
    weights = weights.float()
    sums = [(losses["latent_delta"].detach().float() * weights).sum()]
    for key in ("negative_random", "negative_aug"):
        value = losses[key].detach().float()
        sums += [(value * weights).sum(), ((value > 0).float() * weights).sum()]
    return torch.stack(sums)


def delta_record(sums, denominator):
    """latent_delta on the flow term's scale (their sum is the optimized flow objective), each mean
    distance over the rows that had that negative (compare with `flow`), and the covered share."""
    return {
        "latent_delta": (sums[0] / denominator).item(),
        "negative_random": (sums[1] / sums[2].clamp_min(1e-12)).item(),
        "negative_aug": (sums[3] / sums[4].clamp_min(1e-12)).item(),
        "negative_random_coverage": (sums[2] / denominator).item(),
        "negative_aug_coverage": (sums[4] / denominator).item(),
    }


def load_silence(directory, channels):
    """The cache's silence latent [C] from `silence.pt`, or None when the cache has none.

    Expected already standardized with the cache statistics (the format the silence-latent issue #11
    specifies): a tensor [C], a tensor [T,C] (averaged over frames) or a dict holding one of them
    under `latent`.
    """
    path = Path(directory) / "silence.pt"
    if not path.exists():
        return None
    value = torch.load(path, map_location="cpu", weights_only=True)
    value = value.get("latent") if isinstance(value, dict) else value
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path}: expected a tensor or a dict with a `latent` tensor")
    value = value.float().mean(0) if value.ndim == 2 else value.float()
    if value.shape != (channels,) or not torch.isfinite(value).all():
        raise ValueError(f"{path}: silence latent must be finite [{channels}] or [T,{channels}]")
    return value
