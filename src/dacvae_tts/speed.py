"""Opt-in training throughput options: length padding, loader-side text negatives and sync-free
finiteness checks.

The generator is small (66.5M) and overhead/memory bound, not FLOP bound: at 0.6 s per update an
RTX 4090 runs at roughly 9% of its bf16 peak. The gains therefore come from fewer host-device
synchronizations, fused pointwise kernels and less recomputation, not from faster matrix products.
Every option is off by default, and the defaults reproduce the previous training numerics exactly.
"""

import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from .data import collate
from .text import corrupt_transcript


class NonfiniteWatch:
    """Sync-free replacement of `if not torch.isfinite(x): raise` (`train.strict_checks: false`).

    Every such `if` makes the host wait for the GPU, once per micro-batch for the loss and once per
    update for the gradient norm. The watch keeps the first update with a nonfinite objective or
    gradient norm on the device and raises at the next point where the host synchronizes anyway (a
    log record, validation or a checkpoint), always before anything is written, so no checkpoint
    ever holds nonfinite weights. The price: a diverged run stops up to `log_every` updates late.
    Under DDP the first offending update is reduced over ranks so that all ranks raise together.
    """

    KINDS = ("objective", "gradient")
    NEVER = torch.iinfo(torch.int64).max

    def __init__(self, device):
        self.first = torch.full((len(self.KINDS),), self.NEVER, dtype=torch.int64, device=device)

    def note(self, kind, value, step):
        slot = self.KINDS.index(kind)
        bad = ~torch.isfinite(value.detach()).all()
        self.first[slot] = torch.where(bad, self.first[slot].clamp(max=step + 1), self.first[slot])

    def check(self):
        first = self.first.clone()
        if dist.is_initialized():
            dist.all_reduce(first, op=dist.ReduceOp.MIN)
        for kind, update in zip(self.KINDS, first.tolist()):
            if update != self.NEVER:
                raise FloatingPointError(f"Nonfinite {kind} at update {update} (deferred check)")


def padded_costs(costs, multiple=1):
    """Row costs rounded up to the padding multiple, so the frame budget counts padded frames.

    The bucket sampler bounds (rows + 1) x cost of the row being added, the longest so far; with the
    padded cost that bound covers the padded length collate will produce.
    """
    if multiple <= 1:
        return costs
    costs = np.asarray(costs)
    return (costs + multiple - 1) // multiple * multiple


FRAME_KEYS = ("latents", "prompt", "prompt_mask", "valid")
TEXT_KEYS = ("tokens", "segments", "negative_tokens", "negative_segments")


def pad_lengths(batch, frame_multiple=1, text_multiple=1):
    """Round the audio length L of a collated batch up to a multiple of `frame_multiple` and the text
    length S (transcripts and negatives) up to one of `text_multiple`.

    The added positions are padding in every mask (valid and prompt_mask false, PAD tokens), the same
    padding collate already gives every row shorter than the longest, so each loss is unchanged. Only
    the number of distinct tensor shapes changes, which bounds recompilations of compiled blocks and
    gives the kernels aligned sizes. The flow noise is drawn for the padded shape, so a padded run
    uses different random numbers than an unpadded one, not a different objective. Negatives drawn
    inside the step (loader_negatives off) keep their natural width; with compiled blocks, draw them
    in the loader so their width is padded too.
    """
    result = dict(batch)
    for keys, multiple in ((FRAME_KEYS, frame_multiple), (TEXT_KEYS, text_multiple)):
        for key in keys:
            value = result.get(key)
            if multiple <= 1 or value is None or not value.size(1) % multiple:
                continue
            extra = value.new_zeros(value.size(0), -value.size(1) % multiple, *value.shape[2:])
            result[key] = torch.cat([value, extra], 1)
    return result


def corrupt_rows(tokens, segments, rngs):
    """`Objective.negatives` with one random generator per row: corrupted transcripts [B,S'] and a
    boolean [B] mask of the rows that could be corrupted (the others keep their true transcript)."""
    rows, usable = [], []
    for row_tokens, row_segments, rng in zip(tokens, segments, rngs):
        corrupted = corrupt_transcript(row_tokens, row_segments, rng)
        usable.append(corrupted is not None)
        rows.append(corrupted if corrupted is not None else (row_tokens, row_segments))
    width = max(len(t) for t, _ in rows)
    padded_tokens = torch.zeros(len(rows), width, dtype=torch.int64)
    padded_segments = torch.zeros(len(rows), width, dtype=torch.int64)
    for i, (t, s) in enumerate(rows):
        padded_tokens[i, : len(t)], padded_segments[i, : len(s)] = t, s
    return padded_tokens, padded_segments, torch.tensor(usable)


class NegativeSeeds(Dataset):
    """Training items plus a `negative_seed`, unique per (seed, epoch, row), for `TrainCollate`.

    The string seed keeps the negative stream independent of the item's own pairing generator
    (seed + epoch * rows + index) and deterministic across workers, ranks and exact resumes.
    """

    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        epoch, row = index if isinstance(index, tuple) else (self.dataset.epoch, index)
        item = self.dataset[index]
        item["negative_seed"] = f"negatives:{self.seed}:{epoch}:{row}"
        return item


class TrainCollate:
    """`collate` plus the loader-side options; picklable for spawned loader workers.

    `negatives`: one skip/repeat text negative per example, drawn here in the worker from the
    item's `negative_seed` (see NegativeSeeds) and carried as `negative_tokens`,
    `negative_segments` and `negative_usable`. `Objective.negatives` then uses them and the step
    no longer copies the transcripts to the CPU. `frame_multiple` / `text_multiple`: pad_lengths.
    """

    def __init__(self, frame_multiple=1, text_multiple=1, negatives=False):
        self.frame_multiple, self.text_multiple, self.negatives = frame_multiple, text_multiple, negatives

    def __call__(self, items):
        batch = collate(items)
        if self.negatives:
            rngs = [random.Random(item["negative_seed"]) for item in items]
            tokens, segments, usable = corrupt_rows(batch["tokens"], batch["segments"], rngs)
            batch.update(negative_tokens=tokens, negative_segments=segments, negative_usable=usable)
        return pad_lengths(batch, self.frame_multiple, self.text_multiple)


def training_loader(dataset, train):
    """(items, collate_fn, sampler costs) of the training loader under the options of `train`.

    With the defaults this is exactly (dataset, collate, dataset.costs).
    """
    negatives = train.loader_negatives and train.contrastive_weight > 0
    costs = padded_costs(dataset.costs, train.pad_multiple)
    if not negatives and train.pad_multiple == 1 and train.text_pad_multiple == 1:
        return dataset, collate, costs
    items = NegativeSeeds(dataset, train.seed) if negatives else dataset
    return items, TrainCollate(train.pad_multiple, train.text_pad_multiple, negatives), costs
