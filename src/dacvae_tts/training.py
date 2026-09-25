"""DDP training with sample-weighted accumulation and per-rank resumable RNG state."""

import copy
import dataclasses
import json
import math
import os
import random
import sqlite3
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .alignment import teacher_layers, teacher_terms
from .codec import check_compatibility
from .config import Config
from .contracts import sanitize, target_mask
from .data import BucketBatchSampler, LatentDataset, collate, move_batch
from .data import load_silence as raw_silence
from .diagnostics import ActivationProbe, gradient_contributions, gradient_groups, loss_buckets
from .model import FlowTTS, condition_inputs, flow_loss, flow_target, reduce_flow
from .negatives import (
    augmented_negatives,
    delta_record,
    delta_sums,
    load_silence,
    negative_distance,
    random_negatives,
)
from .optim import build_optimizer
from .parallel import device_batches, loader_options
from .speed import NonfiniteWatch, compile_blocks, training_epoch_costs, training_loader
from .teacher import teacher_sources
from .text import BYTE_OFFSET, corrupt_transcript
from .tracking import Tracker

TEACHER_LOGS = ("repa", "tla", "tla_entropy")


class Objective(nn.Module):
    """Flow (+ optional duration) loss. Text and reference are encoded once per step.

    `expansion` > 1 is context-sharing batch expansion (SupertonicTTS, arXiv:2503.23108): every
    utterance receives several independent (time, noise) draws that share one condition encoding.
    Flow entries are then [B * expansion] while duration entries stay [B]. `guidance_weight` > 0 trains
    toward the model-guidance target (see `guidance_direction`); evaluation keeps the plain target.

    `contrastive_mode` picks the negatives: `text_hinge` (a one-word skip/repeat transcript, one more
    text encoding and generator pass), `latent_delta` (corrupted target latents with the correct text,
    target-only, see `dacvae_tts.negatives`) or `none`.
    """

    def __init__(
        self,
        model,
        duration_weight=0.1,
        time_sampling="uniform",
        expansion=1,
        ctc_weight=0.0,
        contrastive_weight=0.0,
        contrastive_margin=0.1,
        contrastive_mode="text_hinge",
        random_weight=0.2,
        aug_weight=0.2,
        span=(3, 125),
        repeat_coverage=(0.2, 0.4),
        skip_coverage=(0.4, 0.8),
        negative_cap=0.0,
        silence=None,
        repa_weight=0.0,
        repa_frames="all",
        tla_weight=0.0,
        tla_entropy=0.01,
        guidance_weight=0.0,
    ):
        super().__init__()
        self.model = model
        self.guidance_weight = guidance_weight
        self.duration_weight, self.ctc_weight = duration_weight, ctc_weight
        self.contrastive_weight, self.contrastive_margin = contrastive_weight, contrastive_margin
        self.time_sampling, self.expansion = time_sampling, expansion
        self.rng = random.Random(0)
        if contrastive_mode not in {"text_hinge", "latent_delta", "none"}:
            raise ValueError("contrastive_mode must be text_hinge, latent_delta or none")
        self.contrastive_mode = contrastive_mode
        # latent_delta only (see dacvae_tts.negatives); `silence` is a standardized [C] tail padding.
        self.random_weight, self.aug_weight, self.negative_cap = random_weight, aug_weight, negative_cap
        self.augment = dict(span=tuple(span), repeat_coverage=repeat_coverage, skip_coverage=skip_coverage)
        self.silence = silence
        # Teacher alignment (alignment.py); the train loop clears `repa_active` at repa_stop_step.
        self.repa_weight, self.repa_frames = repa_weight, repa_frames
        self.tla_weight, self.tla_entropy = tla_weight, tla_entropy
        self.repa_active = True

    def negatives(self, batch):
        """Corrupted transcripts [B,S'] plus a mask of the examples that could be corrupted."""
        if "negative_tokens" in batch:  # drawn by the loader workers (train.loader_negatives)
            return batch["negative_tokens"], batch["negative_segments"], batch["negative_usable"]
        tokens, segments = batch["tokens"].cpu(), batch["segments"].cpu()
        starts = batch["target_start"].tolist() if "target_start" in batch else [0] * len(tokens)
        rows, usable = [], []
        for row_tokens, row_segments, start in zip(tokens, segments, starts):
            corrupted = corrupt_transcript(row_tokens, row_segments, self.rng, start)
            usable.append(corrupted is not None)
            rows.append(corrupted if corrupted is not None else (row_tokens, row_segments))
        width = max(len(t) for t, _ in rows)
        padded_tokens = torch.zeros(len(rows), width, dtype=torch.int64)
        padded_segments = torch.zeros(len(rows), width, dtype=torch.int64)
        for i, (t, s) in enumerate(rows):
            padded_tokens[i, : len(t)], padded_segments[i, : len(s)] = t, s
        device = batch["tokens"].device
        return padded_tokens.to(device), padded_segments.to(device), torch.tensor(usable, device=device)

    def latent_delta(self, batch, prediction, details, copies, offset=None):
        """RobustSpeechFlow/ΔFM latent negatives on the (expanded) batch, from this pass's prediction.

        Returns per-row [B * expansion] tensors: the raw distances `negative_random` and `negative_aug`
        (zero where not applied) and `latent_delta` = -λ_rand d_rand - λ_aug d_aug, which the training
        loop adds with the flow term's own weights. Rows dropped for classifier-free guidance learn the
        unconditional field and get no negatives; prompt and padding frames never enter a distance.
        With model guidance (`offset` = w sg(out_cond - out_null), #14) every negative target is shifted by
        the same offset as the positive one, so F+ - F- and the push away from the negatives are exactly
        those without guidance and the optimum is the guided target plus the usual ΔFM step.
        """
        valid, prompt_mask = batch["valid"], batch["prompt_mask"]
        x1, mask = sanitize(batch["latents"], valid), target_mask(valid, prompt_mask)
        noise, time = details["noise"], details["times"]
        positive = flow_target(self.model, x1, noise, time) if self.negative_cap > 0 else None
        if positive is not None and offset is not None:
            positive = positive + offset
        terms = {"latent_delta": torch.zeros(x1.size(0), device=x1.device)}
        for name, weight in (("negative_random", self.random_weight), ("negative_aug", self.aug_weight)):
            terms[name] = torch.zeros_like(terms["latent_delta"])
            if weight == 0:
                continue
            negative, usable = (
                random_negatives(x1, valid, prompt_mask, copies, self.silence)
                if name == "negative_random"
                else augmented_negatives(x1, valid, prompt_mask, **self.augment, fill=self.silence)
            )
            distance = negative_distance(
                self.model, prediction, negative, noise, time, mask, positive, self.negative_cap, offset
            )
            terms[name] = distance.masked_fill(details["drop"] | ~usable, 0)
            terms["latent_delta"] = terms["latent_delta"] - weight * terms[name]
        return terms

    def forward(self, batch):
        # Optional voice conditions (speaker embedding, speaker context), encoded once like the text.
        voice_inputs = condition_inputs(batch)
        cached = self.model.conditions(
            batch["prompt"], batch["prompt_mask"], batch["tokens"], batch["segments"], **voice_inputs
        )
        expanded, shared = batch, cached
        if self.expansion > 1 and self.training:
            # The context clips are only read through `cached`: not repeated for the flow draws.
            expanded = {key: value.repeat_interleave(self.expansion, 0) for key, value in batch.items()
                        if key not in ("context", "context_mask")}
            shared = tuple(value.repeat_interleave(self.expansion, 0) for value in cached)
        repa = self.training and self.repa_weight > 0 and self.repa_active
        tla = self.training and self.tla_weight > 0
        layers = teacher_layers(self.model, repa, tla)
        details = flow_loss(
            self.model,
            expanded,
            self.model.cfg.cond_dropout if self.training else 0,
            return_details=True,
            cached=shared,
            time_sampling=self.time_sampling,
            **({"hidden_layers": layers} if layers else {}),
            guidance_weight=self.guidance_weight if self.training else 0.0,
        )
        prediction = details.pop("prediction")
        offset = details.pop("guidance_offset", None)
        if self.training:
            hidden = details.pop("hidden", {})
            times, drop = details["times"], details["drop"]
            terms = teacher_terms(self.model, expanded, hidden, times, drop, repa, tla, self.repa_frames)
            details.update(terms)
        flow = details["flow"]
        if self.model.duration is None:
            duration_loss = flow.new_zeros(batch["latents"].size(0))
        else:
            frames = (batch["valid"] & ~batch["prompt_mask"]).sum(1)
            characters = ((batch["segments"] == 1) & (batch["tokens"] >= BYTE_OFFSET)).sum(1)
            log_rate = (frames / characters.clamp_min(1)).log()
            duration = self.model.predict_duration(
                batch["prompt"], batch["prompt_mask"], batch["tokens"], batch["segments"], cached=cached
            )
            duration_loss = F.smooth_l1_loss(duration.float(), log_rate, reduction="none")
        copies = flow.numel() // duration_loss.numel()
        if self.contrastive_mode == "text_hinge" and self.contrastive_weight > 0 and self.training:
            # Same audio, noise and time as each utterance's first draw, wrong transcript by one word:
            # the true transcript must explain the audio better by a margin.
            tokens, segments, usable = self.negatives(batch)
            wrong = {**batch, "tokens": tokens, "segments": segments}
            negative = flow_loss(
                self.model,
                wrong,
                0,
                details["times"][::copies],
                details["noise"][::copies],
                cached=self.model.conditions(batch["prompt"], batch["prompt_mask"], tokens, segments, **voice_inputs),
            )
            positive = flow[::copies]
            hinge = F.relu(positive + self.contrastive_margin * positive.detach() - negative)
            hinge = hinge.masked_fill(details["drop"][::copies] | ~usable, 0)
            details["contrastive"] = hinge.repeat_interleave(copies) / copies
        elif self.contrastive_mode == "latent_delta" and self.training:
            # Correct text, corrupted target latents: only the regression target changes (no forward).
            details.update(self.latent_delta(expanded, prediction, details, copies, offset))
        total = flow + self.duration_weight * duration_loss.repeat_interleave(copies)
        return {"loss": total, "duration": duration_loss, **details}

    def auxiliary(self, losses):
        """Weighted auxiliary terms [B * expansion] that join the flow term in the optimized loss."""
        terms = []
        if "ctc" in losses:
            terms.append(self.ctc_weight * losses["ctc"])
        if "contrastive" in losses:
            terms.append(self.contrastive_weight * losses["contrastive"])
        if "repa" in losses:
            terms.append(self.repa_weight * losses["repa"])
        if "tla" in losses:
            terms.append(self.tla_weight * (losses["tla"] + self.tla_entropy * losses["tla_entropy"]))
        if "teacher_idle" in losses:
            terms.append(losses["teacher_idle"])
        return sum(terms) if terms else None

    def teacher_sums(self, losses):
        """Summed [repa, tla, tla_entropy] terms of one micro-batch, for logging."""
        zero = losses["flow"].new_zeros(())
        return torch.stack([losses[k].detach().sum() if k in losses else zero for k in TEACHER_LOGS])

    def teacher_record(self, means):
        """Logged means of the enabled teacher terms; no keys at all when both are off."""
        enabled = (self.repa_weight > 0, self.tla_weight > 0, self.tla_weight > 0)
        return {key: float(value) for key, value, on in zip(TEACHER_LOGS, means, enabled) if on}


def distributed_device(requested="auto"):
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank, local = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available() and requested != "cpu"
    device = torch.device(f"cuda:{local}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")
    if world > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo")
    return device, rank, world


def autocast(device, precision):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16")


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def ema_key(decay):
    """Checkpoint key of an extra EMA track, e.g. ema_0.999 (shortest round-trip float spelling)."""
    return f"ema_{float(decay)!r}"


def ema_tracks(train):
    """Extra EMA tracks {checkpoint key: decay}; `ema_decay` itself always stays under `ema`."""
    return {ema_key(decay): decay for decay in train.ema_decays if decay != train.ema_decay}


def ema_rate(decay, step, warmup=True):
    """Decay of an EMA track at 0-based update `step`. The warm-up keeps the average from being dominated
    by the random initialization; without it (train.ema_warmup: false, warm starts only) `decay` applies."""
    return min(decay, (1 + step) / (10 + step)) if warmup else decay


def weights_key(checkpoint, ema=True):
    """Checkpoint entry for `ema`: True -> "ema", False -> "model", a decay (0.999 or "0.999") or a key
    ("ema_0.999") -> that EMA track; the primary decay maps to "ema"."""
    if ema is True or ema is False:
        return "ema" if ema else "model"
    key = ema if str(ema).startswith("ema") else ema_key(float(ema))
    if key == ema_key(Config.from_dict(checkpoint["config"]).train.ema_decay):
        key = "ema"
    if key not in checkpoint:
        tracks = sorted(k for k in checkpoint if k == "ema" or k.startswith("ema_"))
        raise KeyError(f"Checkpoint has no {key}; available weights: model, {', '.join(tracks)}")
    return key


def load_model(path, device="cpu", ema=True):
    """`ema`: True loads the primary EMA, False the raw weights, a decay or key one of `train.ema_decays`."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = Config.from_dict(checkpoint["config"])
    model = FlowTTS(cfg.model)
    model.load_state_dict(checkpoint[weights_key(checkpoint, ema)])
    return model.to(device).eval(), checkpoint


def export_ema(path, ema, output):
    """Optimizer-free copy of a checkpoint whose `ema` holds another EMA track, for the evaluation tools
    that load the default EMA (monitor, eval scripts, demo)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    key = weights_key(checkpoint, ema)
    dropped = {"optimizer", "rng"}
    result = {k: v for k, v in checkpoint.items() if k not in dropped and not k.startswith("ema_")}
    result.update(ema=checkpoint[key], exported_ema=key)
    atomic_save(result, output)


def decay_start(steps, decay_fraction):
    """First update of the WSD decay phase."""
    return steps - round(steps * decay_fraction)


def lr_multiplier(
    step, warmup, steps, schedule="cosine", decay_fraction=0.2, decay_shape="1-sqrt", floor=0.1
):
    """Linear warmup, then cosine to `floor`, or WSD: constant until `decay_start`, then a linear or 1-sqrt
    decay to `floor` (Hägele et al., arXiv:2405.18392). The defaults are the original cosine schedule."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    if schedule == "wsd":
        start = decay_start(steps, decay_fraction)
        progress = min(max(step - start, 0) / max(steps - start, 1), 1)
        return floor + (1 - floor) * (1 - (math.sqrt(progress) if decay_shape == "1-sqrt" else progress))
    progress = (step - warmup) / max(steps - warmup, 1)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def schedule_multiplier(step, train):
    return lr_multiplier(
        step,
        train.warmup,
        train.steps,
        train.lr_schedule,
        train.decay_fraction,
        train.decay_shape,
        train.min_lr_ratio,
    )


def time_sampling_at(step, train):
    """Flow-time distribution of an update: `final_time_sampling` from its start on (a function of the step
    only, so exact resume is unaffected)."""
    if train.final_time_sampling is None:
        return train.time_sampling
    start = train.final_time_sampling_start
    first = decay_start(train.steps, train.decay_fraction) if start == "decay" else round(start * train.steps)
    return train.final_time_sampling if step >= first else train.time_sampling


def training_dataset(cache, cfg):
    """The training split of `cache` with every data option of `cfg`: the pairing, the training-pair options
    (#11) and the teacher stores (#10, relative store paths resolve against `cache`)."""
    data = LatentDataset(
        cache,
        "train",
        cfg.train.seed,
        prompt_dropout=cfg.train.prompt_dropout,
        pairing=cfg.train.pairing,
        layout=cfg.model.text_layout,
        text_units=cfg.model.text_units,
        prompt_fraction=(cfg.train.prompt_fraction_min, cfg.train.prompt_fraction_max),
        **pair_options(cfg),
        **teacher_sources(cfg.train, cache),
        **tempo_sources(cfg.train, cache),
        **speaker_sources(cfg, cache),
    )
    if cfg.model.speaker_condition_dim and data.speaker_store.dim != cfg.model.speaker_condition_dim:
        raise ValueError(f"model.speaker_condition_dim={cfg.model.speaker_condition_dim}, the speaker store has "
                         f"{data.speaker_store.dim}-d embeddings")
    return data


def training_batches(data, cfg, rank, world, frame_budget, device):
    """Sampler and loader of a training dataset: the throughput options of #7 (length padding, loader-side
    negatives, padded sampler costs) and #11's exact per-epoch cross-prompt costs. With the defaults this is
    the plain bucket sampler over `data.costs` and `collate`."""
    items, train_collate, costs = training_loader(data, cfg.train)
    sampler = BucketBatchSampler(
        costs,
        cfg.train.batch_size,
        rank,
        world,
        cfg.train.seed,
        frame_budget,
        speaker_counts=data.group_end - data.group_start,
        speaker_balance=cfg.train.speaker_balance,
        # Exact per-epoch lengths of cross prompts, rounded up like `costs` when pad_multiple > 1.
        epoch_costs=training_epoch_costs(data, cfg.train),
    )
    loader = DataLoader(
        items,
        batch_sampler=sampler,
        collate_fn=train_collate,
        **loader_options(
            cfg.train.workers,
            device,
            cfg.train.prefetch_factor,
            cfg.train.worker_threads,
            cfg.train.loader_start_method,
        ),
        generator=torch.Generator().manual_seed(cfg.train.seed + rank),
    )
    return sampler, loader


def check_held_out(decay_index, main_index):
    """No training row of the decay cache may be a val/test row or speaker of the main cache.

    Each cache is split on its own when merged: a main cache split by voice clusters (--split-map) or split
    keys and a decay cache merged with plain label hashing or another seed would otherwise train on the
    main cache's held-out voices during the decay, and validation (on the main cache) would score them.
    """
    uids, speakers = set(), set()
    with sqlite3.connect(main_index) as db:
        for uid, speaker in db.execute("SELECT uid, speaker FROM samples WHERE split != 'train'"):
            uids.add(uid)
            speakers.add(speaker)
    with sqlite3.connect(decay_index) as db:
        for uid, speaker in db.execute("SELECT uid, speaker FROM samples WHERE split = 'train'"):
            if uid in uids or speaker in speakers:
                raise ValueError(
                    f"Decay-cache training row {uid} (speaker {speaker}) is held out (val/test) in the main "
                    "cache; merge both caches with the same --split-map/--split-key and seed"
                )


def decay_phase_loader(cache, reference, cfg, rank, world, frame_budget, device):
    """Sampler and loader over the WSD decay cache, built exactly like the main training loader (the same
    pair, teacher and throughput options; see `training_dataset` and `training_batches`).

    Latents are normalized with the main cache's statistics, so the latent space does not shift at the
    switch and the checkpoint's mean/std stay valid for inference; the codec metadata and the text
    normalization must match, and no decay-cache training row may be held out in the main cache. The
    encoded-silence frame of tail silence / quiet cuts (#11) is re-standardized with the same statistics.
    Teacher stores given as relative paths are looked up in the decay cache, which needs its own
    extraction (or absolute store paths covering its rows).
    """
    data = training_dataset(cache, cfg)
    if not data.meta.get("merged"):
        raise ValueError("Run merge on the decay cache before training")
    check_compatibility(data.meta, reference.meta)
    normalization = [meta.get("text_normalization", "unicode-v1") for meta in (data.meta, reference.meta)]
    if normalization[0] != normalization[1]:
        raise ValueError("The decay cache has a different text_normalization than the main cache")
    check_held_out(data.db_path, reference.db_path)
    if data.channels != reference.channels:
        raise ValueError("The decay cache has a different latent width")
    data.mean, data.std = reference.mean, reference.std
    if data.silence is not None:
        data.silence = (raw_silence(cache, data.meta) - data.mean) / data.std
    return training_batches(data, cfg, rank, world, frame_budget, device)


def rng_state(device, negatives=None):
    """This rank's generator states; `negatives` is the in-step text_hinge generator (Objective.rng)."""
    state = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }
    if negatives is not None:
        state["negatives"] = negatives.getstate()
    return state


def restore_rng(state, device, negatives=None):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if state["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)
    if negatives is not None and "negatives" in state:  # older checkpoints: the stream restarts, as before
        negatives.setstate(state["negatives"])


def shuffled_conditions(batch):
    """The same batch with every example's transcript taken from another example.

    The flow loss under wrong text minus the loss under the right text measures how much the
    model relies on the text, a text-alignment signal that needs no ASR. Zero means the text is
    ignored, which is what an untrained or babbling model does.
    """
    return {**batch, "tokens": batch["tokens"].roll(1, 0), "segments": batch["segments"].roll(1, 0)}


def validate(model, loader, device, precision, max_batches=20, reduction="utterance", duration_weight=0.1):
    objective = Objective(model, duration_weight).eval()
    totals = torch.zeros(5, device=device)
    devices = [device.index] if device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = move_batch(batch, device)
            weights = None
            for column, variant in ((0, batch), (4, shuffled_conditions(batch))):
                torch.manual_seed(12345 + i)  # identical noise and flow times for both variants
                with autocast(device, precision):
                    losses = objective(variant)
                if weights is None:
                    weights = losses["frames"] if reduction == "frame" else torch.ones_like(losses["flow"])
                totals[column] += (losses["flow"] * weights).sum()
            totals[1] += weights.sum()
            totals[2] += losses["duration"].sum()
            totals[3] += losses["duration"].numel()
    if dist.is_initialized():
        dist.all_reduce(totals)
    flow = totals[0] / totals[1].clamp_min(1)
    return {
        "validation_loss": (flow + duration_weight * totals[2] / totals[3].clamp_min(1)).item(),
        "validation_flow": flow.item(),
        "validation_text_gain": (totals[4] / totals[1].clamp_min(1) - flow).item(),
    }


def pair_options(cfg):
    """Training-pair options (issue #11) of the training set. Validation keeps the baseline pairs, so
    validation_flow stays comparable between the A/B arms of these options."""
    names = (
        "cross_prompt_prob",
        "cross_prompt_max_utterances",
        "cross_prompt_max_seconds",
        "long_prompt_prob",
        "prompt_fraction_long_max",
        "tail_silence_prob",
        "tail_silence_max_seconds",
        "prompt_cut",
    )
    options = {**{name: getattr(cfg.train, name) for name in names}, "ctc_targets": cfg.model.ctc_targets}
    if cfg.train.speaker_context_prob:  # off: no context keywords, as for tempo below
        options.update({name: getattr(cfg.train, name) for name in (
            "speaker_context_prob", "speaker_context_min_seconds", "speaker_context_max_seconds",
            "speaker_context_max_utterances")})
    if cfg.train.tempo_prompt_prob:  # off: no tempo keywords at all, the dataset is built exactly as before
        options.update(tempo_prompt_prob=cfg.train.tempo_prompt_prob,
                       tempo_prompt_factors=cfg.train.tempo_prompt_factors,
                       tempo_prompt_pairs=cfg.train.tempo_prompt_pairs)
    return options


CONTEXT_FIELDS = ("speaker_context", "speaker_context_width", "speaker_context_layers", "speaker_context_heads",
                  "speaker_context_patch")


def warm_start_config(warm, target):
    """The warm-start checkpoint's model config as it may differ from `target`: dropout, and the speaker context
    fields when the checkpoint has none (the zero-init context branch starts as that model)."""
    changes = {"dropout": target.dropout}
    if warm.speaker_context == "none":
        changes.update({name: getattr(target, name) for name in CONTEXT_FIELDS})
    return dataclasses.replace(warm, **changes)


def warm_start(module, state):
    """load_state_dict for --init-from: strict, except that a new speaker context branch may be missing."""
    missing, unexpected = module.load_state_dict(state, strict=False)
    stray = [key for key in missing if not key.startswith("speaker_context.")]
    if unexpected or stray:
        raise ValueError(f"--init-from weights do not match the model: missing {stray}, unexpected {unexpected}")


def speaker_sources(cfg, cache):
    """LatentDataset keywords of the voice conditions read from stores: the speaker embedding
    (model.speaker_condition_dim) and the recording quality (model.quality_condition); empty if both are off."""
    from .teacher import resolve_store

    sources = {}
    if cfg.model.speaker_condition_dim:
        sources.update(speaker_condition=resolve_store(cfg.train.speaker_condition, cache),
                       speaker_condition_source=cfg.train.speaker_condition_source,
                       speaker_condition_min_cosine=cfg.train.speaker_condition_min_cosine)
    if cfg.model.quality_condition:
        sources.update(quality_scores=resolve_store(cfg.train.quality_scores, cache),
                       quality_dropout=cfg.train.quality_dropout)
    return sources


def speaker_metadata(data):
    """Checkpoint record of the speaker store's embedder: inference must embed prompts with the same model."""
    if getattr(data, "speaker_store", None) is None:
        return {}
    meta = data.speaker_store.meta
    return {"speaker_condition": {key: meta.get(key) for key in ("embedder", "model", "dim", "audio_source",
                                                                  "audio_sources", "splits")}}


def tempo_sources(train, cache):
    """LatentDataset keyword for the tempo-variant store (relative paths resolve against `cache`); empty if off."""
    if not train.tempo_prompt_prob:
        return {}
    from .teacher import resolve_store

    return {"tempo_variants": resolve_store(train.tempo_variants, cache)}


def train(args):
    device, rank, world = distributed_device(args.device)
    try:
        if (Path(args.output) / "last.pt").exists() and not args.resume:
            raise ValueError("Output already has a checkpoint; pass --resume or choose a new run directory")
        cfg = Config.load(args.config)
        for key in (
            "steps",
            "batch_size",
            "accumulation",
            "workers",
            "precision",
            "learning_rate",
            "optimizer",
            "worker_threads",
            "prefetch_factor",
            "loader_start_method",
            "cuda_prefetch",
            "wandb_project",
        ):
            value = getattr(args, key, None)
            if value is not None:
                setattr(cfg.train, key, value)
        if args.compile:
            cfg.train.compile = True if args.compile == "objective" else args.compile
        cfg.train.__post_init__()
        if not cfg.train.ema_warmup and not args.resume and not getattr(args, "init_from", None):
            # A resumed run was checked when it started (resume requires the same configuration).
            raise ValueError("ema_warmup: false needs --init-from; from scratch the random weights dominate")
        if device.type == "cuda" and cfg.train.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support BF16; pass --precision fp32")
        torch.manual_seed(cfg.train.seed)
        random.seed(cfg.train.seed)
        pairing = dict(
            pairing=cfg.train.pairing,
            layout=cfg.model.text_layout,
            text_units=cfg.model.text_units,
            prompt_fraction=(cfg.train.prompt_fraction_min, cfg.train.prompt_fraction_max),
        )
        data = training_dataset(args.cache, cfg)
        if not data.meta.get("merged"):
            raise ValueError("Run merge on all prepared partitions before training")
        if cfg.model.latent_dim != data.channels:
            raise ValueError(f"Config latent_dim={cfg.model.latent_dim}, codec cache has {data.channels}")
        # Defaults: (data, collate, data.costs). Padding/loader negatives change collate and costs.
        sampler, loader = training_batches(data, cfg, rank, world, args.frame_budget, device)
        decay_loader = None
        if cfg.train.decay_cache:
            decay_loader = decay_phase_loader(
                cfg.train.decay_cache, data, cfg, rank, world, args.frame_budget, device
            )
        validation = None
        if not args.no_validation:
            val_data = LatentDataset(args.cache, "val", cfg.train.seed, **pairing, **speaker_sources(cfg, args.cache))
            val_sampler = BucketBatchSampler(val_data.costs, cfg.train.batch_size, rank, world, 12345)
            validation = DataLoader(
                val_data,
                batch_sampler=val_sampler,
                collate_fn=collate,
                num_workers=0,
                generator=torch.Generator().manual_seed(0),
            )
        model = FlowTTS(cfg.model).to(device)
        model.grad_checkpoint = cfg.train.grad_checkpoint
        model.strict_checks = cfg.train.strict_checks
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        averages = {key: (decay, copy.deepcopy(ema)) for key, decay in ema_tracks(cfg.train).items()}
        optimizer = build_optimizer(
            model,
            cfg.train.optimizer,
            cfg.train.learning_rate,
            cfg.train.weight_decay,
            cfg.train.muon_momentum,
            fused=device.type == "cuda",
        )
        start_step, epoch, batch_offset, resumed_rng = 0, 0, 0, None
        if args.resume:
            saved = torch.load(args.resume, map_location="cpu", weights_only=True)
            # Checkpoints written before the optimizer became configurable were trained with AdamW.
            saved["config"]["train"].setdefault("optimizer", "adamw")
            if Config.from_dict(saved["config"]).to_dict() != cfg.to_dict():
                raise ValueError(
                    "Resume requires the original configuration; use finetuning for changed schedules"
                )
            if saved["world_size"] != world:
                raise ValueError("Exact resume requires the same GPU count")
            if (
                saved["cache_path"] != str(Path(args.cache).resolve())
                or saved["frame_budget"] != args.frame_budget
            ):
                raise ValueError("Resume requires the same cache path and frame budget")
            model.load_state_dict(saved["model"])
            ema.load_state_dict(saved["ema"])
            for key, (_, average) in averages.items():
                average.load_state_dict(saved[key])
            optimizer.load_state_dict(saved["optimizer"])
            start_step, epoch, batch_offset = saved["step"], saved["epoch"], saved["batch_offset"]
            resumed_rng = saved["rng"][rank]  # restored once the objective (and its generator) exists
        else:
            if getattr(args, "init_from", None):
                # Warm start: weights only. Schedule, optimizer state and data order start fresh, so
                # a run can continue on a larger cache than the one that produced the checkpoint.
                warm = torch.load(args.init_from, map_location="cpu", weights_only=True)
                warm_model = Config.from_dict(warm["config"]).model
                # Dropout has no parameters, so a run may switch it on or off when warm starting; a zero-init speaker
                # context may be added to a checkpoint without one (it starts as that checkpoint's model).
                if warm_start_config(warm_model, cfg.model) != cfg.model:
                    raise ValueError("--init-from requires an identical model configuration (except dropout and an "
                                     "added speaker context)")
                warm_start(model, warm["model"])
                warm_start(ema, warm["ema"])
                for key, (_, average) in averages.items():
                    warm_start(average, warm.get(key, warm["ema"]))
            torch.manual_seed(cfg.train.seed + rank)
            random.seed(cfg.train.seed + rank)
        silence = None
        if cfg.train.contrastive_mode == "latent_delta":
            # Tail padding of skip and short random negatives: the cache's silence latent if it has one.
            silence = load_silence(args.cache, data.channels, data.mean, data.std, data.meta)
            silence = None if silence is None else silence.to(device)
            if rank == 0:
                fill = "last target frame (no silence.pt)" if silence is None else "silence.pt"
                print(json.dumps({"latent_negative_fill": fill}), flush=True)
        objective = Objective(
            model,
            cfg.train.duration_weight,
            cfg.train.time_sampling,
            cfg.train.batch_expansion,
            cfg.train.ctc_weight,
            cfg.train.contrastive_weight,
            cfg.train.contrastive_margin,
            contrastive_mode=cfg.train.contrastive_mode,
            random_weight=cfg.train.contrastive_random_weight,
            aug_weight=cfg.train.contrastive_aug_weight,
            span=(cfg.train.contrastive_span_min, cfg.train.contrastive_span_max),
            repeat_coverage=cfg.train.contrastive_repeat_coverage,
            skip_coverage=cfg.train.contrastive_skip_coverage,
            negative_cap=cfg.train.contrastive_negative_cap,
            silence=silence,
            repa_weight=cfg.train.repa_weight,
            repa_frames=cfg.train.repa_frames,
            tla_weight=cfg.train.tla_weight,
            tla_entropy=cfg.train.tla_entropy,
            guidance_weight=cfg.train.model_guidance_weight,
        ).train()
        if resumed_rng is not None:
            restore_rng(resumed_rng, device, objective.rng)
        raw_objective = objective  # compile/DDP wrap the module; helper methods stay reachable here
        eager_forward = model.forward
        eager_runner = model.block_runner
        compiled = bool(cfg.train.compile)
        if cfg.train.compile == "model":
            # Only the generator: the loss, CTC and batch expansion stay eager, which avoids
            # dynamic-shape failures in the compiler while keeping most of the speed-up.
            model.forward = torch.compile(model.forward, dynamic=True)
        elif cfg.train.compile == "blocks":
            compile_blocks(model, cfg.train.compile_dynamic)  # regional: one compiled block step
        elif cfg.train.compile:
            objective = torch.compile(objective, dynamic=True)

        def compute(module, batch, step):
            """Run the objective; a compiler failure on some rare shape falls back to eager for good."""
            nonlocal objective
            nonlocal compiled
            try:
                return module(batch)
            except Exception as error:
                origin = type(error).__module__
                # Under DDP only the generator/block modes can be swapped (the objective is wrapped).
                swappable = cfg.train.compile in ("model", "blocks") or world == 1
                if (
                    not compiled
                    or not swappable
                    or not origin.startswith(("torch._dynamo", "torch._inductor"))
                ):
                    raise
                model.forward = eager_forward
                model.block_runner = eager_runner
                if world == 1:
                    objective = module = raw_objective
                # else: keep the DDP wrapper (it holds only raw_objective here) and retry through it, so
                # this rank still joins the gradient all-reduce and `no_sync` stays available.
                compiled = False
                # Eager activations need roughly twice the memory of the compiled graph; recompute
                # them instead so a run sized for the compiled path survives the switch.
                model.grad_checkpoint = True
                print(
                    json.dumps(
                        {
                            "step": step + 1,
                            "rank": rank,
                            "warning": "compiler failed; continuing eager with activation checkpointing",
                            "error": str(error)[:300],
                        }
                    ),
                    flush=True,
                )
                return module(batch)

        if world > 1:
            objective = DDP(
                objective,
                device_ids=[device.index] if device.type == "cuda" else None,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        out = Path(args.output)
        tracker = Tracker(
            bool(cfg.train.wandb_project) and rank == 0,
            cfg.train.wandb_project,
            out.name,
            {
                **cfg.to_dict(),
                "frame_budget": args.frame_budget,
                "cache": str(Path(args.cache).resolve()),
                "init_from": getattr(args, "init_from", None),
                "world_size": world,
                "parameters": sum(p.numel() for p in model.parameters()),
            },
            group=getattr(args, "wandb_group", None),
            resume_id=getattr(args, "wandb_id", None),
        )
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
            print(
                json.dumps(
                    {
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "optimizer": cfg.train.optimizer,
                        "ranks": world,
                        "maximum_global_batch": cfg.train.batch_size * cfg.train.accumulation * world,
                        "train_rows": len(data),
                    }
                ),
                flush=True,
            )
        switch = decay_start(cfg.train.steps, cfg.train.decay_fraction) if decay_loader else None
        if decay_loader is not None and start_step > switch:
            # Resumed inside the decay phase: the saved epoch and offset count decay-cache batches.
            sampler, loader = decay_loader
        sampler.epoch, sampler.start_batch = epoch, batch_offset
        # strict_checks: false defers the loss/gradient finiteness checks to the next host sync.
        watch = None if cfg.train.strict_checks else NonfiniteWatch(device)
        iterator = iter(loader)
        last_time = time.monotonic()
        inactive = {}
        for step in range(start_step, cfg.train.steps):
            if decay_loader is not None and step == switch:
                sampler, loader = decay_loader
                epoch, batch_offset = 0, 0
                sampler.epoch, sampler.start_batch = 0, 0
                iterator = iter(loader)
            raw_objective.time_sampling = time_sampling_at(step, cfg.train)
            # Collect a window before backward so variable batches receive exact example weighting.
            batches = []
            for _ in range(cfg.train.accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    batch_offset = 0
                    sampler.epoch, sampler.start_batch = epoch, 0
                    iterator = iter(loader)
                    batch = next(iterator)
                batch_offset += 1
                batches.append(batch)
            counts = torch.tensor(
                [
                    sum(b["latents"].size(0) for b in batches),
                    sum(int((b["valid"] & ~b["prompt_mask"]).sum()) for b in batches),
                ],
                device=device,
            )
            if world > 1:
                dist.all_reduce(counts)
            denominator, frame_denominator = counts.unbind()
            # Batch expansion multiplies the flow terms only; duration terms stay per utterance.
            flow_examples = denominator * cfg.train.batch_expansion
            frame_denominator = frame_denominator * cfg.train.batch_expansion
            learning_rate = cfg.train.learning_rate * schedule_multiplier(step, cfg.train)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            metrics = torch.zeros(5, device=device)
            negative_metrics = torch.zeros(5, device=device)  # latent_delta only, see delta_sums
            teacher_metrics = torch.zeros(len(TEACHER_LOGS), device=device)
            raw_objective.repa_active = not cfg.train.repa_stop_step or step < cfg.train.repa_stop_step
            buckets = torch.zeros(9, 2, device=device)
            diagnostics = {}
            diagnose = cfg.train.diagnostics_every > 0 and step % cfg.train.diagnostics_every == 0
            for micro, batch in enumerate(device_batches(batches, device, cfg.train.cuda_prefetch)):
                sync = objective.no_sync() if world > 1 and micro < len(batches) - 1 else nullcontext()
                with sync:
                    probe = ActivationProbe(model) if diagnose and micro == 0 else None
                    with autocast(device, cfg.train.precision):
                        losses = compute(objective, batch, step)
                        flow_weights = (
                            losses["frames"]
                            if cfg.train.flow_reduction == "frame"
                            else torch.ones_like(losses["flow"])
                        )
                        flow_denominator = (
                            frame_denominator if cfg.train.flow_reduction == "frame" else flow_examples
                        )
                        loss = (
                            losses["flow"] * flow_weights
                        ).sum() * world / flow_denominator + cfg.train.duration_weight * losses[
                            "duration"
                        ].sum() * world / denominator
                        auxiliary = raw_objective.auxiliary(losses)
                        if auxiliary is not None:
                            loss = loss + auxiliary.sum() * world / flow_examples
                        if "latent_delta" in losses:
                            # The flow term's own weights and denominator: per frame the objective stays
                            # a convex quadratic in the prediction (see dacvae_tts.negatives).
                            delta = (losses["latent_delta"] * flow_weights).sum()
                            loss = loss + delta * world / flow_denominator
                    if probe is not None:
                        diagnostics["activation_max_abs"] = probe.close()
                        if model.grad_checkpoint == "selective":
                            diagnostics["gradient_contributions"] = (
                                "Unavailable under selective checkpointing, which allows a single backward"
                            )
                        elif world == 1:
                            diagnostics.update(
                                gradient_contributions(
                                    model,
                                    reduce_flow(losses["flow"], losses["frames"], cfg.train.flow_reduction),
                                    cfg.train.duration_weight * losses["duration"].mean(),
                                )
                            )
                        else:
                            diagnostics["gradient_contributions"] = (
                                "Use a single-process diagnostic run; autograd.grad is not used inside DDP"
                            )
                    if watch is not None:
                        watch.note("objective", loss, step)
                    elif not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite objective at update {step + 1}")
                    loss.backward()
                buckets += loss_buckets(losses["flow"], losses["times"], losses["frames"])
                metrics += torch.stack(
                    [
                        losses["loss"].detach().sum(),
                        (losses["flow"].detach() * flow_weights).sum(),
                        losses["duration"].detach().sum(),
                        losses["ctc"].detach().sum() if "ctc" in losses else losses["flow"].new_zeros(()),
                        losses["contrastive"].detach().sum()
                        if "contrastive" in losses
                        else losses["flow"].new_zeros(()),
                    ]
                )
                if "latent_delta" in losses:
                    negative_metrics += delta_sums(losses, flow_weights)
                teacher_metrics += raw_objective.teacher_sums(losses)
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if watch is not None:
                watch.note("gradient", norm, step)
            elif not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite gradient at update {step + 1}")
            if diagnose:
                diagnostics["gradient_groups_after_clip"] = gradient_groups(model)
                for key, value in diagnostics["gradient_groups_after_clip"].items():
                    inactive[key] = inactive.get(key, 0) + 1 if value == 0 else 0
                diagnostics["consecutive_inactive_diagnostic_checks"] = dict(inactive)
            optimizer.step()
            with torch.no_grad():
                decay = ema_rate(cfg.train.ema_decay, step, cfg.train.ema_warmup)
                torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - decay)
                for track_decay, average in averages.values():
                    decay = ema_rate(track_decay, step, cfg.train.ema_warmup)
                    torch._foreach_lerp_(list(average.parameters()), list(model.parameters()), 1 - decay)
            if (step + 1) % cfg.train.log_every == 0 or step == start_step or diagnose:
                if watch is not None:
                    watch.check()
                if world > 1:
                    dist.all_reduce(metrics)
                    dist.all_reduce(buckets)
                    if cfg.train.contrastive_mode == "latent_delta":
                        dist.all_reduce(negative_metrics)
                    dist.all_reduce(teacher_metrics)
                if rank == 0:
                    record = {
                        "step": step + 1,
                        "epoch": epoch,
                        "lr": learning_rate,
                        "loss": (
                            metrics[1] / flow_denominator
                            + cfg.train.duration_weight * metrics[2] / denominator
                        ).item(),
                        "flow": (metrics[1] / flow_denominator).item(),
                        "duration": (metrics[2] / denominator).item(),
                        "ctc": (metrics[3] / flow_examples).item(),
                        "contrastive": (metrics[4] / flow_examples).item(),
                        **raw_objective.teacher_record(teacher_metrics / flow_examples),
                        "elapsed_seconds": time.monotonic() - last_time,
                        "valid_target_frames": int(frame_denominator),
                        "gradient_norm_before_clip": float(norm),
                        "peak_cuda_gb": torch.cuda.max_memory_allocated(device) / 2**30
                        if device.type == "cuda"
                        else 0.0,
                        "flow_reduction": cfg.train.flow_reduction,
                        "time_and_length_buckets_sum_count": buckets.cpu().tolist(),
                        "diagnostics_rank0": diagnostics,
                    }
                    if cfg.train.contrastive_mode == "latent_delta":
                        record.update(delta_record(negative_metrics, flow_denominator))
                    print(json.dumps(record), flush=True)
                    with open(out / "train.jsonl", "a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    tracker.log(record, step=step + 1, prefix="train/")
                last_time = time.monotonic()
            if validation is not None and (step + 1) % cfg.train.validate_every == 0:
                if watch is not None:
                    watch.check()
                val = validate(
                    ema,
                    validation,
                    device,
                    cfg.train.precision,
                    reduction=cfg.train.flow_reduction,
                    duration_weight=cfg.train.duration_weight,
                )
                for key, (_, average) in averages.items():
                    extra = validate(
                        average,
                        validation,
                        device,
                        cfg.train.precision,
                        reduction=cfg.train.flow_reduction,
                        duration_weight=cfg.train.duration_weight,
                    )
                    label = key.replace(".", "_")  # dots are W&B's nested-key separator
                    val.update({f"{label}/{name}": value for name, value in extra.items()})
                if rank == 0:
                    record = {"step": step + 1, **val}
                    print(json.dumps(record), flush=True)
                    with open(out / "train.jsonl", "a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    tracker.log(val, step=step + 1, prefix="val/")
            stopping = args.stop_after is not None and step + 1 >= args.stop_after
            keeping = cfg.train.keep_every and (step + 1) % cfg.train.keep_every == 0
            if (
                (step + 1) % cfg.train.checkpoint_every == 0
                or step + 1 == cfg.train.steps
                or stopping
                or keeping
            ):
                if watch is not None:
                    watch.check()  # never write a checkpoint after an unnoticed nonfinite update
                states = [None] * world
                state = rng_state(device, raw_objective.rng)
                if world > 1:
                    dist.all_gather_object(states, state)
                else:
                    states[0] = state
                if rank == 0:
                    saved = {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": cfg.to_dict(),
                        "step": step + 1,
                        "epoch": epoch,
                        "batch_offset": batch_offset,
                        "rng": states,
                        "world_size": world,
                        "frame_budget": args.frame_budget,
                        "cache_path": str(Path(args.cache).resolve()),
                        "codec": data.meta,
                        **speaker_metadata(data),
                        "mean": data.mean,
                        "std": data.std,
                        "stage": "pretrain",
                        "init_from": getattr(args, "init_from", None),
                    }
                    saved.update({key: average.state_dict() for key, (_, average) in averages.items()})
                    if cfg.train.model_guidance_weight:
                        saved["recommended_guidance"] = 1.0  # guidance is baked in: sample without CFG
                    atomic_save(saved, out / "last.pt")
                    if keeping:
                        # Permanent, optimizer-free snapshot for later speech evaluation and selection.
                        keep = {k: v for k, v in saved.items() if k not in {"optimizer", "rng"}}
                        atomic_save(keep, out / f"step-{step + 1:07d}.pt")
                if world > 1:
                    dist.barrier()
            if stopping:
                break
    finally:
        if "tracker" in locals():
            tracker.finish()
        if dist.is_initialized():
            dist.destroy_process_group()
