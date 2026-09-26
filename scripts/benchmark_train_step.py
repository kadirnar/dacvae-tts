"""Training-step throughput of FlowTTS + Objective under the speed options of `dacvae_tts.speed`.

Synthetic within-utterance items of realistic lengths are batched under the frame budget by the
training sampler, collated by the training collate of each variant (padding, loader negatives), and
run through the same objective, loss reduction, finiteness checks, backward, clipping, optimizer and
EMA update as `training.train`. Every variant starts from the same weights. Reported per variant:
mean update time, valid (unpadded) frames per second, warm-up time (includes compilation), compiled
graphs and peak CUDA memory. Losses differ between variants only through their batches and random
draws; the options themselves keep the objective (see tests/test_speed.py).

RTX 4090, the real measurement (the full grid takes roughly 20-40 minutes, mostly compilation of the
compiled variants: one graph per padded length pair under compile_dynamic batch; --variants picks rows):
    python scripts/benchmark_train_step.py --config configs/nano_tr_w512.yaml --frame-budget 6000 \
        --output runs/benchmark-train-step.jsonl
CPU smoke test:
    python scripts/benchmark_train_step.py --config configs/tiny.yaml --device cpu --frames 20 60 \
        --text 10 40 --frame-budget 400 --batches 2 --steps 1 --repeats 1 --variants baseline,loader_negatives
"""

import argparse
import copy
import gc
import json
import math
import platform
import statistics
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from dacvae_tts.config import Config
from dacvae_tts.data import BucketBatchSampler, move_batch
from dacvae_tts.diagnostics import loss_buckets
from dacvae_tts.model import FlowTTS
from dacvae_tts.optim import build_optimizer
from dacvae_tts.speed import NonfiniteWatch, compile_blocks, training_loader
from dacvae_tts.text import BOS, BYTE_OFFSET, EOS, ID_DTYPE
from dacvae_tts.training import Objective, autocast

# Cumulative ladder: each variant adds one option, so neighbouring rows show its marginal effect.
LOADER = dict(strict_checks=False, pad_multiple=64, text_pad_multiple=32, loader_negatives=True)
VARIANTS = {
    "baseline": {},
    "no_strict": dict(strict_checks=False),
    "pad": dict(strict_checks=False, pad_multiple=64, text_pad_multiple=32),
    "loader_negatives": LOADER,
    "selective": dict(LOADER, grad_checkpoint="selective"),
    "every2": dict(LOADER, grad_checkpoint=2),
    "no_checkpoint": dict(LOADER, grad_checkpoint=False),
    "blocks": dict(LOADER, compile="blocks"),
    "blocks_selective": dict(LOADER, compile="blocks", grad_checkpoint="selective"),
    "blocks_selective_auto": dict(
        LOADER, compile="blocks", grad_checkpoint="selective", compile_dynamic="auto"
    ),
    "blocks_every2": dict(LOADER, compile="blocks", grad_checkpoint=2),
    # No recomputation at all: fits a 32 GB GPU at frame budget 6000 (RTX 5090).
    "blocks_no_checkpoint": dict(LOADER, compile="blocks", grad_checkpoint=False),
    "blocks_no_checkpoint_auto": dict(LOADER, compile="blocks", grad_checkpoint=False, compile_dynamic="auto"),
}


class SyntheticPool(Dataset):
    """Within-utterance items like LatentDataset's: random latents, word-like byte transcripts.

    The transcript length follows the utterance length (a speaking rate that varies by +-15%), and
    30% of the items have no prompt, as with prompt_dropout 0.3.
    """

    def __init__(self, count, frames, text, channels, layout, seed=0):
        rng = np.random.default_rng(seed)
        self.items, self.epoch = [], 0
        for index in range(count):
            total = int(rng.integers(frames[0], frames[1] + 1))
            share = (total - frames[0]) / max(frames[1] - frames[0], 1)
            tokens = (text[0] + share * (text[1] - text[0])) * rng.uniform(0.85, 1.15)
            characters = int(np.clip(round(tokens), text[0], text[1])) - 2  # BOS and EOS
            words = []
            while sum(len(w) + 1 for w in words) < characters:
                words.append(bytes(rng.integers(97, 123, int(rng.integers(2, 10))).astype(np.uint8)))
            body = np.frombuffer(b" ".join(words)[:characters].strip(), dtype=np.uint8)
            ids = np.concatenate([[BOS], body.astype(np.int64) + BYTE_OFFSET, [EOS]]).astype(ID_DTYPE)
            cut = 0 if rng.random() < 0.3 else int(total * rng.uniform(0.1, 0.6))
            cut = min(max(cut, 1 if layout == "segments" else 0), total - 1)
            latents = torch.from_numpy(rng.standard_normal((total, channels), dtype=np.float32))
            self.items.append(
                dict(
                    target=latents[cut:],
                    reference=latents[:cut],
                    text="",
                    reference_text="",
                    token_ids=ids,
                    reference_token_ids=None,
                    layout=layout,
                    uid=str(index),
                )
            )
        self.costs = np.array([len(i["target"]) + len(i["reference"]) for i in self.items], dtype=np.int64)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        _, index = index if isinstance(index, tuple) else (self.epoch, index)
        return dict(self.items[index])


def make_batches(pool, train, frame_budget, count, pin):
    items, collate_fn, costs = training_loader(pool, train)
    sampler = BucketBatchSampler(costs, train.batch_size, 0, 1, train.seed, frame_budget)
    batches = []
    for indices in sampler.batches()[:count]:
        batch = collate_fn([items[(0, index)] for index in indices])
        batches.append({k: v.pin_memory() for k, v in batch.items()} if pin else batch)
    return batches


def train_step(runner, objective, model, optimizer, ema, batch, host, train, device, watch, step):
    """One update as the single-process training loop (accumulation 1) performs it; `runner` is the
    objective or its compiled wrapper."""
    counts = torch.tensor([host["examples"], host["target_frames"]], device=device)
    denominator, frame_denominator = counts.unbind()
    flow_examples = denominator * train.batch_expansion
    frame_denominator = frame_denominator * train.batch_expansion
    optimizer.zero_grad(set_to_none=True)
    batch = move_batch(batch, device)
    with autocast(device, train.precision):
        losses = runner(batch)
        frame = train.flow_reduction == "frame"
        flow_weights = losses["frames"] if frame else torch.ones_like(losses["flow"])
        flow_denominator = frame_denominator if frame else flow_examples
        loss = (losses["flow"] * flow_weights).sum() / flow_denominator + train.duration_weight * losses[
            "duration"
        ].sum() / denominator
        auxiliary = objective.auxiliary(losses)
        if auxiliary is not None:
            loss = loss + auxiliary.sum() / flow_examples
    if watch is not None:
        watch.note("objective", loss, step)
    elif not torch.isfinite(loss):
        raise FloatingPointError(f"Nonfinite objective at update {step + 1}")
    loss.backward()
    buckets = loss_buckets(losses["flow"], losses["times"], losses["frames"])
    metrics = torch.stack([losses["loss"].detach().sum(), (losses["flow"].detach() * flow_weights).sum()])
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if watch is not None:
        watch.note("gradient", norm, step)
    elif not torch.isfinite(norm):
        raise FloatingPointError(f"Nonfinite gradient at update {step + 1}")
    optimizer.step()
    with torch.no_grad():
        decay = min(train.ema_decay, (1 + step) / (10 + step))
        torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - decay)
    return buckets, metrics


def run_variant(name, overrides, base, args, device, pool):
    cfg = copy.deepcopy(base)
    for key, value in overrides.items():
        if not hasattr(cfg.train, key):
            raise ValueError(f"Unknown train option in variant {name}: {key}")
        setattr(cfg.train, key, value)
    cfg.train.__post_init__()
    train = cfg.train
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    torch.manual_seed(0)
    model = FlowTTS(cfg.model).to(device)
    model.grad_checkpoint, model.strict_checks = train.grad_checkpoint, train.strict_checks
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = build_optimizer(
        model,
        train.optimizer,
        train.learning_rate,
        train.weight_decay,
        train.muon_momentum,
        device.type == "cuda",
    )
    objective = Objective(
        model,
        train.duration_weight,
        train.time_sampling,
        train.batch_expansion,
        train.ctc_weight,
        train.contrastive_weight,
        train.contrastive_margin,
    ).train()
    runner = objective
    if train.compile == "blocks":
        compile_blocks(model, train.compile_dynamic)
    elif train.compile == "model":
        model.forward = torch.compile(model.forward, dynamic=True)
    elif train.compile:
        runner = torch.compile(objective, dynamic=True)
    batches = make_batches(pool, train, args.frame_budget, args.batches, device.type == "cuda")
    hosts = [
        dict(
            examples=b["latents"].size(0),
            target_frames=int((b["valid"] & ~b["prompt_mask"]).sum()),
            valid_frames=int(b["valid"].sum()),
            padded_frames=b["valid"].numel(),
        )
        for b in batches
    ]
    watch = None if train.strict_checks else NonfiniteWatch(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    step = 0

    def run(count):
        nonlocal step
        frames = 0
        for _ in range(count):
            index = step % len(batches)
            train_step(
                runner,
                objective,
                model,
                optimizer,
                ema,
                batches[index],
                hosts[index],
                train,
                device,
                watch,
                step,
            )
            frames += hosts[index]["valid_frames"]
            step += 1
        return frames

    # Warm-up visits every batch once, so every padded shape is compiled before timing starts.
    sync()
    tick = time.perf_counter()
    run(len(batches) + args.warmup)
    sync()
    warmup = time.perf_counter() - tick
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    chunks, frames = [], 0
    for _ in range(args.repeats):
        sync()
        tick = time.perf_counter()
        frames += run(args.steps)
        sync()
        chunks.append(time.perf_counter() - tick)
    if watch is not None:
        watch.check()
    total = sum(chunks)
    return {
        "variant": name,
        "options": overrides,
        "contrastive_weight": train.contrastive_weight,
        "step_seconds": total / (args.repeats * args.steps),
        "step_seconds_spread": statistics.pstdev(chunks) / args.steps if len(chunks) > 1 else 0.0,
        "frames_per_second": frames / total,
        "warmup_seconds": warmup,
        "compiled_graphs": int(torch._dynamo.utils.counters["stats"]["unique_graphs"]),
        "peak_cuda_gb": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
        "batches": len(batches),
        "mean_batch_size": statistics.mean(h["examples"] for h in hosts),
        "padding_fraction": 1
        - sum(h["valid_frames"] for h in hosts) / sum(h["padded_frames"] for h in hosts),
        "distinct_shapes": len(
            {(tuple(b["latents"].shape[1:2]), tuple(b["tokens"].shape[1:])) for b in batches}
        ),
    }


def parse_value(text):
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--frame-budget", type=int, default=6000, help="As `train --frame-budget`")
    parser.add_argument("--frames", type=int, nargs=2, default=(150, 500), help="prompt+target frame range")
    parser.add_argument("--text", type=int, nargs=2, default=(80, 250), help="transcript token range")
    parser.add_argument("--pool", type=int, default=2048, help="synthetic utterances to batch from")
    parser.add_argument("--batches", type=int, default=12, help="distinct batches cycled through")
    parser.add_argument("--warmup", type=int, default=3, help="extra warm-up updates after one pass")
    parser.add_argument("--steps", type=int, default=10, help="timed updates per repeat")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--variants", default=",".join(VARIANTS), help=f"comma list of {sorted(VARIANTS)}")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME:KEY=VALUE[,KEY=VALUE]",
        help="extra variant with train overrides, e.g. fast:strict_checks=false,compile=blocks",
    )
    parser.add_argument("--contrastive", choices=["config", "off", "both"], default="config")
    parser.add_argument("--output", help="append one JSON line per variant")
    args = parser.parse_args(argv)
    if min(args.pool, args.batches, args.steps, args.repeats) < 1 or args.warmup < 0:
        parser.error("pool, batches, steps and repeats must be positive; warmup nonnegative")
    if not 1 <= args.frames[0] <= args.frames[1] or not 3 <= args.text[0] <= args.text[1]:
        parser.error("Invalid frame or text length range")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    base = Config.load(args.config)
    variants = {}
    for name in filter(None, args.variants.split(",")):
        if name not in VARIANTS:
            parser.error(f"Unknown variant {name}; choose from {sorted(VARIANTS)}")
        variants[name] = VARIANTS[name]
    for spec in args.set:
        name, _, assignments = spec.partition(":")
        pairs = [item.split("=", 1) for item in filter(None, assignments.split(","))]
        if not name or any(len(pair) != 2 for pair in pairs):
            parser.error(f"--set needs NAME:KEY=VALUE[,KEY=VALUE], got {spec!r}")
        variants[name] = {key: parse_value(value) for key, value in pairs}
    if args.contrastive != "config":
        off = {f"{name}+no_contrastive": dict(o, contrastive_weight=0.0) for name, o in variants.items()}
        variants = off if args.contrastive == "off" else {**variants, **off}
    pool = SyntheticPool(
        args.pool, args.frames, args.text, base.model.latent_dim, base.model.text_layout, base.train.seed
    )
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(device) if use_cuda else platform.processor() or "cpu",
                "config": args.config,
                "frame_budget": args.frame_budget,
                "frames": args.frames,
                "text": args.text,
                "precision": base.train.precision,
            }
        ),
        flush=True,
    )
    results = []
    for name, overrides in variants.items():
        try:
            record = run_variant(name, overrides, base, args, device, pool)
        except torch.cuda.OutOfMemoryError:
            record = {"variant": name, "options": overrides, "error": "out of memory"}
        except Exception as error:  # a failing variant (e.g. no compiler) must not end the grid
            traceback.print_exc()
            record = {
                "variant": name,
                "options": overrides,
                "error": f"{type(error).__name__}: {error}"[:300],
            }
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        results.append(record)
        print(json.dumps(record), flush=True)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "a") as stream:
                stream.write(json.dumps(record) + "\n")
    reference = next((r for r in results if "error" not in r), None)
    columns = ("s/update", "speedup", "frames/s", "warmup s", "graphs", "peak GB")
    widths = (9, 8, 10, 9, 6, 8)
    print(f"\n{'variant':34} " + " ".join(f"{c:>{w}}" for c, w in zip(columns, widths)))
    for r in results:
        if "error" in r:
            print(f"{r['variant']:34} {r['error']}")
            continue
        speedup = reference["step_seconds"] / r["step_seconds"] if reference else math.nan
        print(
            f"{r['variant']:34} {r['step_seconds']:9.4f} {speedup:7.2f}x {r['frames_per_second']:10.0f} "
            f"{r['warmup_seconds']:9.1f} {r['compiled_graphs']:6d} {r['peak_cuda_gb']:8.2f}"
        )
    return results


if __name__ == "__main__":
    main()
