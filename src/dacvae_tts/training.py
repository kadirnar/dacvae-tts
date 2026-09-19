"""DDP training with sample-weighted accumulation and per-rank resumable RNG state."""

import copy
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .config import Config
from .data import BucketBatchSampler, LatentDataset, collate, move_batch
from .diagnostics import ActivationProbe, gradient_contributions, gradient_groups, loss_buckets
from .model import FlowTTS, flow_loss, reduce_flow
from .parallel import device_batches, loader_options
from .text import BYTE_OFFSET


class Objective(nn.Module):
    def __init__(self, model, duration_weight=0.1):
        super().__init__()
        self.model = model
        self.duration_weight = duration_weight

    def forward(self, batch):
        details = flow_loss(
            self.model, batch, self.model.cfg.cond_dropout if self.training else 0, return_details=True
        )
        flow = details["flow"]
        frames = (batch["valid"] & ~batch["prompt_mask"]).sum(1)
        characters = ((batch["segments"] == 1) & (batch["tokens"] >= BYTE_OFFSET)).sum(1)
        log_rate = (frames / characters.clamp_min(1)).log()
        duration = self.model.predict_duration(
            batch["prompt"], batch["prompt_mask"], batch["tokens"], batch["segments"]
        )
        duration_loss = F.smooth_l1_loss(duration.float(), log_rate, reduction="none")
        return {"loss": flow + self.duration_weight * duration_loss, "duration": duration_loss, **details}


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


def load_model(path, device="cpu", ema=True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = Config.from_dict(checkpoint["config"])
    model = FlowTTS(cfg.model)
    model.load_state_dict(checkpoint["ema"] if ema else checkpoint["model"])
    return model.to(device).eval(), checkpoint


def lr_multiplier(step, warmup, steps):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(steps - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def rng_state(device):
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if state["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def validate(model, loader, device, precision, max_batches=20, reduction="utterance", duration_weight=0.1):
    objective = Objective(model, duration_weight).eval()
    totals = torch.zeros(4, device=device)
    devices = [device.index] if device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(12345)
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            with autocast(device, precision):
                losses = objective(move_batch(batch, device))
            weights = losses["frames"] if reduction == "frame" else torch.ones_like(losses["flow"])
            totals[0] += (losses["flow"] * weights).sum()
            totals[1] += weights.sum()
            totals[2] += losses["duration"].sum()
            totals[3] += losses["duration"].numel()
    if dist.is_initialized():
        dist.all_reduce(totals)
    return (totals[0] / totals[1].clamp_min(1) + duration_weight * totals[2] / totals[3].clamp_min(1)).item()


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
            "worker_threads",
            "prefetch_factor",
            "loader_start_method",
            "cuda_prefetch",
        ):
            value = getattr(args, key, None)
            if value is not None:
                setattr(cfg.train, key, value)
        if args.compile:
            cfg.train.compile = True
        cfg.train.__post_init__()
        if device.type == "cuda" and cfg.train.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support BF16; pass --precision fp32")
        torch.manual_seed(cfg.train.seed)
        random.seed(cfg.train.seed)
        data = LatentDataset(args.cache, "train", cfg.train.seed)
        if not data.meta.get("merged"):
            raise ValueError("Run merge on all prepared partitions before training")
        if cfg.model.latent_dim != data.channels:
            raise ValueError(f"Config latent_dim={cfg.model.latent_dim}, codec cache has {data.channels}")
        sampler = BucketBatchSampler(
            data.costs,
            cfg.train.batch_size,
            rank,
            world,
            cfg.train.seed,
            args.frame_budget,
            speaker_counts=data.group_end - data.group_start,
            speaker_balance=cfg.train.speaker_balance,
        )
        loader_rng = torch.Generator().manual_seed(cfg.train.seed + rank)
        loader = DataLoader(
            data,
            batch_sampler=sampler,
            collate_fn=collate,
            **loader_options(
                cfg.train.workers,
                device,
                cfg.train.prefetch_factor,
                cfg.train.worker_threads,
                cfg.train.loader_start_method,
            ),
            generator=loader_rng,
        )
        validation = None
        if not args.no_validation:
            val_data = LatentDataset(args.cache, "val", cfg.train.seed)
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
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            betas=(0.9, 0.95),
            fused=device.type == "cuda",
        )
        start_step, epoch, batch_offset = 0, 0, 0
        if args.resume:
            saved = torch.load(args.resume, map_location="cpu", weights_only=True)
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
            optimizer.load_state_dict(saved["optimizer"])
            start_step, epoch, batch_offset = saved["step"], saved["epoch"], saved["batch_offset"]
            restore_rng(saved["rng"][rank], device)
        else:
            torch.manual_seed(cfg.train.seed + rank)
            random.seed(cfg.train.seed + rank)
        objective = Objective(model, cfg.train.duration_weight).train()
        if cfg.train.compile:
            objective = torch.compile(objective, dynamic=True)
        if world > 1:
            objective = DDP(
                objective,
                device_ids=[device.index] if device.type == "cuda" else None,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        out = Path(args.output)
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
            print(
                json.dumps(
                    {
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "ranks": world,
                        "maximum_global_batch": cfg.train.batch_size * cfg.train.accumulation * world,
                        "train_rows": len(data),
                    }
                ),
                flush=True,
            )
        sampler.epoch, sampler.start_batch = epoch, batch_offset
        iterator = iter(loader)
        last_time = time.monotonic()
        inactive = {}
        for step in range(start_step, cfg.train.steps):
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
            learning_rate = cfg.train.learning_rate * lr_multiplier(step, cfg.train.warmup, cfg.train.steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            metrics = torch.zeros(3, device=device)
            buckets = torch.zeros(9, 2, device=device)
            diagnostics = {}
            diagnose = cfg.train.diagnostics_every > 0 and step % cfg.train.diagnostics_every == 0
            for micro, batch in enumerate(device_batches(batches, device, cfg.train.cuda_prefetch)):
                sync = objective.no_sync() if world > 1 and micro < len(batches) - 1 else nullcontext()
                with sync:
                    probe = ActivationProbe(model) if diagnose and micro == 0 else None
                    with autocast(device, cfg.train.precision):
                        losses = objective(batch)
                        flow_weights = (
                            losses["frames"]
                            if cfg.train.flow_reduction == "frame"
                            else torch.ones_like(losses["flow"])
                        )
                        flow_denominator = (
                            frame_denominator if cfg.train.flow_reduction == "frame" else denominator
                        )
                        loss = (
                            losses["flow"] * flow_weights
                        ).sum() * world / flow_denominator + cfg.train.duration_weight * losses[
                            "duration"
                        ].sum() * world / denominator
                    if probe is not None:
                        diagnostics["activation_max_abs"] = probe.close()
                        if world == 1:
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
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite objective at update {step + 1}")
                    loss.backward()
                buckets += loss_buckets(losses["flow"], losses["times"], losses["frames"])
                metrics += torch.stack(
                    [
                        losses["loss"].detach().sum(),
                        (losses["flow"].detach() * flow_weights).sum(),
                        losses["duration"].detach().sum(),
                    ]
                )
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite gradient at update {step + 1}")
            if diagnose:
                diagnostics["gradient_groups_after_clip"] = gradient_groups(model)
                for key, value in diagnostics["gradient_groups_after_clip"].items():
                    inactive[key] = inactive.get(key, 0) + 1 if value == 0 else 0
                diagnostics["consecutive_inactive_diagnostic_checks"] = dict(inactive)
            optimizer.step()
            with torch.no_grad():
                torch._foreach_lerp_(
                    list(ema.parameters()), list(model.parameters()), 1 - cfg.train.ema_decay
                )
            if (step + 1) % cfg.train.log_every == 0 or step == start_step or diagnose:
                if world > 1:
                    dist.all_reduce(metrics)
                    dist.all_reduce(buckets)
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
                        "elapsed_seconds": time.monotonic() - last_time,
                        "valid_target_frames": int(frame_denominator),
                        "gradient_norm_before_clip": float(norm),
                        "flow_reduction": cfg.train.flow_reduction,
                        "time_and_length_buckets_sum_count": buckets.cpu().tolist(),
                        "diagnostics_rank0": diagnostics,
                    }
                    print(json.dumps(record), flush=True)
                    with open(out / "train.jsonl", "a") as stream:
                        stream.write(json.dumps(record) + "\n")
                last_time = time.monotonic()
            if validation is not None and (step + 1) % cfg.train.validate_every == 0:
                val_loss = validate(
                    ema,
                    validation,
                    device,
                    cfg.train.precision,
                    reduction=cfg.train.flow_reduction,
                    duration_weight=cfg.train.duration_weight,
                )
                if rank == 0:
                    print(json.dumps({"step": step + 1, "validation_loss": val_loss}), flush=True)
            stopping = args.stop_after is not None and step + 1 >= args.stop_after
            if (step + 1) % cfg.train.checkpoint_every == 0 or step + 1 == cfg.train.steps or stopping:
                states = [None] * world
                state = rng_state(device)
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
                        "mean": data.mean,
                        "std": data.std,
                        "stage": "pretrain",
                    }
                    atomic_save(saved, out / "last.pt")
                if world > 1:
                    dist.barrier()
            if stopping:
                break
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
