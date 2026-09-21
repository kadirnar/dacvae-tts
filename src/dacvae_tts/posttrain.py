"""Offline candidate ranking, flow-error preference learning, and trajectory distillation.

The preference loss is a research surrogate, NOT a deterministic-flow log likelihood.
"""

import copy
import hashlib
import json
import math
import random
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import soundfile as sf
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .codec import backend_options, check_compatibility
from .data import LatentDataset, collate, jsonl, move_batch
from .inference import Synthesizer
from .model import flow_loss, per_example_mse, sample
from .optim import build_optimizer
from .parallel import loader_options
from .training import Objective, atomic_save, autocast, distributed_device, load_model


def representation_id(checkpoint):
    digest = hashlib.sha256()
    for key in ("mean", "std"):
        digest.update(checkpoint[key].cpu().numpy().tobytes())
    codec = checkpoint["codec"]
    for key in ("weights_sha256", "preprocessing", "text_normalization"):
        if key in codec:
            digest.update(f"{key}:{codec[key]}".encode())
    digest.update(
        json.dumps(
            {k: codec[k] for k in ("checkpoint", "latent_dim", "hop_length", "sample_rate", "posterior")},
            sort_keys=True,
        ).encode()
    )
    return digest.hexdigest()


def candidates(args):
    tts = Synthesizer(args.checkpoint, args.device, args.precision, codec_options=backend_options(args))
    dataset = LatentDataset(args.cache, args.split)
    check_compatibility(dataset.meta, tts.checkpoint["codec"])
    if not torch.equal(dataset.mean, tts.mean.cpu()) or not torch.equal(dataset.std, tts.std.cpu()):
        raise ValueError("Candidate cache and model must use the same training statistics")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / f"candidates-{args.shard_index:03d}.jsonl"
    if manifest.exists():
        raise ValueError("Candidate manifest exists")
    identity = representation_id(tts.checkpoint)
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), min(args.limit, len(dataset)))
    with open(manifest, "w") as stream:
        for position, index in enumerate(indices):
            if position % args.num_shards != args.shard_index:
                continue
            item = dataset[index]
            group = f"{index:09d}"
            reference_path = output / f"{group}-reference.pt"
            reference_audio = output / f"{group}-reference.wav"
            torch.save(
                {
                    "reference": item["reference"],
                    "reference_text": item["reference_text"],
                    "text": item["text"],
                    "representation": identity,
                },
                reference_path,
            )
            ref_wave = tts.codec.decode(item["reference"].to(tts.device) * tts.std + tts.mean)
            sf.write(reference_audio, ref_wave.numpy(), tts.codec.sample_rate, subtype="FLOAT")
            batch = tts.make_batch(
                item["reference"], item["reference_text"], item["text"], duration_scale=args.duration_scale
            )
            for k in range(args.candidates):
                seed = args.seed + position * args.candidates + k
                audio, target, timing = tts.generate(batch, args.steps, args.guidance, seed, args.sway)
                latent_path = output / f"{group}-{k:02d}.pt"
                audio_path = latent_path.with_suffix(".wav")
                torch.save({"target": target, "representation": identity}, latent_path)
                sf.write(audio_path, audio.numpy(), tts.codec.sample_rate, subtype="FLOAT")
                row = {
                    "group": group,
                    "uid": item["uid"],
                    "reference_uid": item["reference_uid"],
                    "speaker": item["speaker"],
                    "split": args.split,
                    "text": item["text"],
                    "reference": str(reference_path),
                    "reference_audio": str(reference_audio),
                    "audio": str(audio_path),
                    "latent": str(latent_path),
                    "representation": identity,
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    **timing,
                }
                stream.write(json.dumps(row) + "\n")
                stream.flush()


def reward(row):
    required = ("wer", "cer", "dnsmos_ovrl", "speaker_similarity")
    if any(key not in row or not math.isfinite(row[key]) for key in required):
        raise ValueError(f"Preference ranking requires finite metrics: {required}")
    return (
        -2 * min(row["wer"], 2)
        - min(row["cer"], 2)
        + 0.5 * (row["dnsmos_ovrl"] - 1) / 4
        + row["speaker_similarity"]
    )


def rank_pairs(args):
    groups = defaultdict(list)
    for row in jsonl(args.scores):
        if row.get("split") != "train":
            raise ValueError("Only training-split candidates may become post-training pairs")
        if "error" not in row:
            row["reward"] = reward(row)
            groups[(row["reference"], row["group"])].append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output, "x") as stream:
        for rows in groups.values():
            rows.sort(key=lambda row: row["reward"], reverse=True)
            if len(rows) < 2:
                continue
            winner, loser = rows[0], rows[-1]
            if any(
                winner[key] != loser[key] for key in ("text", "reference", "representation", "checkpoint")
            ):
                raise ValueError("Mismatched candidate conditions")
            if winner["reward"] - loser["reward"] < args.min_margin:
                continue
            if winner["wer"] > args.max_wer or winner["cer"] > args.max_cer:
                continue
            if winner["speaker_similarity"] < args.min_similarity:
                continue
            # A higher MOS score must not buy worse transcript accuracy or voice identity.
            if (
                winner["wer"] > loser["wer"]
                or winner["cer"] > loser["cer"]
                or winner["speaker_similarity"] < loser["speaker_similarity"] - 0.01
                or winner["dnsmos_ovrl"] < loser["dnsmos_ovrl"] - 0.05
            ):
                continue
            stream.write(
                json.dumps(
                    {
                        "winner": winner["latent"],
                        "loser": loser["latent"],
                        "reference": winner["reference"],
                        "representation": winner["representation"],
                        "split": "train",
                        "winner_metrics": winner,
                        "loser_metrics": loser,
                    }
                )
                + "\n"
            )
            count += 1
    print(json.dumps({"groups": len(groups), "accepted_pairs": count}))
    if not count:
        raise ValueError("No acceptable pairs; improve the baseline or inspect candidate metrics")


class PairDataset(Dataset):
    def __init__(self, path, representation):
        self.rows = list(jsonl(path))
        if not self.rows:
            raise ValueError("Empty preference dataset")
        if any(r.get("split") != "train" or r["representation"] != representation for r in self.rows):
            raise ValueError("Preference data is not training-only or uses incompatible latent statistics")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        ref = torch.load(row["reference"], weights_only=True)
        win = torch.load(row["winner"], weights_only=True)
        lose = torch.load(row["loser"], weights_only=True)
        if any(obj["representation"] != row["representation"] for obj in (ref, win, lose)):
            raise ValueError("Latent representation mismatch")
        if win["target"].shape != lose["target"].shape:
            raise ValueError("Preference pairs require equal durations; rank within a fixed duration group")
        base = {key: ref[key] for key in ("reference", "reference_text", "text")}
        return ({**base, "target": win["target"]}, {**base, "target": lose["target"]})


def collate_pairs(rows):
    return {"winner": collate([r[0] for r in rows]), "loser": collate([r[1] for r in rows])}


def preference_loss(win, lose, ref_win, ref_lose, beta=10.0):
    # Lower winner error relative to the frozen baseline gives a positive preference logit.
    logit = -beta * ((win - ref_win) - (lose - ref_lose))
    return -F.logsigmoid(logit)


class PreferenceObjective(nn.Module):
    def __init__(self, model, reference, beta=10.0, anchor=0.1, replay_weight=1.0):
        super().__init__()
        self.model, self.reference = model, reference.eval().requires_grad_(False)
        self.beta, self.anchor, self.replay_weight = beta, anchor, replay_weight
        self.replay_objective = Objective(model)

    def forward(self, pair, replay):
        winner, loser = pair["winner"], pair["loser"]
        noise = torch.randn_like(winner["latents"])
        time = torch.rand(noise.size(0), device=noise.device)
        win = flow_loss(self.model, winner, 0, time, noise)
        lose = flow_loss(self.model, loser, 0, time, noise)
        with torch.no_grad():
            ref_win = flow_loss(self.reference, winner, 0, time, noise)
            ref_lose = flow_loss(self.reference, loser, 0, time, noise)
        pref = preference_loss(win, lose, ref_win, ref_lose, self.beta)
        replay_loss = self.replay_objective(replay)["loss"].mean()
        return pref.mean() + self.anchor * win.mean() + self.replay_weight * replay_loss


def distill_cache(args):
    if args.teacher_steps % args.student_steps or args.student_steps < 1:
        raise ValueError("teacher-steps must be divisible by student-steps")
    tts = Synthesizer(args.checkpoint, args.device, args.precision, codec_options=backend_options(args))
    data = LatentDataset(args.cache, "train")
    check_compatibility(data.meta, tts.checkpoint["codec"])
    if not torch.equal(data.mean, tts.mean.cpu()) or not torch.equal(data.std, tts.std.cpu()):
        raise ValueError("Teacher and cache statistics differ")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = representation_id(tts.checkpoint)
    indices = random.Random(args.seed).sample(range(len(data)), min(args.limit, len(data)))
    with open(output / f"trajectories-{args.shard_index:03d}.jsonl", "x") as stream:
        for position, index in enumerate(indices):
            if position % args.num_shards != args.shard_index:
                continue
            item = data[index]
            batch = move_batch(collate([item]), tts.device)
            condition = {k: v for k, v in batch.items() if k != "latents"}
            with autocast(tts.device, args.precision):
                _, times, states = sample(
                    tts.model,
                    **condition,
                    steps=args.teacher_steps,
                    guidance=args.guidance,
                    seed=args.seed + position,
                    sway=args.sway,
                    return_trajectory=True,
                )
            stride = args.teacher_steps // args.student_steps
            path = output / f"trajectory-{index:09d}.pt"
            torch.save(
                {
                    "states": states[::stride, 0].cpu(),
                    "times": times[::stride].cpu(),
                    "condition": {k: v.cpu() for k, v in condition.items()},
                    "representation": identity,
                },
                path,
            )
            stream.write(
                json.dumps(
                    {
                        "path": str(path),
                        "split": "train",
                        "representation": identity,
                        "student_steps": args.student_steps,
                        "sway": args.sway,
                        "teacher_guidance": args.guidance,
                    }
                )
                + "\n"
            )


class TrajectoryDataset(Dataset):
    def __init__(self, path, representation):
        self.rows = list(jsonl(path))
        if not self.rows or any(
            r["split"] != "train" or r["representation"] != representation for r in self.rows
        ):
            raise ValueError("Empty or incompatible distillation dataset")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        obj = torch.load(row["path"], weights_only=True)
        if obj["representation"] != row["representation"]:
            raise ValueError("Trajectory representation mismatch")
        return obj


def list_collate(items):
    return items


class DistillObjective(nn.Module):
    def __init__(self, model, replay_weight=0.1):
        super().__init__()
        self.model = model
        self.replay = Objective(model)
        self.replay_weight = replay_weight

    def forward(self, trajectories, replay):
        device = replay["latents"].device
        losses = []
        for obj in trajectories:
            index = torch.randint(len(obj["times"]) - 1, ()).item()
            t0, t1 = obj["times"][index : index + 2].to(device)
            start, end = obj["states"][index : index + 2].to(device)
            condition = move_batch(obj["condition"], device)
            prediction = self.model(start[None], t0[None], **condition)
            target = (end - start) / (t1 - t0)
            mask = condition["valid"] & ~condition["prompt_mask"]
            losses.append(per_example_mse(prediction, target[None], mask).mean())
        return torch.stack(losses).mean() + self.replay_weight * self.replay(replay)["loss"].mean()


def post_train(args):
    device, rank, world = distributed_device(args.device)
    try:
        torch.manual_seed(args.seed)
        model, saved = load_model(args.checkpoint, device)
        model.train()
        model.grad_checkpoint = args.grad_checkpoint
        identity = representation_id(saved)
        real = LatentDataset(args.cache, "train", args.seed)
        check_compatibility(real.meta, saved["codec"])
        if not torch.equal(real.mean, saved["mean"]) or not torch.equal(real.std, saved["std"]):
            raise ValueError("Replay cache differs from pretraining statistics")
        if args.mode == "preference":
            data = PairDataset(args.data, identity)
            objective = PreferenceObjective(
                model, copy.deepcopy(model), args.beta, args.anchor, args.replay_weight
            )
            collator = collate_pairs
        else:
            data = TrajectoryDataset(args.data, identity)
            objective = DistillObjective(model, args.replay_weight)
            collator = list_collate
        sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
        loader = DataLoader(
            data,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=collator,
            **loader_options(
                args.workers,
                device,
                getattr(args, "prefetch_factor", None) or 2,
                getattr(args, "worker_threads", None) or 1,
                getattr(args, "loader_start_method", None) or "spawn",
            ),
            generator=torch.Generator().manual_seed(args.seed + rank),
        )
        # Fine-tune with the optimizer family used for pretraining unless explicitly overridden.
        optimizer = build_optimizer(
            model,
            getattr(args, "optimizer", None) or saved["config"]["train"].get("optimizer", "adamw"),
            args.learning_rate,
            0.01,
            saved["config"]["train"].get("muon_momentum", 0.95),
            fused=device.type == "cuda",
        )
        if world > 1:
            objective = DDP(
                objective,
                device_ids=[device.index] if device.type == "cuda" else None,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        torch.manual_seed(args.seed + rank)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        epoch, offset = 0, 0
        iterator = iter(loader)
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=device)
            for micro in range(args.accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    sampler.set_epoch(epoch)
                    iterator = iter(loader)
                    batch = next(iterator)
                if args.mode == "preference":
                    batch = {k: move_batch(v, device) for k, v in batch.items()}
                # Deterministic rank-distinct replay, cycling through the same corpus only.
                replay_indices = [
                    (offset * world + rank + i * world) % len(real) for i in range(args.batch_size)
                ]
                offset += args.batch_size
                replay = move_batch(collate([real[(epoch, i)] for i in replay_indices]), device)
                sync = objective.no_sync() if world > 1 and micro < args.accumulation - 1 else nullcontext()
                with sync:
                    with autocast(device, args.precision):
                        loss = objective(batch, replay) / args.accumulation
                    loss.backward()
                total += loss.detach()
            grad = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad):
                raise FloatingPointError("Nonfinite post-training gradient")
            optimizer.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 0.01)
            if step % 10 == 0:
                if world > 1:
                    dist.all_reduce(total)
                if rank == 0:
                    print(json.dumps({"step": step + 1, "loss": total.item() / world}), flush=True)
            if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
                if rank == 0:
                    result = {k: saved[k] for k in ("config", "codec", "mean", "std")}
                    result.update(
                        {
                            "model": model.state_dict(),
                            "ema": ema.state_dict(),
                            "stage": args.mode,
                            "step": step + 1,
                            "parent": str(Path(args.checkpoint).resolve()),
                            "posttrain_args": vars(args),
                            "recommended_guidance": 1.0 if args.mode == "distill" else 1.5,
                        }
                    )
                    result["posttrain_args"] = {k: v for k, v in vars(args).items() if k != "func"}
                    atomic_save(result, Path(args.output) / f"{args.mode}-{step + 1:06d}.pt")
                if world > 1:
                    dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
