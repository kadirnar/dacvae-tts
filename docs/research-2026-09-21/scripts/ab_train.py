"""Small controlled A/B runs on real DACVAE latents (LibriSpeech), using the repo's components.

Nothing in the repository is modified. Variants only change the flow parameterisation,
the time sampler or the packing factor. The metric is held-out-speaker flow error at a
fixed (t, noise) grid, reported both as velocity MSE and as implied clean-latent MSE.
This is a learnability probe at toy scale (~7 h, a few thousand updates), NOT a speech
quality result.
"""

import argparse
import copy
import json
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dacvae_tts.config import Config
from dacvae_tts.contracts import mask_values, sanitize
from dacvae_tts.data import BucketBatchSampler, LatentDataset, collate, move_batch
from dacvae_tts.model import FlowTTS, per_example_mse
from dacvae_tts.optim import build_optimizer
from dacvae_tts.text import BYTE_OFFSET

parser = argparse.ArgumentParser()
parser.add_argument("--cache", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--variants", nargs="+", required=True)
parser.add_argument("--config", default="tiny")
parser.add_argument("--steps", type=int, default=4000)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--eval-every", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
args = parser.parse_args()

device = torch.device("cuda")
torch.set_float32_matmul_precision("high")
CLIP = 0.05

VARIANTS = {
    # name: (prediction, time sampler, patch size, model overrides)
    "base": ("v", "uniform", 2, {}),
    "xpred": ("x", "uniform", 2, {}),
    "p1": ("v", "uniform", 1, {}),
    "xpred_p1": ("x", "uniform", 1, {}),
    "lognorm": ("v", "lognorm", 2, {}),
    "xpred_lognorm": ("x", "lognorm", 2, {}),
    "xpred_p1_lognorm": ("x", "lognorm", 1, {}),
    "edm": ("edm", "uniform", 2, {}),
    "edm_p1": ("edm", "uniform", 1, {}),
    "p1_lognorm": ("v", "lognorm", 1, {}),
    "edm_p1_lognorm": ("edm", "lognorm", 1, {}),
}


def velocity(model, prediction, xt, time_, batch, drop=None, cached=None):
    out = model(
        xt,
        time_,
        batch["prompt"],
        batch["prompt_mask"],
        batch["valid"],
        batch["tokens"],
        batch["segments"],
        drop=drop,
        cached=cached,
    )
    if prediction == "v":
        return out, xt + (1 - time_[:, None, None]) * out
    if prediction == "edm":
        t = time_[:, None, None]
        s = t.square() + (1 - t).square()
        # x_hat = c_skip * x_t + c_out * F ;  v_hat = (x_hat - x_t) / (1 - t), singularity cancels.
        x_hat = (t / s) * xt + ((1 - t) / s.sqrt()) * out
        return ((2 * t - 1) / s) * xt + out / s.sqrt(), x_hat
    # x-prediction: the network output is the clean latent; velocity follows analytically.
    return (out - xt) / (1 - time_[:, None, None]).clamp_min(CLIP), out


def training_loss(model, prediction, sampler, batch, duration_weight=0.1):
    mask = mask_values(batch["valid"], batch["prompt_mask"])
    x1 = sanitize(batch["latents"], batch["valid"])
    b = x1.size(0)
    if sampler == "uniform":
        time_ = torch.rand(b, device=device)
    else:
        time_ = torch.sigmoid(torch.randn(b, device=device))
    noise = sanitize(torch.randn_like(x1), batch["valid"])
    xt = (1 - time_[:, None, None]) * noise + time_[:, None, None] * x1
    xt = torch.where(batch["prompt_mask"][..., None], x1, xt) * batch["valid"][..., None]
    drop = torch.rand(b, device=device) < model.cfg.cond_dropout
    xt = xt.masked_fill((drop[:, None] & batch["prompt_mask"])[..., None], 0)
    cached = model.conditions(batch["prompt"], batch["prompt_mask"], batch["tokens"], batch["segments"])
    if prediction == "edm":
        t = time_[:, None, None]
        root = (t.square() + (1 - t).square()).sqrt()
        raw = model(
            xt,
            time_,
            batch["prompt"],
            batch["prompt_mask"],
            batch["valid"],
            batch["tokens"],
            batch["segments"],
            drop=drop,
            cached=cached,
        )
        # Unit-variance target at every t, written without the 1/(1-t) cancellation:
        # F* = (x1 - c_skip x_t) / c_out = ((1-t) x1 - t eps) / sqrt(t^2 + (1-t)^2)
        flow = per_example_mse(raw, ((1 - t) * x1 - t * noise) / root, mask)
    else:
        v_hat, _ = velocity(model, prediction, xt, time_, batch, drop, cached)
        flow = per_example_mse(v_hat, x1 - noise, mask)
    frames = mask.sum(1)
    characters = ((batch["segments"] == 1) & (batch["tokens"] >= BYTE_OFFSET)).sum(1)
    duration = model.predict_duration(
        batch["prompt"], batch["prompt_mask"], batch["tokens"], batch["segments"], cached=cached
    )
    duration_loss = F.smooth_l1_loss(
        duration.float(), (frames / characters.clamp_min(1)).log(), reduction="none"
    )
    return (flow + duration_weight * duration_loss).mean(), flow.mean().detach()


GRID = torch.tensor([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])


@torch.no_grad()
def evaluate(model, prediction, val_batches):
    model.eval()
    v_sum, x_sum, count = torch.zeros(len(GRID)), torch.zeros(len(GRID)), torch.zeros(len(GRID))
    for index, batch in enumerate(val_batches):
        generator = torch.Generator(device=device).manual_seed(1000 + index)
        mask = mask_values(batch["valid"], batch["prompt_mask"])
        x1 = sanitize(batch["latents"], batch["valid"])
        b = x1.size(0)
        bins = (torch.arange(b) + index) % len(GRID)
        time_ = GRID[bins].to(device)
        noise = sanitize(torch.randn(x1.shape, device=device, generator=generator), batch["valid"])
        xt = (1 - time_[:, None, None]) * noise + time_[:, None, None] * x1
        xt = torch.where(batch["prompt_mask"][..., None], x1, xt) * batch["valid"][..., None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_hat, x_hat = velocity(model, prediction, xt, time_, batch)
        v_err = per_example_mse(v_hat, x1 - noise, mask).cpu()
        x_err = per_example_mse(x_hat, x1, mask).cpu()
        v_sum.scatter_add_(0, bins, v_err)
        x_sum.scatter_add_(0, bins, x_err)
        count.scatter_add_(0, bins, torch.ones(b))
    model.train()
    v, x = v_sum / count, x_sum / count
    return {
        "v_mse_mean": float(v.mean()),
        "x_mse_mean": float(x.mean()),
        "v_mse_by_t": [round(float(a), 4) for a in v],
        "x_mse_by_t": [round(float(a), 4) for a in x],
    }


@torch.no_grad()
def conditioning_gain(model, prediction, val_batches):
    """Held-out v-MSE at mid noise levels with full / no-text / no-reference conditioning."""
    model.eval()
    totals = {"full": 0.0, "no_text": 0.0, "no_reference": 0.0}
    count = 0
    for index, batch in enumerate(val_batches):
        generator = torch.Generator(device=device).manual_seed(5000 + index)
        mask = mask_values(batch["valid"], batch["prompt_mask"])
        x1 = sanitize(batch["latents"], batch["valid"])
        b = x1.size(0)
        time_ = torch.tensor([0.25, 0.4, 0.55, 0.7], device=device)[(torch.arange(b) + index) % 4]
        noise = sanitize(torch.randn(x1.shape, device=device, generator=generator), batch["valid"])
        xt = (1 - time_[:, None, None]) * noise + time_[:, None, None] * x1
        xt = torch.where(batch["prompt_mask"][..., None], x1, xt) * batch["valid"][..., None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            text, text_valid, voice = model.conditions(
                batch["prompt"], batch["prompt_mask"], batch["tokens"], batch["segments"]
            )
            full, _ = velocity(model, prediction, xt, time_, batch, cached=(text, text_valid, voice))
            no_text, _ = velocity(
                model, prediction, xt, time_, batch, cached=(torch.zeros_like(text), text_valid, voice)
            )
            stripped = dict(batch)
            stripped["prompt"] = torch.zeros_like(batch["prompt"])
            stripped["prompt_mask"] = torch.zeros_like(batch["prompt_mask"])
            no_ref, _ = velocity(
                model,
                prediction,
                xt.masked_fill(batch["prompt_mask"][..., None], 0),
                time_,
                stripped,
                cached=(text, text_valid, torch.zeros_like(voice)),
            )
            # In-distribution probes: WRONG text / WRONG speaker prompt instead of zeroed inputs.
            wrong_text = dict(batch)
            wrong_text["tokens"] = batch["tokens"].roll(1, 0)
            wrong_text["segments"] = batch["segments"].roll(1, 0)
            shuffled_text, _ = velocity(model, prediction, xt, time_, wrong_text)
            donor = batch["latents"].roll(1, 0)
            donor_length = batch["prompt_mask"].roll(1, 0).sum(1).clamp_min(1)
            positions = torch.arange(x1.size(1), device=device)[None] % donor_length[:, None]
            tiled = donor.gather(1, positions[..., None].expand_as(donor))
            wrong_ref = dict(batch)
            wrong_ref["prompt"] = tiled * batch["prompt_mask"][..., None]
            xt_wrong = torch.where(batch["prompt_mask"][..., None], wrong_ref["prompt"], xt)
            shuffled_ref, _ = velocity(model, prediction, xt_wrong, time_, wrong_ref)
        for key, value in (
            ("full", full),
            ("no_text", no_text),
            ("no_reference", no_ref),
            ("shuffled_text", shuffled_text),
            ("shuffled_reference", shuffled_ref),
        ):
            totals[key] = totals.get(key, 0.0) + float(per_example_mse(value, x1 - noise, mask).sum())
        count += b
    model.train()
    return {key: value / count for key, value in totals.items()}


def main():
    train_data = LatentDataset(args.cache, "train", args.seed)
    val_data = LatentDataset(args.cache, "val", args.seed)
    val_sampler = BucketBatchSampler(val_data.costs, args.batch, 0, 1, 12345)
    val_batches = [
        move_batch(collate([val_data[(0, i)] for i in indices]), device)
        for indices in val_sampler.batches()[:12]
    ]
    print(
        "train rows", len(train_data), "val rows", len(val_data), "val batches", len(val_batches), flush=True
    )

    results = {}
    for name in args.variants:
        prediction, sampler_name, patch, overrides = VARIANTS[name]
        cfg = Config.load(f"configs/{args.config}.yaml")  # run from the repository root
        cfg.model.patch_size = patch
        for key, value in overrides.items():
            setattr(cfg.model, key, value)
        torch.manual_seed(args.seed)
        model = FlowTTS(cfg.model).to(device).train()
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = build_optimizer(model, args.optimizer, args.lr, 0.01, fused=True)
        sampler = BucketBatchSampler(train_data.costs, args.batch, 0, 1, args.seed)
        loader = DataLoader(
            train_data,
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=4,
            pin_memory=True,
            multiprocessing_context="spawn",
        )
        history, step, epoch, started = [], 0, 0, time.time()
        running = 0.0
        params = sum(p.numel() for p in model.parameters())
        print(f"== {name}: prediction={prediction} t={sampler_name} P={patch} params={params}", flush=True)
        torch.manual_seed(args.seed + 1)
        while step < args.steps:
            sampler.epoch = epoch
            for batch in loader:
                batch = move_batch(batch, device)
                lr = args.lr * min(1.0, (step + 1) / 300)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, flow = training_loss(model, prediction, sampler_name, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(name)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - 0.999)
                running = 0.98 * running + 0.02 * float(flow) if step else float(flow)
                step += 1
                if step % args.eval_every == 0 or step == args.steps:
                    record = {
                        "step": step,
                        "epoch": epoch,
                        "train_flow_running": running,
                        "minutes": (time.time() - started) / 60,
                        "model": evaluate(model, prediction, val_batches),
                        "ema": evaluate(ema, prediction, val_batches),
                    }
                    history.append(record)
                    print(json.dumps({"variant": name, **record}), flush=True)
                if step >= args.steps:
                    break
            epoch += 1
        results[name] = {
            "parameters": params,
            "history": history,
            "conditioning_gain_v_mse": conditioning_gain(model, prediction, val_batches),
        }
        print(
            json.dumps({"variant": name, "conditioning_gain": results[name]["conditioning_gain_v_mse"]}),
            flush=True,
        )
        json.dump(results, open(args.out, "w"), indent=1)
        del loader, model, ema, optimizer
        torch.cuda.empty_cache()
    print("done")


if __name__ == "__main__":
    main()
