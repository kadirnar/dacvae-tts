"""GPU micro-benchmarks of the current FlowTTS implementation (no repo changes)."""

import json
import sys
import time

import torch
from torch.nn import functional as F

from dacvae_tts.config import Config
from dacvae_tts.contracts import mask_values, sanitize
from dacvae_tts.model import FlowTTS, per_example_mse, sample
from dacvae_tts.text import BYTE_OFFSET
from dacvae_tts.training import Objective

device = torch.device("cuda")
torch.set_float32_matmul_precision("high")
results = {}


def timed(function, warmup=5, repeats=20):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / repeats * 1000


def make_batch(b, ref_frames, target_frames, ref_bytes=90, target_bytes=110, channels=128):
    length = ref_frames + target_frames
    latents = torch.randn(b, length, channels, device=device)
    prompt_mask = torch.arange(length, device=device)[None].expand(b, -1) < ref_frames
    valid = torch.ones(b, length, dtype=torch.bool, device=device)
    tokens = torch.randint(
        BYTE_OFFSET + 32, BYTE_OFFSET + 120, (b, ref_bytes + target_bytes + 3), device=device
    )
    tokens[:, 0], tokens[:, ref_bytes + 1], tokens[:, -1] = 1, 2, 3
    segments = torch.zeros_like(tokens)
    segments[:, ref_bytes + 2 :] = 1
    return {
        "latents": latents,
        "prompt": latents * prompt_mask[..., None],
        "prompt_mask": prompt_mask,
        "valid": valid,
        "tokens": tokens,
        "segments": segments,
    }


def shared_condition_objective(model, batch, duration_weight=0.1):
    """Same maths as Objective, but text/reference encoders run once instead of twice."""
    mask = mask_values(batch["valid"], batch["prompt_mask"])
    x1 = sanitize(batch["latents"], batch["valid"])
    b = x1.size(0)
    time_ = torch.rand(b, device=x1.device)
    noise = sanitize(torch.randn_like(x1), batch["valid"])
    xt = (1 - time_[:, None, None]) * noise + time_[:, None, None] * x1
    xt = torch.where(batch["prompt_mask"][..., None], x1, xt) * batch["valid"][..., None]
    drop = torch.rand(b, device=x1.device) < model.cfg.cond_dropout
    cached = model.conditions(batch["prompt"], batch["prompt_mask"], batch["tokens"], batch["segments"])
    pred = model(
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
    flow = per_example_mse(pred, x1 - noise, mask)
    frames = mask.sum(1)
    characters = ((batch["segments"] == 1) & (batch["tokens"] >= BYTE_OFFSET)).sum(1)
    duration = model.predict_duration(
        batch["prompt"], batch["prompt_mask"], batch["tokens"], batch["segments"], cached=cached
    )
    duration_loss = F.smooth_l1_loss(
        duration.float(), (frames / characters.clamp_min(1)).log(), reduction="none"
    )
    return (flow + duration_weight * duration_loss).mean()


for name in ("tiny", "small"):
    cfg = Config.load(f"configs/{name}.yaml")  # run from the repository root
    entry = {}
    # ---------------- inference -------------------------------------------------------
    model = FlowTTS(cfg.model).to(device).eval()
    for block in model.blocks:  # leave zero-init: timing is unaffected by values
        pass
    batch = make_batch(1, 150, 250)
    kwargs = {k: v for k, v in batch.items() if k != "latents"}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        cond = model.conditions(kwargs["prompt"], kwargs["prompt_mask"], kwargs["tokens"], kwargs["segments"])
        null = (torch.zeros_like(cond[0]), cond[1], torch.zeros_like(cond[2]))
        x = torch.randn_like(batch["latents"])
        t = torch.full((1,), 0.5, device=device)

        def two_calls():
            model(x, t, **kwargs, cached=cond)
            model(
                x.masked_fill(kwargs["prompt_mask"][..., None], 0),
                t,
                torch.zeros_like(kwargs["prompt"]),
                torch.zeros_like(kwargs["prompt_mask"]),
                kwargs["valid"],
                kwargs["tokens"],
                kwargs["segments"],
                cached=null,
            )

        x2 = torch.cat([x, x.masked_fill(kwargs["prompt_mask"][..., None], 0)])
        both = {
            "prompt": torch.cat([kwargs["prompt"], torch.zeros_like(kwargs["prompt"])]),
            "prompt_mask": torch.cat([kwargs["prompt_mask"], torch.zeros_like(kwargs["prompt_mask"])]),
            "valid": kwargs["valid"].repeat(2, 1),
            "tokens": kwargs["tokens"].repeat(2, 1),
            "segments": kwargs["segments"].repeat(2, 1),
        }
        cond2 = (torch.cat([cond[0], null[0]]), cond[1].repeat(2, 1), torch.cat([cond[2], null[2]]))
        t2 = t.repeat(2)

        def one_batched_call():
            model(x2, t2, **both, cached=cond2)

        entry["cfg_step_two_separate_calls_ms"] = timed(two_calls)
        entry["cfg_step_one_batched_call_ms"] = timed(one_batched_call)
        # Verify the batched evaluation is numerically the same guidance input.
        a = model(x, t, **kwargs, cached=cond)
        bth = model(x2, t2, **both, cached=cond2)
        entry["batched_vs_separate_max_abs_diff"] = float((a - bth[:1]).abs().max())
        entry["sample_16_steps_cfg_ms"] = timed(
            lambda: sample(model, **kwargs, steps=16, guidance=1.5, condition_cache=cond), 2, 5
        )
        entry["sample_16_steps_no_cfg_ms"] = timed(
            lambda: sample(model, **kwargs, steps=16, guidance=1.0, condition_cache=cond), 2, 5
        )
        # Cost of carrying the clean reference prefix through every block at every step.
        target_only = make_batch(1, 1, 250)
        tk = {k: v for k, v in target_only.items() if k != "latents"}
        tcond = model.conditions(tk["prompt"], tk["prompt_mask"], tk["tokens"], tk["segments"])
        entry["forward_b1_ref150_tgt250_ms"] = timed(lambda: model(x, t, **kwargs, cached=cond))
        entry["forward_b1_tgt250_only_ms"] = timed(
            lambda: model(target_only["latents"], t, **tk, cached=tcond)
        )
        big = make_batch(16, 150, 250)
        bk = {k: v for k, v in big.items() if k != "latents"}
        bcond = model.conditions(bk["prompt"], bk["prompt_mask"], bk["tokens"], bk["segments"])
        tb = torch.full((16,), 0.5, device=device)
        small_b = make_batch(16, 1, 250)
        sk = {k: v for k, v in small_b.items() if k != "latents"}
        scond = model.conditions(sk["prompt"], sk["prompt_mask"], sk["tokens"], sk["segments"])
        entry["forward_b16_ref150_tgt250_ms"] = timed(lambda: model(big["latents"], tb, **bk, cached=bcond))
        entry["forward_b16_tgt250_only_ms"] = timed(lambda: model(small_b["latents"], tb, **sk, cached=scond))

    # ---------------- training throughput ---------------------------------------------
    for patch in (2, 1):
        cfg.model.patch_size = patch
        model = FlowTTS(cfg.model).to(device).train()
        model.grad_checkpoint = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        objective = Objective(model).train()
        train_batch = make_batch(16, 150, 250)

        def current_step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = objective(train_batch)["loss"].mean()
            loss.backward()
            optimizer.step()

        def shared_step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = shared_condition_objective(model, train_batch)
            loss.backward()
            optimizer.step()

        torch.cuda.reset_peak_memory_stats()
        entry[f"train_step_b16_P{patch}_current_ms"] = timed(current_step, 5, 15)
        entry[f"train_step_b16_P{patch}_current_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        torch.cuda.reset_peak_memory_stats()
        entry[f"train_step_b16_P{patch}_shared_conditions_ms"] = timed(shared_step, 5, 15)
        entry[f"train_step_b16_P{patch}_shared_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        entry[f"parameters_P{patch}"] = sum(p.numel() for p in model.parameters())
    results[name] = entry
    print(name, json.dumps(entry, indent=1), flush=True)

json.dump(results, open(sys.argv[1] + "/bench_model.json", "w"), indent=1)
