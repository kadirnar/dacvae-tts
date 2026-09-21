"""Training throughput vs micro-batch size / packing / torch.compile (current model code)."""

import json
import sys
import time

import torch
from bench_model_lib import make_batch

from dacvae_tts.config import Config
from dacvae_tts.model import FlowTTS
from dacvae_tts.training import Objective

device = torch.device("cuda")
torch.set_float32_matmul_precision("high")
results = []


def timed(function, warmup=4, repeats=10):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / repeats * 1000


for name, compile_flags in (("tiny", (False, True)), ("small", (False,))):
    for patch in (2, 1):
        for compiled in compile_flags:
            cfg = Config.load(f"configs/{name}.yaml")  # run from the repository root
            cfg.model.patch_size = patch
            model = FlowTTS(cfg.model).to(device).train()
            model.grad_checkpoint = False
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
            objective = Objective(model).train()
            if compiled:
                objective = torch.compile(objective, dynamic=False)
            for b in (16, 32, 64, 128):
                batch = make_batch(b, 150, 250, device=device)

                def step():
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = objective(batch)["loss"].mean()
                    loss.backward()
                    optimizer.step()

                try:
                    torch.cuda.reset_peak_memory_stats()
                    ms = timed(step)
                    row = {
                        "model": name,
                        "patch": patch,
                        "compiled": compiled,
                        "batch": b,
                        "ms_per_step": ms,
                        "target_audio_seconds_per_second": b * 250 / 25 / (ms / 1000),
                        "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                    }
                except torch.cuda.OutOfMemoryError:
                    row = {"model": name, "patch": patch, "compiled": compiled, "batch": b, "oom": True}
                    torch.cuda.empty_cache()
                results.append(row)
                print(json.dumps(row), flush=True)
            torch.cuda.empty_cache()

json.dump(results, open(sys.argv[1] + "/bench_scaling.json", "w"), indent=1)
