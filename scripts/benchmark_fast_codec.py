"""Compare exact codec backends on synthetic signals; excludes TTS and speech metrics."""

import argparse
import gc
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from dacvae_tts.codec import Codec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--codec", default="facebook/dacvae-watermarked")
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.seconds, args.batch_size, args.repeats, args.threads) <= 0:
        parser.error("Workload sizes must be positive")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def timed(function):
        sync()
        tick = time.perf_counter()
        value = function()
        sync()
        return time.perf_counter() - tick, value

    def error(a, b):
        return {
            "max_abs": (a - b).abs().max().item(),
            "relative_rms": ((a - b).square().mean() / b.square().mean().clamp_min(1e-20)).sqrt().item(),
        }

    results = {}
    variants = ["reference", "fast"]
    if device.type == "cuda":
        variants += ["fast_graphs"]
    if args.compile:
        variants += ["fast_compile"]
        if device.type == "cuda":
            variants += ["fast_compile_graphs"]
    if args.channels_last:
        variants += ["fast_channels_last"]
    reference_z = reference_y = audios = None
    for variant in variants:
        print(f"Benchmarking {variant}", flush=True)
        setup, codec = timed(
            lambda: Codec(
                args.codec,
                device,
                fold_weight_norm=True,
                backend="reference" if variant == "reference" else "fast",
                cuda_graphs="graphs" in variant,
                compile_model="compile" in variant,
                layout="channels_last" if variant == "fast_channels_last" else "native",
            )
        )
        if audios is None:
            generator = torch.Generator().manual_seed(17)
            audios = [
                torch.randn(round(args.seconds * codec.sample_rate), generator=generator) * 0.05
                for _ in range(args.batch_size)
            ]
        first_encode, z = timed(lambda: codec.encode_batch(audios))
        if reference_z is None:
            reference_z = z
        torch.manual_seed(99)
        first_decode, y = timed(lambda: codec.decode(reference_z[0]))
        if reference_y is None:
            reference_y = y
        numerical = {"latent": error(z, reference_z), "waveform_fixed_latent_and_seed": error(y, reference_y)}
        warmup_start = time.perf_counter()
        for _ in range(4):
            codec.encode_batch(audios)
            codec.decode(reference_z[0])
        sync()
        warmup = time.perf_counter() - warmup_start
        encode_times, decode_times = [], []
        for _ in range(args.repeats):
            duration, _ = timed(lambda: codec.encode_batch(audios))
            encode_times.append(duration)
            duration, _ = timed(lambda: codec.decode(reference_z[0]))
            decode_times.append(duration)
        # Fresh inputs at captured geometry and another duration: detect stale/static replay.
        changed = [a * 0.4 for a in audios]
        changed_z = codec.encode_batch(changed)
        short_z = codec.encode_batch([a[: 2 * codec.hop_length] for a in audios])
        adapter = codec._fast
        codec._fast = None
        changed_reference = codec.encode_batch(changed)
        short_reference = codec.encode_batch([a[: 2 * codec.hop_length] for a in audios])
        codec._fast = adapter
        numerical["changed_input"] = error(changed_z, changed_reference)
        numerical["short_shape_fallback"] = error(short_z, short_reference)
        results[variant] = dict(
            setup_seconds=setup,
            first_encode_seconds=first_encode,
            first_decode_seconds=first_decode,
            warmup_seconds=warmup,
            encode_seconds=encode_times,
            decode_seconds=decode_times,
            median_encode_seconds=statistics.median(encode_times),
            median_decode_seconds=statistics.median(decode_times),
            numerical=numerical,
            graph_statistics=adapter.graph_statistics() if adapter else {},
            codec=codec.metadata,
        )
        codec = adapter = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for name, result in results.items():
        for stage in ("encode", "decode"):
            key = f"median_{stage}_seconds"
            result[f"{stage}_speedup_vs_reference"] = results["reference"][key] / result[key]
    output = {
        "scope": "synthetic FP32 codec: batched encode with host transfers; single decode with full watermark; no TTS",
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else (platform.processor() or platform.machine()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "threads": args.threads,
        "seconds_per_audio": args.seconds,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                k: {n: v for n, v in r.items() if n.startswith("median_") or "speedup" in n}
                for k, r in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
