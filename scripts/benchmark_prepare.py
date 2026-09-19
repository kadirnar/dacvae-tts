"""Bounded real-codec benchmark on synthetic signals, not a speech-quality evaluation.

Run: python scripts/benchmark_prepare.py --device cuda --output /tmp/prepare-benchmark.json
"""

import argparse
import contextlib
import io
import json
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import soundfile as sf
import torch

from dacvae_tts.codec import Codec
from dacvae_tts.prepare import encode_records, prepare


def benchmark_pipeline(args, records, sample_rate):
    """Measure loading/resampling, encoding and cache writing; no corpus is modified."""
    results = {"serial": [], "optimized": []}
    with tempfile.TemporaryDirectory(prefix="dacvae-prepare-benchmark-") as directory:
        root = Path(directory)
        rows = []
        for i, record in enumerate(records):
            path = root / f"audio-{i}.wav"
            # Exercise CPU resampling instead of only reading codec-rate files.
            sf.write(path, record["audio"][::3].numpy(), sample_rate // 3, subtype="FLOAT")
            rows.append(
                dict(
                    id=str(i),
                    audio=str(path),
                    speaker_id=f"speaker-{i // 2}",
                    text=f"Synthetic benchmark sentence {i}.",
                    split="train",
                )
            )
        manifest = root / "manifest.jsonl"
        manifest.write_text("\n".join(map(json.dumps, rows)))
        for repeat in range(args.repeats):
            for name in ("serial", "optimized") if repeat % 2 == 0 else ("optimized", "serial"):
                output = root / f"{name}-{repeat}"
                options = SimpleNamespace(
                    manifest=manifest,
                    output=output,
                    codec=args.codec,
                    device=args.device,
                    text_column="text",
                    audio_column="audio",
                    speaker_column="speaker_id",
                    text_normalization="unicode-v1",
                    min_seconds=0.5,
                    max_seconds=10,
                    seed=42,
                    shard_index=0,
                    num_shards=1,
                    precision=args.precision,
                    workers=0 if name == "serial" else 4,
                    prefetch=16,
                    batch_size=1 if name == "serial" else args.batch_size,
                    bucket_size=1 if name == "serial" else 256,
                    batch_seconds=args.batch_seconds,
                    no_fold_weight_norm=name == "serial",
                )
                tick = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    prepare(options)
                wall = time.perf_counter() - tick
                meta = json.loads((output / "metadata.json").read_text())
                results[name].append(
                    {
                        **meta["preparation"],
                        "total_wall_seconds": wall,
                        "accepted": meta["accepted"],
                        "rejected": meta["rejected"],
                    }
                )
    medians = {
        name: statistics.median(r["processing_seconds"] for r in runs) for name, runs in results.items()
    }
    return {
        "scope": "synthetic WAV load/resample, encode, index and shard writes; excludes merge",
        "runs": results,
        "median_processing_seconds": medians,
        "median_processing_speedup": medians["serial"] / medians["optimized"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec", default="facebook/dacvae-watermarked")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--records", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-seconds", type=float, default=64)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pipeline", action="store_true", help="Also benchmark temporary WAV-to-cache jobs")
    args = parser.parse_args()
    if min(args.records, args.repeats, args.batch_size, args.batch_seconds) < 1:
        parser.error("record/repeat/batch limits must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    baseline = Codec(args.codec, args.device, encoder_only=True)
    optimized = Codec(args.codec, args.device, encoder_only=True, fold_weight_norm=True)
    assert not any(p.requires_grad for p in optimized.model.parameters())
    records = []
    for i in range(args.records):
        # Different exact lengths sharing four hop-rounded buckets, including odd lengths.
        n = (1, 2, 4, 8)[i % 4] * baseline.sample_rate + 1 + (i * 29) % (baseline.hop_length - 1)
        t = torch.arange(n) / baseline.sample_rate
        waveform = 0.1 * torch.sin(2 * torch.pi * (120 + i * 13) * t) + 0.01 * torch.randn(n)
        records.append({"audio": waveform})
    seconds = sum(len(r["audio"]) for r in records) / baseline.sample_rate

    def serial():
        with torch.backends.cudnn.flags(allow_tf32=False):
            return [baseline.encode(r["audio"]).cpu() for r in records]

    def batched():
        counters = {"encoder_calls": 0, "oom_retries": 0}
        result = encode_records(
            optimized, records, args.batch_size, args.batch_seconds, args.precision, counters
        )
        return result, counters

    # Warm both paths and all tested shapes. Alternate order across repetitions.
    reference = serial()
    batched()
    timings = {"serial": [], "batched": []}
    if optimized.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(optimized.device)
    for repeat in range(args.repeats):
        for name in ("serial", "batched") if repeat % 2 == 0 else ("batched", "serial"):
            tick = time.perf_counter()
            if name == "serial":
                serial()
            else:
                output, counters = batched()
            timings[name].append(time.perf_counter() - tick)
    max_abs = max((a - b).abs().max().item() for a, b in zip(reference, output, strict=True))
    error = sum((a.double() - b.double()).square().sum().item() for a, b in zip(reference, output))
    energy = sum(a.double().square().sum().item() for a in reference)
    means = {name: statistics.median(values) for name, values in timings.items()}
    report = {
        "scope": "warm encoder + transfers only; synthetic signals; excludes disk, merge, decoder and startup",
        "codec": baseline.metadata,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(optimized.device) if optimized.device.type == "cuda" else "cpu",
        "records": args.records,
        "audio_seconds": seconds,
        "precision": args.precision,
        "cudnn_allow_tf32": False,
        "batch_size": args.batch_size,
        "batch_seconds": args.batch_seconds,
        "serial_encoder_calls": args.records,
        "batched": counters,
        "wall_seconds": timings,
        "median_seconds": means,
        "audio_seconds_per_wall_second": {name: seconds / wall for name, wall in means.items()},
        "median_speedup": means["serial"] / means["batched"],
        "max_latent_absolute_error": max_abs,
        "relative_latent_rmse": (error / max(energy, 1e-30)) ** 0.5,
        "shapes_match": all(a.shape == b.shape for a, b in zip(reference, output)),
        "peak_cuda_allocated_bytes_both_models": torch.cuda.max_memory_allocated(optimized.device)
        if optimized.device.type == "cuda"
        else None,
    }
    if args.precision == "fp32":
        for expected, actual in zip(reference, output, strict=True):
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    if args.pipeline:
        sample_rate = baseline.sample_rate
        del baseline, optimized
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        report["pipeline"] = benchmark_pipeline(args, records, sample_rate)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
