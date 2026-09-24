"""Encoded-silence latent of a cache's codec: the tail-silence padding and the target of quiet prompt cuts.

Encodes digital silence and a few draws of very low-level white noise with the cache's DACVAE, drops the
frames next to the padded edges and takes the per-channel median of the rest (robust to edge effects and to
the noise draws). Writes <cache>/silence.pt = {"frame": [C] normalized with the cache mean/std, "raw": [C],
"codec": codec metadata, "signal": settings, "report": checks}. Irodori-TTS and Echo pad with an
encoded-silence latent the same way. The waveform is not loudness-normalized even for a LUFS cache: that
would lift the noise floor to speech level; --loudness only has to match the cache's preprocessing tag.

The report gives the spread between the digital-silence and noise medians and the distance quantiles of real
cache frames to the silence frame (quiet frames of pauses should sit in the lowest percentiles).

  python scripts/silence_latent.py --cache data/tr55/merged --device cuda
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dacvae_tts.data import LatentDataset, load_stats


def silence_frames(codec, seconds, noise_dbfs, draws, edge):
    """Interior latent frames [N,C] of digital silence and `draws` noise signals at `noise_dbfs` RMS."""
    samples = int(seconds * codec.sample_rate)
    signals = {"digital": torch.zeros(samples)}
    for draw in range(draws):
        noise = torch.randn(samples, generator=torch.Generator().manual_seed(draw))
        signals[f"noise{draw}"] = noise * 10 ** (noise_dbfs / 20)
    frames = {}
    for name, audio in signals.items():
        z = codec.encode(audio).float().cpu()
        if len(z) <= 2 * edge:
            raise ValueError("Signal too short for the edge trim; raise --seconds")
        frames[name] = z[edge : len(z) - edge]
    return frames


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--codec", help="DACVAE checkpoint (default: the one recorded in the cache metadata)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--loudness", type=float, help="Cache LUFS target (default: from the metadata)")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--noise-dbfs", type=float, default=-80.0, help="RMS level of the low-level noise")
    parser.add_argument("--noise-draws", type=int, default=4)
    parser.add_argument("--edge-frames", type=int, default=3, help="Frames dropped at each padded edge")
    parser.add_argument("--probe", type=int, default=256, help="Cache rows for the distance report (0: off)")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing silence.pt")
    args = parser.parse_args(argv)
    from dacvae_tts.codec import Codec, check_compatibility

    cache = Path(args.cache)
    output = cache / "silence.pt"
    if output.exists() and not args.force:
        raise SystemExit(f"{output} exists; pass --force to recreate it")
    meta = json.loads((cache / "metadata.json").read_text())
    stats = load_stats(cache)
    loudness = args.loudness if args.loudness is not None else meta.get("loudness_lufs")
    codec = Codec(args.codec or meta["checkpoint"], args.device, encoder_only=True, loudness=loudness)
    check_compatibility(codec.metadata, meta)
    frames = silence_frames(codec, args.seconds, args.noise_dbfs, args.noise_draws, args.edge_frames)
    raw = torch.cat(list(frames.values())).median(0).values
    frame = (raw - stats["mean"]) / stats["std"]
    channels = len(raw)

    def distance(latent):  # normalized RMS distance per frame
        return ((latent - frame).square().sum(-1) / channels).sqrt()

    report = {
        "frames": sum(len(f) for f in frames.values()),
        "median_distance_by_signal": {
            name: float(distance((f.median(0).values - stats["mean"]) / stats["std"]))
            for name, f in frames.items()
        },
        "frame_distance_mean": float(
            distance((torch.cat(list(frames.values())) - stats["mean"]) / stats["std"]).mean()
        ),
        "distance_to_cache_mean": float((frame.square().sum() / channels).sqrt()),
    }
    if args.probe:
        data = LatentDataset(cache, "train", pairing="within", layout="joined")
        rows = np.unique(np.linspace(0, len(data) - 1, min(args.probe, len(data))).round().astype(int))
        distances = torch.cat([distance(data.row(int(i))["latents"]) for i in rows])
        quantiles = torch.tensor([0.01, 0.05, 0.25, 0.5])
        report["cache_frame_distance_quantiles"] = dict(
            zip(["p1", "p5", "p25", "p50"], torch.quantile(distances, quantiles).tolist())
        )
    torch.save(
        {
            "frame": frame.float(),
            "raw": raw.float(),
            "codec": codec.metadata,
            "signal": dict(
                seconds=args.seconds,
                noise_dbfs=args.noise_dbfs,
                noise_draws=args.noise_draws,
                edge=args.edge_frames,
            ),
            "report": report,
        },
        output,
    )
    print(json.dumps({"output": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
