"""Build the tempo-variant latent store of a cache for prompt tempo perturbation (train.tempo_prompt_prob).

Every row's waveform is stretched to each tempo with WSOLA (dacvae_tts.tempo: pitch and formants kept, only the
number of pitch periods changes) and re-encoded with the cache's DACVAE checkpoint; the store is a sidecar
directory (format: dacvae_tts/tempo.py), the cache is never modified. Audio comes from each row's original file
with the cache's loudness normalization when it still exists, else from decoding the stored latents (rows prepared
from embedded Parquet audio). Tempo x1.0 is stored as well: it is the same decode->encode round trip without the
stretch, drawn among the factors so that "re-encoded" is not a cue for "other tempo".

  python scripts/build_tempo_variants.py build --cache data/tr55/clean --output data/tr55/clean/tempo/wsola-v1 \
      --device cuda
  # 8 GPUs: one partition per GPU, then merge
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i python scripts/build_tempo_variants.py build --cache C \
      --output C/tempo/wsola-v1 --device cuda --shard-index $i --num-shards 8 & done; wait
  python scripts/build_tempo_variants.py merge --output C/tempo/wsola-v1 --cache C

then train with `tempo_prompt_prob: 0.3` and `tempo_variants: tempo/wsola-v1` (configs/experiments/
tr_w512_tempo_prompts.yaml). Storage is about sum(1/t) x the cache's latents: ~5.1x for the five default tempos.
Resumable: (row, tempo) records already in a partition are skipped.
"""

import argparse
import json
import sys
from pathlib import Path

from dacvae_tts.teacher import CacheAudio, ParquetAudio
from dacvae_tts.tempo import DEFAULT_TEMPOS, build_tempo_variants, merge_tempo_parts, tempo_key


def load_codec(cache, checkpoint, device):
    """DACVAE of the cache, checked against its metadata; loudness is applied by CacheAudio, not here."""
    from dacvae_tts.codec import Codec

    meta = json.loads((Path(cache) / "metadata.json").read_text())
    codec = Codec(checkpoint or meta["checkpoint"], device)
    for key in ("sample_rate", "hop_length", "latent_dim"):
        if getattr(codec, key) != meta[key]:
            raise ValueError(f"Codec {key}={getattr(codec, key)} does not match the cache ({meta[key]})")
    return codec


def build(args):
    import torch

    codec = load_codec(args.cache, args.codec, args.device)

    def encode(waveform):
        return codec.encode(torch.from_numpy(waveform)).cpu().numpy()

    if args.audio_source == "parquet":  # the original audio from the Hub dataset: no decode round trip
        import os

        audio = ParquetAudio(args.cache, args.parquet_repo, args.parquet_dir or Path(args.output) / "parquet",
                             token=os.environ.get("HF_TOKEN"))
    else:
        audio = CacheAudio(args.cache, args.audio_source, codec.decode)
    return build_tempo_variants(args.cache, args.output, encode, audio, [tempo_key(t) for t in args.tempos],
                                args.splits, args.shard_index, args.num_shards, progress=not args.quiet)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build", help="Stretch, encode and store the rows of one partition")
    b.add_argument("--cache", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--tempos", nargs="+", default=[t / 1000 for t in DEFAULT_TEMPOS],
                   help="Tempo factors (x0.8 slower .. x1.25 faster); 1.0 is the re-encoded round trip")
    b.add_argument("--splits", nargs="+", default=["train"])
    b.add_argument("--audio-source", choices=["auto", "original", "decode", "parquet"], default="auto")
    b.add_argument("--parquet-repo", help="--audio-source parquet: the HF dataset the cache was prepared from")
    b.add_argument("--parquet-dir", help="--audio-source parquet: download directory (one shard at a time)")
    b.add_argument("--codec", help="DACVAE checkpoint (default: the cache's)")
    b.add_argument("--device", default="cuda")
    b.add_argument("--shard-index", type=int, default=0)
    b.add_argument("--num-shards", type=int, default=1)
    b.add_argument("--quiet", action="store_true")
    m = commands.add_parser("merge", help="Combine the partitions of a sharded build")
    m.add_argument("--output", required=True)
    m.add_argument("--cache", help="Check that every row of the built splits has every tempo")
    args = parser.parse_args(argv)
    result = build(args) if args.command == "build" else merge_tempo_parts(args.output, args.cache)
    print(json.dumps({k: v for k, v in result.items() if k != "cache"}, indent=2))
    return result


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
