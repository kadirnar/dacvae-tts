"""Stream a sharded Hugging Face Parquet dataset into latent-cache partitions, one shard at a time.

Each shard is downloaded, encoded into `OUTPUT/parts/part-XXXXX` with the normal `prepare` code path and
then deleted, so the raw corpus never has to fit on disk. The frozen codec is loaded once. Finished
partitions are skipped, which makes the job restartable; merge the partitions afterwards:

  HF_TOKEN=... python scripts/prepare_hf_shards.py --repo ORG/DATASET --output data/corpus --shards 0-99 \
      --speaker-column speaker --loudness -16 --max-seconds 20
  dacvae-tts merge --inputs data/corpus/parts/part-* --output data/corpus/merged --keep-singletons
"""

import argparse
import json
import multiprocessing
import os
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import hf_hub_download


def encode_shard(options):
    """Runs in a fresh process per shard: audio decoding leaked ~100 MB per shard when one
    long-lived process handled the whole corpus, and a spawned process returns it all."""
    import dacvae_tts.prepare as prepare_module

    prepare_module.prepare(options)


def shard_list(spec):
    result = []
    for part in spec.split(","):
        first, _, last = part.partition("-")
        result.extend(range(int(first), int(last or first) + 1))
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pattern", default="data/train-{index:05d}-of-{total:05d}.parquet")
    parser.add_argument("--total", type=int, required=True, help="Number of shards in the repository")
    parser.add_argument("--shards", required=True, help="For example 0-99 or 0-9,50,60-69")
    parser.add_argument("--every", type=int, default=1, help="Take every N-th shard of the list ...")
    parser.add_argument("--offset", type=int, default=0, help="... starting at this position (one GPU each)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--speaker-column", default="speaker_id")
    parser.add_argument("--quality-column")
    parser.add_argument("--min-quality", type=float)
    parser.add_argument("--reject-digits", action="store_true")
    parser.add_argument("--loudness", type=float)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.output).resolve()
    (root / "parts").mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    context = multiprocessing.get_context("spawn")

    def fetch(index):
        folder = root / "raw" / f"shard-{index:05d}"
        name = args.pattern.format(index=index, total=args.total)
        for attempt in range(6):
            try:
                hf_hub_download(args.repo, name, repo_type="dataset", token=token, local_dir=folder)
                return folder
            except Exception as error:  # network hiccups are expected over hours of downloading
                if attempt == 5:
                    raise
                print(f"download retry {attempt + 1} for shard {index}: {error}", flush=True)
                time.sleep(10 * (attempt + 1))

    if not 0 <= args.offset < args.every:
        raise ValueError("--offset must be smaller than --every")
    pending = [
        i
        for position, i in enumerate(shard_list(args.shards))
        if position % args.every == args.offset
        and not (root / "parts" / f"part-{i:05d}" / "metadata.json").exists()
    ]
    print(f"{len(pending)} shards to process", flush=True)
    downloads = {}

    def prefetch(index):
        downloads[index] = SimpleNamespace(folder=None, error=None)
        try:
            downloads[index].folder = fetch(index)
        except Exception as error:
            downloads[index].error = error

    threads = {}
    for position, index in enumerate(pending):
        for ahead in pending[position : position + 2]:  # keep one shard downloading while one encodes
            if ahead not in threads:
                threads[ahead] = threading.Thread(target=prefetch, args=(ahead,), daemon=True)
                threads[ahead].start()
        threads[index].join()
        if downloads[index].error is not None:
            raise downloads[index].error
        folder = downloads[index].folder
        part = root / "parts" / f"part-{index:05d}"
        if part.exists():
            shutil.rmtree(part)  # an interrupted partition has no metadata.json and cannot be resumed
        started = time.time()
        options = SimpleNamespace(
            manifest=str(folder),
            output=str(part),
            codec="facebook/dacvae-watermarked",
            device=args.device,
            text_column=args.text_column,
            audio_column=args.audio_column,
            speaker_column=args.speaker_column,
            quality_column=args.quality_column,
            min_quality=args.min_quality,
            reject_digits=args.reject_digits,
            loudness=args.loudness,
            text_normalization="unicode-v1",
            min_seconds=args.min_seconds,
            max_seconds=args.max_seconds,
            workers=args.workers,
            worker_backend="thread",
            worker_threads=1,
            prefetch=4 * args.workers,
            batch_size=4,  # batching does not speed this encoder up; keep memory low
            bucket_size=1024,
            batch_seconds=48.0,
            precision="fp32",
            no_fold_weight_norm=False,
            shard_index=0,
            num_shards=1,
            seed=args.seed,
        )
        worker = context.Process(target=encode_shard, args=(options,))
        worker.start()
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(f"shard {index} failed with exit code {worker.exitcode}")
        shutil.rmtree(folder)
        meta = json.loads((part / "metadata.json").read_text())
        record = {
            "shard": index,
            "accepted": meta["accepted"],
            "rejected": meta["rejected"],
            "audio_hours": meta["preparation"]["audio_seconds"] / 3600,
            "wall_seconds": time.time() - started,
            "x_realtime": meta["preparation"]["audio_seconds_per_wall_second"],
        }
        print(json.dumps(record), flush=True)
        with open(root / f"progress-{args.offset}.jsonl", "a") as stream:
            stream.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
