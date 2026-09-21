"""Upper-bound DataLoader throughput of the current LatentDataset/collate path (page-cached tiny corpus)."""

import argparse
import json
import time

from torch.utils.data import DataLoader

from dacvae_tts.data import BucketBatchSampler, LatentDataset, collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = LatentDataset(args.cache, "train", 42)
    results = []
    for workers in (0, 1, 2, 4):
        sampler = BucketBatchSampler(data.costs, 64, 0, 1, 42)
        options = (
            dict(
                num_workers=workers,
                persistent_workers=True,
                prefetch_factor=4,
                multiprocessing_context="spawn",
            )
            if workers
            else dict(num_workers=0)
        )
        loader = DataLoader(data, batch_sampler=sampler, collate_fn=collate, **options)
        pairs, frames, started, batches = 0, 0, None, 0
        for epoch in range(3):
            sampler.epoch = epoch
            for batch in loader:
                if started is None:  # skip worker start-up
                    started = time.perf_counter()
                    continue
                pairs += batch["latents"].size(0)
                frames += int(batch["valid"].sum())
                batches += 1
        seconds = time.perf_counter() - started
        row = {
            "workers": workers,
            "pairs_per_second": pairs / seconds,
            "valid_frames_per_second": frames / seconds,
            "batches": batches,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        del loader
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
