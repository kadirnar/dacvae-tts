import json
import sqlite3

import pytest
import torch

from dacvae_tts.data import SCHEMA, ShardWriter, save_stats


@pytest.fixture(autouse=True)
def deterministic_threads():
    torch.set_num_threads(1)
    torch.manual_seed(7)


@pytest.fixture
def cache(tmp_path):
    path = tmp_path / "cache"
    path.mkdir()
    writer = ShardWriter(path, 4, shard_bytes=400)
    sums, squares, count = torch.zeros(4).double(), torch.zeros(4).double(), 0
    with sqlite3.connect(path / "index.sqlite") as db:
        db.executescript(SCHEMA)
        for split in ("train", "val", "test"):
            for speaker in range(4):
                for utterance in range(3):
                    z = torch.randn(7 + utterance * 2, 4)
                    shard, offset = writer.write(z.numpy())
                    uid = f"{split}-{speaker}-{utterance}"
                    db.execute(
                        "INSERT INTO samples VALUES(NULL,?,?,?,?,?,?,?,?,?,?)",
                        (
                            uid,
                            f"{split}-{speaker}",
                            f"Sentence number {utterance}.",
                            "audio.wav",
                            shard,
                            offset,
                            len(z),
                            split,
                            len(z) * 512,
                            uid,
                        ),
                    )
                    if split == "train":
                        sums += z.double().sum(0)
                        squares += z.double().square().sum(0)
                        count += len(z)
    writer.close()
    save_stats(path / "stats.pt", count, sums, squares)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "latent_dim": 4,
                "sample_rate": 24000,
                "hop_length": 512,
                "posterior": "mean",
                "checkpoint": "test-codec",
                "merged": True,
                "complete": True,
                "seed": 42,
            }
        )
    )
    return path
