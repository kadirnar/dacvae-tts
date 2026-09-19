"""Packed float16 latent storage, SQLite metadata, and deterministic DDP batching."""

import hashlib
import json
import os
import random
import sqlite3
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from .contracts import normalization_stats
from .text import tokenize, tokenize_bytes

SCHEMA = """
CREATE TABLE samples (
 id INTEGER PRIMARY KEY, uid TEXT UNIQUE NOT NULL, speaker TEXT NOT NULL,
 text TEXT NOT NULL, audio TEXT NOT NULL, shard TEXT NOT NULL,
 offset INTEGER NOT NULL, frames INTEGER NOT NULL, split TEXT NOT NULL,
 samples INTEGER NOT NULL, digest TEXT UNIQUE NOT NULL
);
CREATE INDEX speaker_split ON samples(split, speaker, id);
CREATE TABLE provenance (uid TEXT PRIMARY KEY, original_text TEXT, normalized_text TEXT,
 session TEXT, source_recording TEXT, start_seconds REAL, end_seconds REAL);
CREATE TABLE text_tokens (uid TEXT PRIMARY KEY, utf8 BLOB NOT NULL);
"""


def jsonl(path):
    with open(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc


def speaker_split(speaker, seed=42, val_fraction=0.01, test_fraction=0.01):
    value = int.from_bytes(hashlib.sha256(f"{seed}:{speaker}".encode()).digest()[:8], "big") / 2**64
    return "test" if value < test_fraction else "val" if value < test_fraction + val_fraction else "train"


class ShardWriter:
    def __init__(self, directory, channels, shard_bytes=512 * 1024**2):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.channels, self.shard_bytes = channels, shard_bytes
        self.file = None
        self.index = 0

    def write(self, latents):
        data = np.asarray(latents, dtype="<f2")
        if data.ndim != 2 or data.shape[1] != self.channels or not np.isfinite(data).all():
            raise ValueError("Invalid latent array")
        if self.file is None or self.file.tell() + data.nbytes > self.shard_bytes:
            if self.file:
                self.file.close()
            self.path = self.directory / f"latents-{self.index:06d}.bin"
            self.file = open(self.path, "xb")
            self.index += 1
        offset = self.file.tell() // (2 * self.channels)
        self.file.write(data.tobytes())
        return str(self.path), offset

    def close(self):
        if self.file:
            self.file.close()


def load_stats(directory):
    obj = torch.load(Path(directory) / "stats.pt", map_location="cpu", weights_only=True)
    if obj["count"] < 1:
        raise ValueError("No training frames in cache")
    normalization_stats(obj["mean"], obj["std"])
    return obj


class LatentDataset(Dataset):
    def __init__(self, directory, split="train", seed=42):
        self.directory = Path(directory).resolve()
        self.db_path = self.directory / "index.sqlite"
        self.meta = json.loads((self.directory / "metadata.json").read_text())
        stats = load_stats(directory)
        self.mean, self.std = stats["mean"], stats["std"]
        self.channels = len(self.mean)
        normalization_stats(self.mean, self.std, self.meta["latent_dim"])
        self.seed = seed
        self.epoch = 0
        # Only compact integers live in RAM, never 4M transcripts or latent tensors.
        with sqlite3.connect(self.db_path) as db:
            total = db.execute("SELECT count(*) FROM samples WHERE split=?", (split,)).fetchone()[0]
            self.ids = np.empty(total, dtype=np.int64)
            self.lengths = np.empty(total, dtype=np.int32)
            self.group_start = np.empty(total, dtype=np.int64)
            self.group_end = np.empty(total, dtype=np.int64)
            cursor = db.execute(
                "SELECT id, frames, speaker FROM samples WHERE split=? ORDER BY speaker,id", (split,)
            )
            start, previous, count = 0, None, 0
            for i, (rowid, frames, speaker) in enumerate(cursor):
                if speaker != previous and previous is not None:
                    self.group_start[start:i], self.group_end[start:i] = start, i
                    start = i
                self.ids[i], self.lengths[i] = rowid, frames
                previous, count = speaker, i + 1
            if count:
                self.group_start[start:count], self.group_end[start:count] = start, count
        if total == 0:
            raise ValueError(f"Empty {split} split; need enough speakers or explicit split assignments")
        if np.any(self.group_end - self.group_start < 2):
            raise ValueError(
                f"{split}: every speaker needs at least two utterances; run merge to filter singletons"
            )
        self.max_ref_lengths = np.empty(total, dtype=np.int32)
        for start in np.flatnonzero(self.group_start == np.arange(total)):
            end = self.group_end[start]
            self.max_ref_lengths[start:end] = self.lengths[start:end].max()
        self.costs = self.lengths + self.max_ref_lengths
        self._pid, self._db, self._maps = None, None, OrderedDict()

    def __len__(self):
        return len(self.ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        # SQLite handles and mmap objects are reopened in each spawned worker.
        state.update(_pid=None, _db=None, _maps=OrderedDict())
        return state

    def _connection(self):
        if self._pid != os.getpid():
            self._db = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            self._pid, self._maps = os.getpid(), OrderedDict()
            self._has_text_tokens = bool(
                self._db.execute("SELECT 1 FROM sqlite_master WHERE name='text_tokens'").fetchone()
            )
        return self._db

    def row(self, index):
        connection = self._connection()
        query = (
            "SELECT s.uid,s.text,s.shard,s.offset,s.frames,s.speaker,t.utf8 "
            "FROM samples s LEFT JOIN text_tokens t ON t.uid=s.uid WHERE s.id=?"
            if self._has_text_tokens
            else "SELECT uid,text,shard,offset,frames,speaker,NULL FROM samples WHERE id=?"
        )
        row = connection.execute(query, (int(self.ids[index]),)).fetchone()
        uid, text, shard, offset, frames, speaker, text_bytes = row
        if shard not in self._maps:
            self._maps[shard] = np.memmap(shard, dtype="<f2", mode="r").reshape(-1, self.channels)
            if len(self._maps) > 32:
                self._maps.popitem(last=False)
        self._maps.move_to_end(shard)
        z = torch.from_numpy(np.array(self._maps[shard][offset : offset + frames], dtype=np.float32))
        if offset < 0 or frames < 1 or z.shape != (frames, self.channels) or not torch.isfinite(z).all():
            raise ValueError(f"Corrupt/truncated latent record: {uid}")
        return {
            "uid": uid,
            "text": text,
            "text_bytes": text_bytes,
            "speaker": speaker,
            "latents": (z - self.mean) / self.std,
        }

    def __getitem__(self, index):
        # Epoch travels through the sampler index so persistent workers see it.
        epoch, index = index if isinstance(index, tuple) else (self.epoch, index)
        rng = random.Random(self.seed + epoch * len(self) + index)
        start, end = int(self.group_start[index]), int(self.group_end[index])
        ref_index = rng.randrange(start, end - 1)
        ref_index += ref_index >= index
        target, ref = self.row(index), self.row(ref_index)
        return {
            "target": target["latents"],
            "reference": ref["latents"],
            "text": target["text"],
            "reference_text": ref["text"],
            "text_bytes": target["text_bytes"],
            "reference_text_bytes": ref["text_bytes"],
            "uid": target["uid"],
            "reference_uid": ref["uid"],
            "speaker": target["speaker"],
            "text_normalization": self.meta.get("text_normalization", "unicode-v1"),
        }


def collate(items):
    if not items:
        raise ValueError("Cannot collate an empty batch")
    latents, prompts, masks, tokens, segments, lengths = [], [], [], [], [], []
    for item in items:
        ref, target = item["reference"], item["target"]
        if (
            ref.ndim != 2
            or target.ndim != 2
            or ref.size(1) != target.size(1)
            or min(len(ref), len(target)) < 1
        ):
            raise ValueError("Each item requires nonempty [Lref,C] reference and [Ltgt,C] target")
        if not torch.isfinite(ref).all() or not torch.isfinite(target).all():
            raise ValueError("Reference and target latents must be finite")
        z = torch.cat([ref, target])
        mask = torch.arange(len(z)) < len(ref)
        if item.get("text_bytes") is not None and item.get("reference_text_bytes") is not None:
            tok, seg = tokenize_bytes(item["reference_text_bytes"], item["text_bytes"])
        else:
            tok, seg = tokenize(
                item["reference_text"], item["text"], item.get("text_normalization", "unicode-v1")
            )
        latents.append(z)
        prompts.append(z * mask[:, None])
        masks.append(mask)
        tokens.append(tok)
        segments.append(seg)
        lengths.append(len(z))
    return {
        "latents": pad_sequence(latents, batch_first=True),
        "prompt": pad_sequence(prompts, batch_first=True),
        "prompt_mask": pad_sequence(masks, batch_first=True),
        "valid": torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None],
        "tokens": pad_sequence(tokens, batch_first=True),
        "segments": pad_sequence(segments, batch_first=True),
    }


class BucketBatchSampler(Sampler):
    """Equal steps per rank, deterministic shuffling and bounded padded-frame budgets.

    Form global batches, drop at most world_size-1 batches, then distribute them.
    Unlike independent per-rank token batching, this cannot deadlock DDP at epoch end.
    """

    def __init__(
        self,
        costs,
        batch_size,
        rank=0,
        world_size=1,
        seed=42,
        frame_budget=0,
        bucket_size=4096,
        speaker_counts=None,
        speaker_balance=0.0,
    ):
        self.costs = costs
        self.batch_size, self.rank, self.world_size = batch_size, rank, world_size
        self.seed, self.frame_budget, self.bucket_size = seed, frame_budget, bucket_size
        self.epoch = 0
        self.start_batch = 0
        self.weights = None
        if not 0 <= speaker_balance <= 1:
            raise ValueError("speaker_balance must lie in [0,1]")
        if speaker_balance:
            if speaker_counts is None or len(speaker_counts) != len(costs):
                raise ValueError("Speaker balancing needs a positive speaker count for every row")
            if np.any(speaker_counts <= 0):
                raise ValueError("Speaker counts must be positive")
            weights = np.asarray(speaker_counts, dtype=np.float64) ** (-speaker_balance)
            self.weights = weights / weights.sum()
        if frame_budget and int(max(costs)) > frame_budget:
            raise ValueError("frame-budget is smaller than a single prompt+target pair")

    def batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        # Balancing intentionally samples with replacement; alpha=0 retains the original no-replacement baseline.
        indices = (
            rng.permutation(len(self.costs))
            if self.weights is None
            else rng.choice(len(self.costs), len(self.costs), replace=True, p=self.weights)
        )
        batches = []
        for offset in range(0, len(indices), self.bucket_size):
            bucket = indices[offset : offset + self.bucket_size]
            bucket = bucket[np.argsort(self.costs[bucket], kind="stable")]
            current = []
            for idx in bucket:
                if current and (
                    len(current) >= self.batch_size
                    or (self.frame_budget and self.costs[idx] * (len(current) + 1) > self.frame_budget)
                ):
                    batches.append(current)
                    current = []
                current.append(int(idx))
            if current:
                batches.append(current)
        rng.shuffle(batches)
        usable = len(batches) // self.world_size * self.world_size
        if not usable:
            raise ValueError("Too few batches for this world size")
        return batches[self.rank : usable : self.world_size]

    def __iter__(self):
        for batch in self.batches()[self.start_batch :]:
            yield [(self.epoch, idx) for idx in batch]

    def __len__(self):
        return len(self.batches()) - self.start_batch


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def save_stats(path, count, sums, squares):
    if not count:
        mean, std = torch.zeros_like(sums), torch.ones_like(sums)
    else:
        mean = sums / count
        std = (squares / count - mean.square()).clamp_min(1e-6).sqrt()
    torch.save(
        {"count": count, "sum": sums, "squares": squares, "mean": mean.float(), "std": std.float()}, path
    )
