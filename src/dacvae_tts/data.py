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
from .teacher import TeacherInputs, collate_teacher, pair_teacher
from .text import assemble, char_ctc_targets, decode_ids, join_ids, tokenize, tokenize_bytes

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
CREATE TABLE token_ids (uid TEXT PRIMARY KEY, ids BLOB NOT NULL);
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


def load_silence(directory, meta):
    """Raw (unnormalized) encoded-silence frame [C] written by scripts/silence_latent.py."""
    path = Path(directory) / "silence.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; tail_silence_prob and prompt_cut: quiet need the encoded-silence latent "
            f"(python scripts/silence_latent.py --cache {directory})"
        )
    obj = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("checkpoint", "latent_dim", "sample_rate", "hop_length"):
        if key in obj.get("codec", {}) and key in meta and obj["codec"][key] != meta[key]:
            raise ValueError(f"{path} was encoded with another codec ({key}); recreate it for this cache")
    raw = obj["raw"].float()
    if raw.shape != (meta["latent_dim"],) or not torch.isfinite(raw).all():
        raise ValueError(f"{path} must hold one finite [{meta['latent_dim']}] frame")
    return raw


# Independent random streams per (seed, epoch, index): toggling one pair option never moves the draws
# of another, and every stream stays clear of the original seed range seed + epoch * rows + index.
CROSS_STREAM, TAIL_STREAM, LONG_STREAM = 1 << 48, 2 << 48, 3 << 48
QUIET_CUT_SECONDS = 0.3


class LatentDataset(Dataset):
    """`cross` pairs a target with another utterance of the same speaker; `within` cuts the voice
    prompt from the start of the target utterance itself, so no speaker labels are needed."""

    def __init__(
        self,
        directory,
        split="train",
        seed=42,
        pairing="cross",
        layout="segments",
        prompt_fraction=(0.1, 0.5),
        prompt_dropout=0.0,
        teacher_features=None,
        speaker_embeddings=None,
        **pair_options,
    ):
        if pairing not in {"cross", "within"} or layout not in {"segments", "joined"}:
            raise ValueError("pairing must be cross or within; layout must be segments or joined")
        if pairing == "within" and layout != "joined":
            raise ValueError("Within-utterance prompts have no transcript boundary; use the joined layout")
        if not 0 <= prompt_fraction[0] <= prompt_fraction[1] < 1 or not 0 <= prompt_dropout <= 1:
            raise ValueError("Invalid prompt fraction range or prompt dropout")
        self.pairing, self.layout = pairing, layout
        self.prompt_fraction, self.prompt_dropout = tuple(prompt_fraction), prompt_dropout
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
        if pairing == "within":
            self.costs = self.lengths.copy()  # the prompt is part of the utterance: exact cost
        else:
            if np.any(self.group_end - self.group_start < 2):
                raise ValueError(
                    f"{split}: every speaker needs at least two utterances; run merge to filter singletons"
                )
            self.max_ref_lengths = np.empty(total, dtype=np.int32)
            for start in np.flatnonzero(self.group_start == np.arange(total)):
                end = self.group_end[start]
                self.max_ref_lengths[start:end] = self.lengths[start:end].max()
            self.costs = self.lengths + self.max_ref_lengths
        self.configure_pairs(**pair_options)
        self._pid, self._db, self._maps = None, None, OrderedDict()
        # Precomputed teacher targets (teacher.py) for the alignment losses; None leaves items unchanged.
        self.teacher = None
        if teacher_features or speaker_embeddings:
            rate = self.meta["sample_rate"] / self.meta["hop_length"]
            self.teacher = TeacherInputs(self.db_path, split, rate, teacher_features, speaker_embeddings)

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
            tables = {
                name for (name,) in self._db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            self._has_text_tokens = "text_tokens" in tables
            self._has_token_ids = "token_ids" in tables
        return self._db

    def row(self, index):
        connection = self._connection()
        # Preferred: token ids written at preparation time. Older caches fall back to cached
        # normalized bytes, and the oldest to the raw transcript.
        utf8 = "t.utf8" if self._has_text_tokens else "NULL"
        ids = "k.ids" if self._has_token_ids else "NULL"
        joins = (" LEFT JOIN text_tokens t ON t.uid=s.uid" if self._has_text_tokens else "") + (
            " LEFT JOIN token_ids k ON k.uid=s.uid" if self._has_token_ids else ""
        )
        query = f"SELECT s.uid,s.text,s.shard,s.offset,s.frames,s.speaker,{utf8},{ids} FROM samples s{joins} WHERE s.id=?"
        row = connection.execute(query, (int(self.ids[index]),)).fetchone()
        uid, text, shard, offset, frames, speaker, text_bytes, ids = row
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
            "token_ids": decode_ids(ids),
            "speaker": speaker,
            "latents": (z - self.mean) / self.std,
            **(self.teacher.lookup(uid, frames) if self.teacher is not None else {}),
        }

    def __getitem__(self, index):
        # Epoch travels through the sampler index so persistent workers see it.
        epoch, index = index if isinstance(index, tuple) else (self.epoch, index)
        rng = random.Random(self.seed + epoch * len(self) + index)
        if self.pairing == "within":
            references = self.cross_plan(epoch, index)
            if references:
                return self.finish_item(self.cross_prompt_item(index, references), epoch, index)
            row = self.row(index)
            frames = len(row["latents"])
            # Prompt dropout trains prompt-free synthesis of the whole utterance from its text.
            cut = (
                0
                if rng.random() < self.prompt_dropout
                else round(frames * rng.uniform(*self.prompt_range(epoch, index)))
            )
            cut = self.quiet_cut(row["latents"], cut)
            cut = max(min(cut, frames - 1), 0)  # always keep at least one target frame
            item = {
                "target": row["latents"][cut:],
                "reference": row["latents"][:cut],
                "text": row["text"],
                "reference_text": "",
                "text_bytes": row["text_bytes"],
                "reference_text_bytes": b"" if row["text_bytes"] is not None else None,
                "token_ids": row["token_ids"],
                "reference_token_ids": None,
                "uid": row["uid"],
                "reference_uid": row["uid"],
                "speaker": row["speaker"],
                "text_normalization": self.meta.get("text_normalization", "unicode-v1"),
                "layout": self.layout,
                **pair_teacher(row, row, cut),
            }
            return self.finish_item(item, epoch, index)
        start, end = int(self.group_start[index]), int(self.group_end[index])
        ref_index = rng.randrange(start, end - 1)
        ref_index += ref_index >= index
        target, ref = self.row(index), self.row(ref_index)
        item = {
            "target": target["latents"],
            "reference": ref["latents"],
            "text": target["text"],
            "reference_text": ref["text"],
            "text_bytes": target["text_bytes"],
            "reference_text_bytes": ref["text_bytes"],
            "token_ids": target["token_ids"],
            "reference_token_ids": ref["token_ids"],
            "uid": target["uid"],
            "reference_uid": ref["uid"],
            "speaker": target["speaker"],
            "text_normalization": self.meta.get("text_normalization", "unicode-v1"),
            "layout": self.layout,
            **pair_teacher(ref, target),
        }
        return self.finish_item(item, epoch, index)

    # Training-pair options (issue #11). Each one is a no-op when off: no extra random draws, no new
    # item keys, unchanged costs, so the default data stream is bit-identical to the one before.

    def configure_pairs(
        self,
        cross_prompt_prob=0.0,
        cross_prompt_max_utterances=3,
        cross_prompt_max_seconds=12.0,
        long_prompt_prob=0.0,
        prompt_fraction_long_max=0.85,
        tail_silence_prob=0.0,
        tail_silence_max_seconds=0.8,
        prompt_cut="random",
        ctc_targets="bytes",
    ):
        """Validate the options and widen `costs` to an upper bound of every prompt+target they can form.

        `costs` stays the static bound the sampler checks against the frame budget. Cross prompts change a
        row's length from epoch to epoch, so `epoch_costs` gives the exact per-epoch lengths (never above
        `costs`); batching by the bound would reserve up to 12 s for every row that keeps a within cut.
        """
        if not all(0 <= p <= 1 for p in (cross_prompt_prob, long_prompt_prob, tail_silence_prob)):
            raise ValueError("Pair option probabilities must lie in [0,1]")
        if cross_prompt_max_utterances < 1 or cross_prompt_max_seconds <= 0 or tail_silence_max_seconds <= 0:
            raise ValueError("Cross-prompt and tail-silence limits must be positive")
        if long_prompt_prob and not self.prompt_fraction[1] <= prompt_fraction_long_max < 1:
            raise ValueError("Need prompt_fraction_max <= prompt_fraction_long_max < 1")
        if prompt_cut not in {"random", "quiet"} or ctc_targets not in {"bytes", "chars"}:
            raise ValueError("prompt_cut must be random or quiet; ctc_targets bytes or chars")
        if self.pairing != "within" and (cross_prompt_prob or long_prompt_prob or prompt_cut != "random"):
            raise ValueError("Cross prompts, long prompts and quiet cuts act on within pairing")
        frame_rate = self.meta["sample_rate"] / self.meta["hop_length"]
        self.cross_prompt_prob, self.cross_prompt_utterances = cross_prompt_prob, cross_prompt_max_utterances
        self.cross_prompt_frames = int(cross_prompt_max_seconds * frame_rate)  # floor: never above the limit
        self.long_prompt_prob, self.prompt_fraction_long_max = long_prompt_prob, prompt_fraction_long_max
        self.tail_silence_prob = tail_silence_prob
        self.tail_silence_frames = (
            max(round(tail_silence_max_seconds * frame_rate), 1) if tail_silence_prob else 0
        )
        self.prompt_cut, self.quiet_window = prompt_cut, round(QUIET_CUT_SECONDS * frame_rate)
        self.ctc_targets = ctc_targets
        self.silence = None
        if tail_silence_prob or prompt_cut == "quiet":
            self.silence = (load_silence(self.directory, self.meta) - self.mean) / self.std
        if self.tail_silence_frames:
            self.costs = self.costs + self.tail_silence_frames
        if cross_prompt_prob:
            several = self.group_end - self.group_start >= 2
            self.costs = self.costs + np.where(several, self.cross_prompt_frames, 0).astype(self.costs.dtype)

    def cross_plan(self, epoch, index):
        """Rows forming this row's prompt in `epoch` (other utterances of its speaker), or () for a cut.

        A pure function of (seed, epoch, index) and the row lengths, so `epoch_costs` in the sampler's
        process and `__getitem__` in the loader workers agree. Prompt dropout is decided first with the
        same draw `__getitem__` uses, which keeps its rate: cross prompts replace cuts of prompted rows only.
        References are whole utterances (a cropped one would no longer match its transcript); those longer
        than the remaining seconds are skipped, and a row whose draws all exceed them keeps its within cut.
        """
        if not self.cross_prompt_prob:
            return ()
        start, end = int(self.group_start[index]), int(self.group_end[index])
        if end - start < 2:
            return ()
        base = self.seed + epoch * len(self) + index
        if random.Random(base).random() < self.prompt_dropout:
            return ()
        rng = random.Random(base + CROSS_STREAM)
        if rng.random() >= self.cross_prompt_prob:
            return ()
        count = rng.randint(1, min(self.cross_prompt_utterances, end - start - 1))
        chosen, total = [], 0
        for other in rng.sample(range(start, end - 1), count):
            other += other >= index
            if total + int(self.lengths[other]) <= self.cross_prompt_frames:
                chosen.append(other)
                total += int(self.lengths[other])
        return tuple(chosen)

    def epoch_costs(self, epoch):
        """Exact prompt+target frames of every row in `epoch` (plus the tail-silence maximum)."""
        costs = self.lengths + self.tail_silence_frames
        if self.cross_prompt_prob:
            for index in np.flatnonzero(self.group_end - self.group_start >= 2):
                references = self.cross_plan(epoch, int(index))
                if references:
                    costs[index] += int(self.lengths[list(references)].sum())
        return costs

    def cross_prompt_item(self, index, references):
        """Prompt = the references' latents back to back, transcript = their texts + target text, target =
        the whole utterance; the joined layout reads it as one stream like a within item."""
        target, refs = self.row(index), [self.row(r) for r in references]
        ids = [r["token_ids"] for r in refs]
        utf8 = [r["text_bytes"] for r in refs]
        return {
            "target": target["latents"],
            "reference": torch.cat([r["latents"] for r in refs]),
            "text": target["text"],
            "reference_text": " ".join(r["text"] for r in refs),
            "text_bytes": target["text_bytes"],
            "reference_text_bytes": None if None in utf8 else b" ".join(utf8),
            "token_ids": target["token_ids"],
            "reference_token_ids": None if any(i is None for i in ids) else join_ids(ids),
            "uid": target["uid"],
            "reference_uid": "|".join(r["uid"] for r in refs),
            "speaker": target["speaker"],
            "text_normalization": self.meta.get("text_normalization", "unicode-v1"),
            "layout": self.layout,
        }

    def prompt_range(self, epoch, index):
        """Cut-fraction range: the long one ([max, long max], short targets) with long_prompt_prob."""
        if self.long_prompt_prob:
            draw = random.Random(self.seed + epoch * len(self) + index + LONG_STREAM).random()
            if draw < self.long_prompt_prob:
                return self.prompt_fraction[1], self.prompt_fraction_long_max
        return self.prompt_fraction

    def quiet_cut(self, latents, cut):
        """Move a nonzero cut so the prompt ends on the frame closest (L2) to silence within the window.

        Ties go to the frame nearest the sampled cut; at least one prompt and one target frame remain.
        """
        if self.prompt_cut != "quiet" or cut < 1:
            return cut
        low, high = max(cut - 1 - self.quiet_window, 0), min(cut - 1 + self.quiet_window, len(latents) - 2)
        if low > high:
            return cut
        distance = (latents[low : high + 1] - self.silence).square().sum(-1).tolist()
        last = min(range(high - low + 1), key=lambda i: (distance[i], abs(low + i - cut + 1)))
        return low + last + 1

    def finish_item(self, item, epoch, index):
        """Tail silence and the CTC label flag; the item itself when both are off."""
        if self.ctc_targets == "chars":
            item = {**item, "ctc_targets": "chars"}
        if not self.tail_silence_prob:
            return item
        rng = random.Random(self.seed + epoch * len(self) + index + TAIL_STREAM)
        frames = rng.randint(1, self.tail_silence_frames) if rng.random() < self.tail_silence_prob else 0
        pad = self.silence.expand(frames, -1)
        return {**item, "target": torch.cat([item["target"], pad]), "tail_silence": frames}


def collate(items):
    if not items:
        raise ValueError("Cannot collate an empty batch")
    latents, prompts, masks, tokens, segments, lengths = [], [], [], [], [], []
    for item in items:
        ref, target = item["reference"], item["target"]
        layout = item.get("layout", "segments")
        # A joined-layout item may come without a prompt; the segment layout always has a reference.
        if (
            ref.ndim != 2
            or target.ndim != 2
            or ref.size(1) != target.size(1)
            or len(target) < 1
            or (len(ref) < 1 and layout != "joined")
        ):
            raise ValueError("Each item requires nonempty [Lref,C] reference and [Ltgt,C] target")
        if not torch.isfinite(ref).all() or not torch.isfinite(target).all():
            raise ValueError("Reference and target latents must be finite")
        z = torch.cat([ref, target])
        mask = torch.arange(len(z)) < len(ref)
        if item.get("token_ids") is not None and (
            item.get("reference_token_ids") is not None or item.get("reference_text", "") == ""
        ):
            tok, seg = assemble(item.get("reference_token_ids"), item["token_ids"], layout)
        elif item.get("text_bytes") is not None and item.get("reference_text_bytes") is not None:
            tok, seg = tokenize_bytes(item["reference_text_bytes"], item["text_bytes"], layout)
        else:
            tok, seg = tokenize(
                item["reference_text"], item["text"], item.get("text_normalization", "unicode-v1"), layout
            )
        latents.append(z)
        prompts.append(z * mask[:, None])
        masks.append(mask)
        tokens.append(tok)
        segments.append(seg)
        lengths.append(len(z))
    batch = {
        "latents": pad_sequence(latents, batch_first=True),
        "prompt": pad_sequence(prompts, batch_first=True),
        "prompt_mask": pad_sequence(masks, batch_first=True),
        "valid": torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None],
        "tokens": pad_sequence(tokens, batch_first=True),
        "segments": pad_sequence(segments, batch_first=True),
        **collate_teacher(items),
    }
    labels = {item.get("ctc_targets", "bytes") for item in items}
    if labels != {"bytes"}:
        # Character CTC targets are built here, in the loader workers, from the exact model tokens.
        if labels != {"chars"}:
            raise ValueError("A batch cannot mix byte and character CTC targets")
        batch["ctc_targets"], batch["ctc_target_lengths"] = char_ctc_targets(tokens)
    return batch


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
        epoch_costs=None,
    ):
        # `epoch_costs(epoch)`: exact per-epoch costs (cross prompts) bounded by the static `costs`.
        self.costs, self.epoch_costs, self._epoch_cache = costs, epoch_costs, None
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
        costs = self.current_costs()
        batches = []
        for offset in range(0, len(indices), self.bucket_size):
            bucket = indices[offset : offset + self.bucket_size]
            bucket = bucket[np.argsort(costs[bucket], kind="stable")]
            current = []
            for idx in bucket:
                if current and (
                    len(current) >= self.batch_size
                    or (self.frame_budget and costs[idx] * (len(current) + 1) > self.frame_budget)
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

    def current_costs(self):
        if self.epoch_costs is None:
            return self.costs
        if self._epoch_cache is None or self._epoch_cache[0] != self.epoch:
            costs = self.epoch_costs(self.epoch)
            if len(costs) != len(self.costs) or np.any(costs > self.costs):
                raise ValueError("Per-epoch costs must stay within the static bound checked against budgets")
            self._epoch_cache = (self.epoch, costs)
        return self._epoch_cache[1]

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
