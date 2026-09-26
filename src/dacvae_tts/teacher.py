"""Precomputed teacher targets for the training-only alignment losses (alignment.py).

A store is a sidecar directory next to a latent cache; the cache itself is never modified:

  metadata.json        kind ("frames" | "speaker"), teacher model/layer, pooling, dim, PCA, frame rate,
                       cache identity, audio sources used, complete/merged flags
  index.sqlite         teacher_features(uid, shard, offset, frames, dim) for frames,
                       speaker_embeddings(uid, dim, vector) for utterance embeddings (float32 LE BLOB)
  features-NNNNNN.bin  little-endian float16 [frames, dim] records; shard paths are relative to the
                       store root, so a cache and its stores can move between machines together
  pca.pt               optional {"mean": [Din], "components": [Din, dim], "explained": [dim]}
  part-III-of-NNN/     partitions of a sharded extraction, combined by `merge_parts`

Frame features are pooled from the SSL rate (50 Hz) to the latent rate (25 fps) and padded/trimmed
to the row's latent `frames`, so frame j of the features describes latent frame j. The dataset then
slices them exactly like the latents (`pair_teacher`) and `collate_teacher` lays them out in the same
[reference | target] order as `collate`.
"""

import io
import json
import math
import os
import random
import sqlite3
from collections import OrderedDict
from contextlib import closing
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS teacher_features (uid TEXT PRIMARY KEY, shard TEXT NOT NULL,
 offset INTEGER NOT NULL, frames INTEGER NOT NULL, dim INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS speaker_embeddings (uid TEXT PRIMARY KEY, dim INTEGER NOT NULL, vector BLOB NOT NULL);
"""
KINDS = {"frames": "teacher_features", "speaker": "speaker_embeddings"}
FORMAT_VERSION = 1


def resolve_store(path, cache):
    """Store path from the configuration: absolute, or relative to the cache directory."""
    path = Path(path)
    return (path if path.is_absolute() else Path(cache) / path).resolve()


def teacher_sources(train, cache):
    """LatentDataset keyword arguments for the enabled teacher terms; empty (bit-identical data) if off."""
    sources = {}
    if train.repa_weight > 0:
        sources["teacher_features"] = resolve_store(train.teacher_features, cache)
    if train.tla_weight > 0:
        sources["speaker_embeddings"] = resolve_store(train.speaker_embeddings, cache)
    return sources


class TeacherStore:
    """Read-only access by uid; SQLite handles and memory maps are reopened in each worker process."""

    def __init__(self, directory, kind):
        self.directory = Path(directory).resolve()
        if kind not in KINDS:
            raise ValueError("Teacher store kind must be frames or speaker")
        path = self.directory / "metadata.json"
        if not path.exists():
            raise ValueError(f"No teacher store at {self.directory}; run scripts/extract_teacher_features.py")
        self.meta = json.loads(path.read_text())
        if self.meta.get("kind") != kind:
            raise ValueError(f"{self.directory} holds {self.meta.get('kind')} targets, expected {kind}")
        if not self.meta.get("complete") or not self.meta.get("merged"):
            raise ValueError(f"Incomplete teacher store {self.directory}; finish extraction and run merge")
        self.kind, self.table, self.dim = kind, KINDS[kind], int(self.meta["dim"])
        self.db_path = self.directory / "index.sqlite"
        self._pid, self._db, self._maps = None, None, OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_pid=None, _db=None, _maps=OrderedDict())
        return state

    def _connection(self):
        if self._pid != os.getpid():
            self._db = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            self._pid, self._maps = os.getpid(), OrderedDict()
        return self._db

    def check(self, cache_db, split, frame_rate):
        """Fail at dataset construction, not hours into training, if any row of the split is missing."""
        if self.kind == "frames" and not math.isclose(self.meta["frame_rate"], frame_rate, rel_tol=1e-6):
            raise ValueError(
                f"Teacher frames at {self.meta['frame_rate']} fps, cache latents at {frame_rate}"
            )
        with closing(sqlite3.connect(f"file:{cache_db}?mode=ro", uri=True)) as db:
            db.execute("ATTACH DATABASE ? AS teacher", (f"file:{self.db_path}?mode=ro",))
            missing = db.execute(
                f"SELECT count(*), min(s.uid) FROM samples s LEFT JOIN teacher.{self.table} t ON t.uid=s.uid "
                "WHERE s.split=? AND t.uid IS NULL",
                (split,),
            ).fetchone()
        if missing[0]:
            raise ValueError(
                f"{missing[0]} {split} rows have no {self.kind} targets in {self.directory} (e.g. {missing[1]}); "
                "extract them with scripts/extract_teacher_features.py --splits"
            )

    def frames(self, uid, frames):
        """float16 [frames, dim] features of one row; the frame count must equal the latents'."""
        row = (
            self._connection()
            .execute("SELECT shard, offset, frames, dim FROM teacher_features WHERE uid=?", (uid,))
            .fetchone()
        )
        if row is None:
            raise KeyError(f"No teacher frames for {uid} in {self.directory}")
        shard, offset, count, dim = row
        if count != frames or dim != self.dim:
            raise ValueError(f"Teacher frames of {uid} are [{count},{dim}], latents have {frames} frames")
        if shard not in self._maps:
            self._maps[shard] = np.memmap(self.directory / shard, dtype="<f2", mode="r").reshape(-1, dim)
            if len(self._maps) > 32:
                self._maps.popitem(last=False)
        self._maps.move_to_end(shard)
        return torch.from_numpy(np.array(self._maps[shard][offset : offset + count]))

    def speaker(self, uid):
        row = (
            self._connection()
            .execute("SELECT dim, vector FROM speaker_embeddings WHERE uid=?", (uid,))
            .fetchone()
        )
        if row is None:
            raise KeyError(f"No speaker embedding for {uid} in {self.directory}")
        vector = torch.from_numpy(np.frombuffer(row[1], dtype="<f4").copy())
        if row[0] != self.dim or vector.shape != (self.dim,):
            raise ValueError(f"Corrupt speaker embedding for {uid}")
        return vector


class TeacherInputs:
    """The stores one LatentDataset reads: frame features and/or utterance speaker embeddings."""

    def __init__(self, cache_db, split, frame_rate, teacher_features=None, speaker_embeddings=None):
        self.features = TeacherStore(teacher_features, "frames") if teacher_features else None
        self.speakers = TeacherStore(speaker_embeddings, "speaker") if speaker_embeddings else None
        for store in (self.features, self.speakers):
            if store is not None:
                store.check(cache_db, split, frame_rate)

    def lookup(self, uid, frames):
        extra = {}
        if self.features is not None:
            extra["teacher"] = self.features.frames(uid, frames)
        if self.speakers is not None:
            extra["speaker_embedding"] = self.speakers.speaker(uid)
        return extra


def pair_teacher(reference, target, cut=None):
    """Teacher targets sliced exactly like the latents of one training item.

    `within` pairing (`cut` given): reference = frames [:cut], target = frames [cut:] of the same row;
    `cross`: the reference row's frames and the target row's frames. A list of reference rows (#11's
    cross-utterance prompts) concatenates their frames in order, like the prompt latents. The speaker
    embedding is always the target utterance's. Rows without teacher entries add nothing, keeping items
    unchanged.
    """
    extra = {}
    if "teacher" in target:
        whole = target["teacher"]
        if cut is not None:
            extra["teacher_reference"] = whole[:cut]
        elif isinstance(reference, (list, tuple)):
            extra["teacher_reference"] = torch.cat([row["teacher"] for row in reference])
        else:
            extra["teacher_reference"] = reference["teacher"]
        extra["teacher_target"] = whole[cut:] if cut is not None else whole
    if "speaker_embedding" in target:
        extra["speaker_embedding"] = target["speaker_embedding"]
    return extra


def collate_teacher(items):
    """Padded "teacher" [B,L,D] in collate's [reference | target] frame order, "speaker_embedding" [B,E].

    With #11's tail silence (items carry `tail_silence`), the appended frames have zero teacher rows, and prompts
    stretched by prompt tempo perturbation (`teacher_prompt_invalid`) have zero rows for all their frames; a
    boolean "teacher_valid" [B,L] marks the frames that have real teacher features, and speech-REPA only aligns
    those. Without either the batch is unchanged.
    """
    batch = {}
    for key in ("teacher_target", "speaker_embedding"):
        present = [key in item for item in items]
        if any(present) and not all(present):
            raise ValueError(f"Either every item or none carries {key}")
    if "teacher_target" in items[0]:
        features, real = [], []
        for item in items:
            reference, target = item["teacher_reference"], item["teacher_target"]
            if len(reference) != len(item["reference"]) or len(target) != len(item["target"]):
                raise ValueError("Teacher frames must align with the reference and target latent frames")
            features.append(torch.cat([reference, target]))
            frames = len(reference) + len(target)
            position = torch.arange(frames)
            first = len(reference) if item.get("teacher_prompt_invalid") else 0  # stretched prompt: no features
            real.append((position >= first) & (position < frames - item.get("tail_silence", 0)))
        batch["teacher"] = pad_sequence(features, batch_first=True)
        if any("tail_silence" in item or item.get("teacher_prompt_invalid") for item in items):
            batch["teacher_valid"] = pad_sequence(real, batch_first=True)
    if "speaker_embedding" in items[0]:
        batch["speaker_embedding"] = torch.stack([item["speaker_embedding"] for item in items]).float()
    return batch


# ----------------------------------------------------------------------------- extraction


def pool_frames(features, ratio, frames, tolerance=4):
    """SSL frames [T,D] at `ratio` x the latent rate -> [frames,D] aligned with the latent frames.

    A 400-sample HuBERT window with hop 320 at 16 kHz yields about 2 * frames - 1 frames for a
    25 fps latent sequence; missing tail frames repeat the last one. An integer ratio averages
    consecutive groups (50 Hz pairs -> 25 fps), any other ratio uses adaptive average pooling. A
    larger length mismatch means a wrong frame rate or audio and is an error, not a silent stretch.
    """
    if features.ndim != 2 or frames < 1 or ratio <= 0:
        raise ValueError("pool_frames needs [T,D] features, a positive ratio and frame count")
    needed = math.ceil(frames * ratio - 1e-6)
    if abs(len(features) - needed) > max(tolerance, 0.02 * needed):
        raise ValueError(f"Teacher produced {len(features)} frames, {needed} expected for {frames} latents")
    if len(features) < needed:
        features = torch.cat([features, features[-1:].expand(needed - len(features), -1)])
    features = features[:needed].float()
    if abs(ratio - round(ratio)) < 1e-6:
        return features.reshape(frames, round(ratio), -1).mean(1)
    return F.adaptive_avg_pool1d(features.T[None], frames)[0].T


def resample(waveform, rate, target_rate):
    if rate == target_rate:
        return np.asarray(waveform, dtype=np.float32)
    from scipy.signal import resample_poly

    factor = math.gcd(int(rate), int(target_rate))
    return resample_poly(waveform, target_rate // factor, rate // factor).astype(np.float32)


def fit_pca(sums, products, count, dim):
    """PCA from streamed first/second moments (FP64): mean [D], components [D,dim], explained ratio."""
    if count < 2 or not 1 <= dim <= len(sums):
        raise ValueError("PCA needs at least two frames and 1 <= dim <= feature width")
    mean = sums / count
    covariance = (products - count * torch.outer(mean, mean)) / (count - 1)
    values, vectors = torch.linalg.eigh(covariance)
    order = values.argsort(descending=True)[:dim]
    components = vectors[:, order]
    # Deterministic signs: the largest-magnitude loading of every component is positive.
    signs = components.gather(0, components.abs().argmax(0, keepdim=True)).sign()
    components = components * torch.where(signs == 0, 1.0, signs)
    explained = values[order].clamp_min(0) / values.clamp_min(0).sum().clamp_min(1e-12)
    return {"mean": mean.float(), "components": components.float(), "explained": explained.float()}


def apply_pca(features, pca):
    return (features.float() - pca["mean"]) @ pca["components"]


def _cache_db(cache):
    return sqlite3.connect(f"file:{Path(cache).resolve() / 'index.sqlite'}?mode=ro", uri=True)


def count_rows(cache, splits=("train",), shard_index=0, num_shards=1):
    with closing(_cache_db(cache)) as db:
        marks = ",".join("?" * len(splits))
        total = db.execute(
            f"SELECT count(*) FROM samples WHERE split IN ({marks})", tuple(splits)
        ).fetchone()[0]
    return len(range(shard_index, total, num_shards))


def cache_rows(cache, splits=("train",), shard_index=0, num_shards=1, positions=None):
    """Yield (uid, shard, offset, frames, samples, audio) of the selected rows in id order.

    Rows are partitioned round-robin (row i belongs to shard i % num_shards); `positions`, a set of
    indices into the unpartitioned selection, picks a sample instead. Nothing is held in RAM.
    """
    if not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard index")
    marks = ",".join("?" * len(splits))
    with closing(_cache_db(cache)) as db:
        query = f"SELECT uid, shard, offset, frames, samples, audio FROM samples WHERE split IN ({marks}) ORDER BY id"
        for index, row in enumerate(db.execute(query, tuple(splits))):
            if (index in positions) if positions is not None else index % num_shards == shard_index:
                yield row


class CacheAudio:
    """Waveform of a cache row: its original file when that still exists (`auto`/`original`), else the
    DACVAE decode of the stored latents (`auto`/`decode`). Rows prepared from embedded Parquet audio
    have no file, so decoding is the common path; the stored latents are the raw codec posterior means.

    `decoder(latents [frames,C] float32) -> waveform` at the cache sample rate, e.g. `Codec.decode`.
    """

    def __init__(self, cache, mode="auto", decoder=None):
        if mode not in {"auto", "original", "decode"}:
            raise ValueError("audio source must be auto, original or decode")
        self.cache = Path(cache).resolve()
        self.meta = json.loads((self.cache / "metadata.json").read_text())
        self.mode, self.decoder = mode, decoder
        self.sample_rate, self.channels = int(self.meta["sample_rate"]), int(self.meta["latent_dim"])
        self.counts = {"original": 0, "decoded": 0}
        self._maps = OrderedDict()

    def latents(self, shard, offset, frames):
        if shard not in self._maps:
            self._maps[shard] = np.memmap(shard, dtype="<f2", mode="r").reshape(-1, self.channels)
            if len(self._maps) > 8:
                self._maps.popitem(last=False)
        return torch.from_numpy(np.array(self._maps[shard][offset : offset + frames], dtype=np.float32))

    def __call__(self, row):
        uid, shard, offset, frames, samples, audio = row
        if self.mode != "decode" and audio and Path(audio).is_file():
            from .codec import read_audio

            self.counts["original"] += 1
            return read_audio(audio, self.sample_rate, self.meta.get("loudness_lufs")).numpy()
        if self.mode == "original":
            raise ValueError(f"Original audio of {uid} is not available: {audio}")
        if self.decoder is None:
            raise ValueError("Decoding latents needs a decoder (the cache's DACVAE checkpoint)")
        waveform = self.decoder(self.latents(shard, offset, frames))
        self.counts["decoded"] += 1
        return np.asarray(waveform, dtype=np.float32)[:samples]


class ParquetAudio:
    """Original audio of cache rows prepared from a sharded Hugging Face Parquet dataset, re-read from the Hub.

    A drop-in for `CacheAudio` (`sample_rate`, `counts`, `__call__(row)`) that needs no DACVAE decode: rows whose uid
    is "<prefix><file>:<row>" (scripts/prepare_hf_shards.py, stream_encode_score.sh) are loaded exactly as `prepare`
    loaded them (codec.read_audio: resampling to the cache rate and its loudness normalization), i.e. the waveform the
    latents were encoded from. Cache rows arrive in id order, which is partition (shard) order and row order within a
    shard, so the source keeps one shard on disk, decodes each 100-row group in a thread pool on first use,
    prefetches the next shard and deletes the previous one. Decoding one row at a time through DACVAE was ~8
    rows/s next to a training run; this reads the audio the codec saw, not its reconstruction.
    """

    def __init__(self, cache, repo, local_dir, prefix="data/", token=None, threads=8):
        from concurrent.futures import ThreadPoolExecutor

        self.cache = Path(cache).resolve()
        self.meta = json.loads((self.cache / "metadata.json").read_text())
        self.sample_rate, self.loudness = int(self.meta["sample_rate"]), self.meta.get("loudness_lufs")
        self.repo, self.local, self.prefix, self.token = repo, Path(local_dir), prefix, token
        self.local.mkdir(parents=True, exist_ok=True)
        self.counts = {"original": 0, "decoded": 0}
        self.pool = ThreadPoolExecutor(threads)
        self.current, self.file, self.offsets, self.group, self.cached = None, None, None, None, {}
        self.prefetched = {}

    def _download(self, name):
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(self.repo, name, repo_type="dataset", local_dir=self.local, token=self.token))

    def _open(self, name):
        import pyarrow.parquet as pq

        future = self.prefetched.pop(name, None)
        path = future.result() if future is not None else self._download(name)
        if self.current is not None and self.current != name:
            (self.local / self.current).unlink(missing_ok=True)
        self.current, self.file = name, pq.ParquetFile(path)
        sizes = [self.file.metadata.row_group(g).num_rows for g in range(self.file.num_row_groups)]
        self.offsets, self.group, self.cached = np.cumsum([0, *sizes]), None, {}
        index, total = self._shard_number(name)
        if index is not None and index + 1 < total:  # the next shard downloads while this one is read
            following = name.replace(f"{index:05d}-of", f"{index + 1:05d}-of")
            self.prefetched[following] = self.pool.submit(self._download, following)

    @staticmethod
    def _shard_number(name):
        import re

        match = re.search(r"(\d{5})-of-(\d{5})", name)
        return (int(match.group(1)), int(match.group(2))) if match else (None, None)

    def _load_group(self, group):
        from .codec import read_audio

        rows = self.file.read_row_group(group, columns=["audio"]).column("audio").to_pylist()
        read = lambda item: read_audio(io.BytesIO(item["bytes"]), self.sample_rate, self.loudness).numpy()  # noqa: E731
        self.group, self.cached = group, dict(enumerate(self.pool.map(read, rows), int(self.offsets[group])))

    def __call__(self, row):
        uid, _, _, _, samples, _ = row
        file, _, index = uid.rpartition(":")
        if not file.startswith(self.prefix) or not index.isdigit():
            raise ValueError(f"{uid} is not a '<prefix><file>:<row>' uid of a Parquet-prepared cache")
        name, index = file[len(self.prefix):], int(index)
        if name != self.current:
            self._open(name)
        group = int(np.searchsorted(self.offsets, index, side="right") - 1)
        if group != self.group:
            self._load_group(group)
        audio = self.cached[index]
        if samples and abs(len(audio) - samples) > 1:
            raise ValueError(f"{uid}: re-read audio has {len(audio)} samples, the cache {samples}")
        self.counts["original"] += 1
        return audio


class FeatureWriter:
    """Append-only records plus one index row each. Resumable: rows already indexed are skipped, and
    every run writes fresh shard files, so an interrupted run never leaves a referenced partial record."""

    def __init__(self, directory, shard_bytes=256 * 1024**2, commit_every=200):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.directory / "index.sqlite")
        self.db.executescript(SCHEMA)
        self.shard_bytes, self.commit_every = shard_bytes, commit_every
        existing = [int(p.stem.split("-")[-1]) for p in self.directory.glob("features-*.bin")]
        self.index = max(existing, default=-1) + 1
        self.file, self.pending = None, 0

    def has(self, table, uid):
        return self.db.execute(f"SELECT 1 FROM {table} WHERE uid=?", (uid,)).fetchone() is not None

    def add_frames(self, uid, features):
        data = np.asarray(features, dtype="<f2")
        if data.ndim != 2 or not np.isfinite(data).all():
            raise ValueError(f"Teacher frames of {uid} must be finite [frames, dim] in float16 range")
        if self.file is None or self.file.tell() + data.nbytes > self.shard_bytes:
            if self.file:
                self.file.close()
            self.name = f"features-{self.index:06d}.bin"
            self.file = open(self.directory / self.name, "xb")
            self.index += 1
        offset = self.file.tell() // (2 * data.shape[1])
        if self.file.tell() % (2 * data.shape[1]):
            raise ValueError("One store holds a single feature width")
        self.file.write(data.tobytes())
        self.db.execute(
            "INSERT INTO teacher_features VALUES (?,?,?,?,?)", (uid, self.name, offset, *data.shape)
        )
        self._tick()

    def add_vector(self, uid, vector):
        data = np.asarray(vector, dtype="<f4").reshape(-1)
        if not np.isfinite(data).all():
            raise ValueError(f"Nonfinite speaker embedding for {uid}")
        self.db.execute("INSERT INTO speaker_embeddings VALUES (?,?,?)", (uid, len(data), data.tobytes()))
        self._tick()

    def _tick(self):
        self.pending += 1
        if self.pending >= self.commit_every:
            self.commit()

    def commit(self):
        # Data reaches the disk before the index rows that reference it.
        if self.file:
            self.file.flush()
            os.fsync(self.file.fileno())
        self.db.commit()
        self.pending = 0

    def close(self):
        self.commit()
        if self.file:
            self.file.close()
        self.db.close()


def part_directory(output, shard_index, num_shards):
    output = Path(output)
    return output if num_shards == 1 else output / f"part-{shard_index:03d}-of-{num_shards:03d}"


def cache_identity(cache):
    meta = json.loads((Path(cache) / "metadata.json").read_text())
    keys = ("checkpoint", "sample_rate", "hop_length", "latent_dim", "index_sha256")
    return {"path": str(Path(cache).resolve()), **{key: meta.get(key) for key in keys}}


def _extract(kind, cache, output, compute, describe, splits, shard_index, num_shards, audio, progress):
    """Shared resumable loop: one partition directory, one record per selected cache row."""
    part = part_directory(output, shard_index, num_shards)
    meta_path = part / "metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("complete"):
        return json.loads(meta_path.read_text())
    table, total = KINDS[kind], count_rows(cache, splits, shard_index, num_shards)
    before = dict(audio.counts)
    rows = cache_rows(cache, splits, shard_index, num_shards)
    if progress:
        from tqdm import tqdm

        rows = tqdm(rows, total=total, desc=f"{kind} {shard_index + 1}/{num_shards}")
    writer = FeatureWriter(part)
    try:
        for row in rows:
            if not writer.has(table, row[0]):
                value = compute(row)
                (writer.add_frames if kind == "frames" else writer.add_vector)(row[0], value)
    finally:
        writer.close()
    with closing(sqlite3.connect(part / "index.sqlite")) as db:
        count, low, high = db.execute(f"SELECT count(*), min(dim), max(dim) FROM {table}").fetchone()
    if count != total or low != high:
        raise ValueError(f"{part}: {count} of {total} rows written, widths {low}..{high}")
    meta = {
        **describe,
        "kind": kind,
        "format_version": FORMAT_VERSION,
        "dim": high,
        "splits": list(splits),
        "cache": cache_identity(cache),
        "partition": shard_index,
        "partitions": num_shards,
        "rows": count,
        # Counts of this run only; a resumed partition reports what it computed after the restart.
        "audio_sources": {key: value - before.get(key, 0) for key, value in audio.counts.items()},
        "complete": True,
        "merged": num_shards == 1,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def extract_frames(
    cache,
    output,
    teacher,
    audio,
    splits=("train",),
    shard_index=0,
    num_shards=1,
    describe=None,
    progress=False,
):
    """Frame features for every selected row: audio -> teacher rate -> SSL frames -> latent rate [-> PCA].

    `teacher(waveform) -> [T,D]` exposes `sample_rate` and `frame_rate`; `audio` is a `CacheAudio`.
    A `pca.pt` in the store root (`fit_cache_pca`) projects every partition with the same basis.
    """
    meta = json.loads((Path(cache) / "metadata.json").read_text())
    latent_rate = meta["sample_rate"] / meta["hop_length"]
    ratio = teacher.frame_rate / latent_rate
    pca_path = Path(output) / "pca.pt"
    pca = torch.load(pca_path, weights_only=True) if pca_path.exists() else None

    def compute(row):
        waveform = resample(audio(row), audio.sample_rate, teacher.sample_rate)
        pooled = pool_frames(torch.as_tensor(teacher(waveform)), ratio, row[3])
        return (apply_pca(pooled, pca) if pca is not None else pooled).numpy()

    describe = {
        **(describe or {}),
        "frame_rate": latent_rate,
        "teacher_frame_rate": teacher.frame_rate,
        "pooling": f"mean-{ratio:g}" if abs(ratio - round(ratio)) < 1e-6 else "adaptive-average",
        "pca": None
        if pca is None
        else {"dim": int(pca["components"].shape[1]), "file": "pca.pt", "rows": int(pca["rows"])},
    }
    return _extract(
        "frames", cache, output, compute, describe, splits, shard_index, num_shards, audio, progress
    )


def extract_speakers(
    cache,
    output,
    embedder,
    audio,
    splits=("train",),
    shard_index=0,
    num_shards=1,
    describe=None,
    progress=False,
):
    """One utterance speaker embedding per selected row; `embedder(waveform) -> [E]` has `sample_rate`."""

    def compute(row):
        waveform = resample(audio(row), audio.sample_rate, embedder.sample_rate)
        return torch.as_tensor(embedder(waveform)).float().reshape(-1).numpy()

    return _extract(
        "speaker", cache, output, compute, describe or {}, splits, shard_index, num_shards, audio, progress
    )


def fit_cache_pca(cache, output, teacher, audio, dim, rows=1000, seed=0, splits=("train",)):
    """Fit the frame PCA once on `rows` deterministically sampled utterances; saves OUTPUT/pca.pt.

    Sharded extraction then projects every partition with the same basis. PCA centres the features,
    which also removes the large common direction that would otherwise dominate the cosine.
    """
    meta = json.loads((Path(cache) / "metadata.json").read_text())
    ratio = teacher.frame_rate / (meta["sample_rate"] / meta["hop_length"])
    total = count_rows(cache, splits)
    positions = set(random.Random(seed).sample(range(total), min(rows, total)))
    sums = products = None
    count = 0
    for row in cache_rows(cache, splits, positions=positions):
        waveform = resample(audio(row), audio.sample_rate, teacher.sample_rate)
        x = pool_frames(torch.as_tensor(teacher(waveform)), ratio, row[3]).double()
        sums = x.sum(0) if sums is None else sums + x.sum(0)
        products = x.T @ x if products is None else products + x.T @ x
        count += len(x)
    pca = fit_pca(sums, products, count, dim)
    pca.update(rows=len(positions), frames=count, seed=seed)
    Path(output).mkdir(parents=True, exist_ok=True)
    torch.save(pca, Path(output) / "pca.pt")
    return pca


def merge_parts(output, cache=None, allow_missing=False):
    """Combine `part-*` partitions into the root index (shards stay in place) and mark the store usable."""
    output = Path(output).resolve()
    parts = sorted(p for p in output.glob("part-*-of-*") if p.is_dir())
    if not parts:
        raise ValueError(f"No partitions under {output}")
    metas = [
        json.loads((p / "metadata.json").read_text()) if (p / "metadata.json").exists() else {} for p in parts
    ]
    incomplete = [str(part) for part, meta in zip(parts, metas) if not meta.get("complete")]
    if incomplete:
        raise ValueError(f"Incomplete partitions: {incomplete}")
    varying = {"partition", "rows", "audio_sources", "complete", "merged"}
    first = {k: v for k, v in metas[0].items() if k not in varying}
    for part, meta in zip(parts, metas):
        if {k: v for k, v in meta.items() if k not in varying} != first:
            raise ValueError(f"Partition {part} was extracted with different settings")
    if sorted(m["partition"] for m in metas) != list(range(first["partitions"])):
        raise ValueError(
            f"Expected {first['partitions']} partitions, found {sorted(m['partition'] for m in metas)}"
        )
    if (output / "index.sqlite").exists():
        raise ValueError(f"{output} already has a merged index")
    table = KINDS[first["kind"]]
    with closing(sqlite3.connect(output / "index.sqlite")) as db:
        db.executescript(SCHEMA)
        for part in parts:
            db.execute("ATTACH DATABASE ? AS part", (str(part / "index.sqlite"),))
            if table == "teacher_features":
                db.execute(
                    "INSERT INTO teacher_features SELECT uid, ? || '/' || shard, offset, frames, dim "
                    "FROM part.teacher_features",
                    (part.name,),
                )
            else:
                db.execute("INSERT INTO speaker_embeddings SELECT * FROM part.speaker_embeddings")
            db.commit()
            db.execute("DETACH DATABASE part")
        count = db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        missing = 0
        if cache is not None:
            db.execute("ATTACH DATABASE ? AS cache", (str(Path(cache).resolve() / "index.sqlite"),))
            marks = ",".join("?" * len(first["splits"]))
            missing = db.execute(
                f"SELECT count(*) FROM cache.samples s LEFT JOIN {table} t ON t.uid=s.uid "
                f"WHERE s.split IN ({marks}) AND t.uid IS NULL",
                tuple(first["splits"]),
            ).fetchone()[0]
    if missing and not allow_missing:
        (output / "index.sqlite").unlink()
        raise ValueError(f"{missing} cache rows of {first['splits']} have no entry after merging")
    sources = {}
    for meta in metas:
        for key, value in meta.get("audio_sources", {}).items():
            sources[key] = sources.get(key, 0) + value
    meta = {
        **first,
        "rows": count,
        "audio_sources": sources,
        "missing_rows": missing,
        "partition": None,
        "complete": True,
        "merged": True,
    }
    (output / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta
