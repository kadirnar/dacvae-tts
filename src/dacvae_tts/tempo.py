"""Pitch-preserving tempo change of speech (WSOLA) for prompt tempo perturbation.

Why: in the joined layout with length-aware RoPE, the text/audio alignment prior at the prompt/target boundary is
only right when the prompt speaks at the target's rate (frames per text unit). Within-utterance training never shows
a prompt at another tempo than its target, so any other target length "breaks" the prior at the boundary. VoiceStar
(arXiv:2505.19462, Table 2) stretches the prompt of cross-utterance training pairs by a factor in 1 +- 0.25 with
ffmpeg `atempo` (WSOLA) before encoding: WER 6.42 -> 5.66 on top of cross-utterance prompts (SIM 0.587 -> 0.583).
The perturbation has to happen on audio before the codec (the latent space has no tempo axis that can be resampled
without changing the decoded sound), hence this module and scripts/build_tempo_variants.py.

WSOLA (waveform-similarity overlap-add, Verhelst & Roelands, ICASSP 1993) reads Hann-windowed frames from the input
at the analysis hop `rate x synthesis hop`, each shifted within +-`tolerance` to the position whose waveform best
continues the previous output frame (mean-removed normalized cross-correlation), and overlap-adds them at the
synthesis hop. Pitch and formants are kept because every output frame is an unmodified piece of the input; only the
number of pitch periods changes. A phase vocoder sounds phasey on speech, PSOLA needs pitch marks, and resampling
("speed") shifts F0 by ~4 semitones at +-25 %, i.e. changes the voice. Settings as ffmpeg atempo / sox `tempo -s`:
40 ms frames at 50 % overlap (periodic Hann windows sum to one) and a 15 ms search, which covers one pitch period
of a 67 Hz voice. Keep rates within [0.8, 1.25]: plosives double when slowing down and vanish when speeding up.
"""

import json
import math
import os
import sqlite3
from collections import OrderedDict
from contextlib import closing
from pathlib import Path

import numpy as np
import torch


def time_stretch(audio, rate, sample_rate, frame_ms=40.0, tolerance_ms=15.0):
    """`audio` [samples] played `rate` times faster (rate > 1: shorter, rate < 1: longer), pitch unchanged.

    Output length is round(len(audio) / rate); rate 1 returns an exact float32 copy. Computed in float64,
    deterministic. The output is not loudness-normalized (a gain change would step at the prompt/target seam).
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be positive and finite")
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("time_stretch needs finite, nonempty audio")
    if rate == 1:
        return audio.copy()
    target = max(int(round(len(audio) / rate)), 1)
    frame = max(int(round(frame_ms * sample_rate / 1000)) // 2 * 2, 4)
    hop = frame // 2
    tolerance = max(int(round(tolerance_ms * sample_rate / 1000)), 1)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame) / frame)  # periodic Hann: halves sum to one
    front, back = hop + tolerance, frame + tolerance
    signal = audio.astype(np.float64)
    # Mirror the edges: zero padding would put an artificial step into the first/last templates and pull the match
    # (and the output) towards silence there; clips shorter than the padding fall back to zeros.
    mode = "reflect" if len(signal) > max(front, back) else "constant"
    source = np.pad(signal, (front, back), mode=mode)
    count = target // hop + 2
    output = np.zeros(count * hop + frame)
    weights = np.zeros_like(output)
    previous = None
    for k in range(count):
        # Output sample k*hop - hop/2 maps to input sample (k*hop - hop/2) * rate; frame k starts half a frame
        # before its centre, and the front padding shifts every input position by `front`.
        nominal = front + int(round((k * hop) * rate)) - hop
        nominal = min(max(nominal, 0), len(source) - frame)
        if previous is None:
            position = nominal
        else:
            continuation = source[previous + hop : previous + hop + frame]
            low, high = max(nominal - tolerance, 0), min(nominal + tolerance, len(source) - frame)
            position = low + best_match(source[low : high + frame], continuation)
        output[k * hop : k * hop + frame] += source[position : position + frame] * window
        weights[k * hop : k * hop + frame] += window
        previous = position
    result = output[hop:] / np.maximum(weights[hop:], 1e-3)  # output sample 0 = centre of frame 0
    return result[:target].astype(np.float32)


def best_match(region, template):
    """Offset in `region` (len >= len(template)) of the segment with the highest mean-removed normalized
    cross-correlation with `template`; 0 when the template or the region is silent (ties keep the nominal centre)."""
    width = len(template)
    template = template - template.mean()
    norm = np.sqrt(np.dot(template, template))
    positions = len(region) - width + 1
    if norm < 1e-9 or positions <= 1:
        return positions // 2
    sums = np.concatenate([[0.0], np.cumsum(region)])
    squares = np.concatenate([[0.0], np.cumsum(region * region)])
    total = sums[width:] - sums[:-width]
    energy = squares[width:] - squares[:-width] - total * total / width
    scores = np.correlate(region, template, mode="valid") / np.sqrt(np.maximum(energy, 1e-12)) / norm
    scores[energy < 1e-12] = -np.inf
    if not np.isfinite(scores).any():
        return positions // 2
    return int(np.argmax(scores))


# ----------------------------------------------------------------------------- tempo-variant latent store
#
# A sidecar directory next to a latent cache (the cache itself is never modified), like the teacher stores:
#
#   metadata.json        kind "tempo", format version, WSOLA settings, tempos (per mille), cache and codec identity,
#                        audio sources used, complete/merged flags
#   index.sqlite         tempo_latents(uid, tempo, shard, offset, frames, samples), one row per cache row and tempo
#   latents-NNNNNN.bin   little-endian float16 [frames, C] raw DACVAE posterior means (normalized at read time with
#                        the dataset's statistics, like the cache); shard paths are relative to the store root
#   part-III-of-NNN/     partitions of a sharded build, combined by `merge_tempo_parts`
#
# Tempo 1000 (x1.0) is stored too: rows prepared from embedded audio are decoded from their latents before the
# stretch, so stretched prompts are decode->encode round trips; drawing the x1.0 round trip as well keeps the model
# from learning "re-encoded" as a cue for "other tempo".


TEMPO_SCHEMA = """
CREATE TABLE IF NOT EXISTS tempo_latents (uid TEXT NOT NULL, tempo INTEGER NOT NULL, shard TEXT NOT NULL,
 offset INTEGER NOT NULL, frames INTEGER NOT NULL, samples INTEGER NOT NULL, PRIMARY KEY (uid, tempo));
"""
TEMPO_FORMAT_VERSION = 1
DEFAULT_TEMPOS = (800, 900, 1000, 1111, 1250)  # per mille, log-symmetric around x1.0 (VoiceStar: 1 +- 0.25)
WSOLA = {"name": "wsola", "frame_ms": 40.0, "tolerance_ms": 15.0}


def tempo_key(factor):
    """Tempo factor (1.25) or per-mille value (1250) -> the store's integer per-mille key."""
    value = float(factor)
    key = int(round(value * 1000)) if value < 10 else int(round(value))
    if not 500 <= key <= 2000:
        raise ValueError(f"Tempo {factor} outside x0.5-x2.0")
    return key


class TempoWriter:
    """Append-only float16 latent records plus one index row each; resumable per (uid, tempo)."""

    def __init__(self, directory, channels, shard_bytes=256 * 1024**2, commit_every=200):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.directory / "index.sqlite")
        self.db.executescript(TEMPO_SCHEMA)
        self.channels, self.shard_bytes, self.commit_every = channels, shard_bytes, commit_every
        existing = [int(p.stem.split("-")[-1]) for p in self.directory.glob("latents-*.bin")]
        self.index = max(existing, default=-1) + 1
        self.file, self.pending, self.name = None, 0, None

    def has(self, uid, tempo):
        return self.db.execute("SELECT 1 FROM tempo_latents WHERE uid=? AND tempo=?", (uid, tempo)).fetchone() is not None

    def add(self, uid, tempo, latents, samples):
        data = np.asarray(latents, dtype="<f2")
        if data.ndim != 2 or data.shape[1] != self.channels or not len(data) or not np.isfinite(data).all():
            raise ValueError(f"Tempo latents of {uid} must be finite [frames, {self.channels}] in float16 range")
        if self.file is None or self.file.tell() + data.nbytes > self.shard_bytes:
            if self.file:
                self.file.close()
            self.name = f"latents-{self.index:06d}.bin"
            self.file = open(self.directory / self.name, "xb")
            self.index += 1
        offset = self.file.tell() // (2 * self.channels)
        self.file.write(data.tobytes())
        self.db.execute("INSERT INTO tempo_latents VALUES (?,?,?,?,?,?)",
                        (uid, tempo, self.name, offset, len(data), int(samples)))
        self.pending += 1
        if self.pending >= self.commit_every:
            self.commit()

    def commit(self):
        if self.file:  # data reaches the disk before the index rows that reference it
            self.file.flush()
            os.fsync(self.file.fileno())
        self.db.commit()
        self.pending = 0

    def close(self):
        self.commit()
        if self.file:
            self.file.close()
        self.db.close()


def build_tempo_variants(cache, output, encode, audio, tempos=DEFAULT_TEMPOS, splits=("train",), shard_index=0,
                         num_shards=1, progress=False, codec_identity=None):
    """Stretch every selected cache row's waveform to each tempo (WSOLA) and store its encoded latents.

    `audio`: teacher.CacheAudio (the original file with the cache's loudness, else the decoded latents).
    `encode(waveform [samples] float32) -> raw latents [frames, C]` at the cache sample rate (Codec.encode).
    Resumable; returns the partition's metadata.
    """
    from .teacher import cache_identity, cache_rows, count_rows, part_directory

    tempos = sorted({tempo_key(t) for t in tempos})
    cache = Path(cache).resolve()
    meta = json.loads((cache / "metadata.json").read_text())
    rate, hop, channels = int(meta["sample_rate"]), int(meta["hop_length"]), int(meta["latent_dim"])
    part = part_directory(output, shard_index, num_shards)
    meta_path = part / "metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("complete"):
        return json.loads(meta_path.read_text())
    total = count_rows(cache, splits, shard_index, num_shards)
    rows = cache_rows(cache, splits, shard_index, num_shards)
    if progress:
        from tqdm import tqdm

        rows = tqdm(rows, total=total, desc=f"tempo {shard_index + 1}/{num_shards}")
    before = dict(audio.counts)
    writer = TempoWriter(part, channels)
    try:
        for row in rows:
            uid = row[0]
            todo = [t for t in tempos if not writer.has(uid, t)]
            if not todo:
                continue
            waveform = np.asarray(audio(row), dtype=np.float32)
            for tempo in todo:
                stretched = time_stretch(waveform, tempo / 1000, rate, WSOLA["frame_ms"], WSOLA["tolerance_ms"])
                latents = np.asarray(encode(stretched), dtype=np.float32)
                expected = math.ceil(len(stretched) / hop)
                if abs(len(latents) - expected) > 1:
                    raise ValueError(f"{uid} x{tempo / 1000}: {len(latents)} latent frames, {expected} expected")
                writer.add(uid, tempo, latents, len(stretched))
    finally:
        writer.close()
    with closing(sqlite3.connect(part / "index.sqlite")) as db:
        count = db.execute("SELECT count(*) FROM tempo_latents").fetchone()[0]
    if count != total * len(tempos):
        raise ValueError(f"{part}: {count} of {total * len(tempos)} records written")
    result = {
        "kind": "tempo",
        "format_version": TEMPO_FORMAT_VERSION,
        "algorithm": dict(WSOLA),
        "tempos": tempos,
        "channels": channels,
        "frame_rate": rate / hop,
        "splits": list(splits),
        "cache": cache_identity(cache),
        "codec": codec_identity or {k: meta.get(k) for k in ("checkpoint", "weights_sha256", "loudness_lufs",
                                                              "encoder_precision", "preprocessing")},
        "partition": shard_index,
        "partitions": num_shards,
        "rows": total,
        "audio_sources": {key: value - before.get(key, 0) for key, value in audio.counts.items()},
        "complete": True,
        "merged": num_shards == 1,
    }
    meta_path.write_text(json.dumps(result, indent=2))
    return result


def merge_tempo_parts(output, cache=None):
    """Combine `part-*` partitions into the root index (shards stay in place) and mark the store usable."""
    output = Path(output).resolve()
    parts = sorted(p for p in output.glob("part-*-of-*") if p.is_dir())
    if not parts:
        raise ValueError(f"No partitions under {output}")
    metas = [json.loads((p / "metadata.json").read_text()) if (p / "metadata.json").exists() else {} for p in parts]
    if not all(m.get("complete") for m in metas):
        raise ValueError(f"Incomplete partitions under {output}")
    varying = {"partition", "rows", "audio_sources", "complete", "merged"}
    first = {k: v for k, v in metas[0].items() if k not in varying}
    if any({k: v for k, v in m.items() if k not in varying} != first for m in metas):
        raise ValueError("Partitions were built with different settings")
    if sorted(m["partition"] for m in metas) != list(range(first["partitions"])):
        raise ValueError(f"Expected {first['partitions']} partitions, found {sorted(m['partition'] for m in metas)}")
    if (output / "index.sqlite").exists():
        raise ValueError(f"{output} already has a merged index")
    with closing(sqlite3.connect(output / "index.sqlite")) as db:
        db.executescript(TEMPO_SCHEMA)
        for part in parts:
            db.execute("ATTACH DATABASE ? AS part", (str(part / "index.sqlite"),))
            db.execute("INSERT INTO tempo_latents SELECT uid, tempo, ? || '/' || shard, offset, frames, samples "
                       "FROM part.tempo_latents", (part.name,))
            db.commit()
            db.execute("DETACH DATABASE part")
        rows = db.execute("SELECT count(DISTINCT uid) FROM tempo_latents").fetchone()[0]
    sources = {}
    for meta in metas:
        for key, value in meta.get("audio_sources", {}).items():
            sources[key] = sources.get(key, 0) + value
    result = {**first, "rows": rows, "audio_sources": sources, "partition": None, "complete": True, "merged": True}
    (output / "metadata.json").write_text(json.dumps(result, indent=2))
    if cache is not None:
        TempoStore(output).check(Path(cache).resolve() / "index.sqlite", first["splits"], first["tempos"])
    return result


class TempoStore:
    """Read-only access by (uid, tempo); SQLite handles and memory maps are reopened in each worker process."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        path = self.directory / "metadata.json"
        if not path.exists():
            raise ValueError(f"No tempo store at {self.directory}; run scripts/build_tempo_variants.py")
        self.meta = json.loads(path.read_text())
        if self.meta.get("kind") != "tempo" or self.meta.get("format_version") != TEMPO_FORMAT_VERSION:
            raise ValueError(f"{self.directory} is not a tempo store of format {TEMPO_FORMAT_VERSION}")
        if not self.meta.get("complete") or not self.meta.get("merged"):
            raise ValueError(f"Incomplete tempo store {self.directory}; finish the build and run merge")
        self.tempos, self.channels = tuple(self.meta["tempos"]), int(self.meta["channels"])
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

    def check(self, cache_db, splits, tempos, cache_meta=None):
        """Fail at dataset construction if a requested tempo or a row of the splits is missing, or if the store was
        built from another cache (index hash) or codec."""
        splits = [splits] if isinstance(splits, str) else list(splits)
        missing_tempos = sorted(set(tempos) - set(self.tempos))
        if missing_tempos:
            raise ValueError(f"{self.directory} has tempos {list(self.tempos)}, not {missing_tempos}")
        if cache_meta is not None:
            stored = self.meta.get("cache", {})
            for key in ("index_sha256", "checkpoint", "latent_dim", "hop_length", "sample_rate"):
                if stored.get(key) is not None and cache_meta.get(key) is not None and stored[key] != cache_meta[key]:
                    raise ValueError(f"Tempo store {self.directory} was built from another cache ({key} differs)")
        marks = ",".join("?" * len(splits))
        with closing(sqlite3.connect(f"file:{cache_db}?mode=ro", uri=True)) as db:
            db.execute("ATTACH DATABASE ? AS tempo", (f"file:{self.db_path}?mode=ro",))
            for tempo in tempos:
                missing = db.execute(
                    f"SELECT count(*), min(s.uid) FROM samples s LEFT JOIN tempo.tempo_latents t "
                    f"ON t.uid=s.uid AND t.tempo=? WHERE s.split IN ({marks}) AND t.uid IS NULL",
                    (tempo, *splits),
                ).fetchone()
                if missing[0]:
                    raise ValueError(f"{missing[0]} rows have no x{tempo / 1000} latents in {self.directory} "
                                     f"(e.g. {missing[1]})")

    def lengths(self, uids, tempo):
        """int32 frame counts of `uids` (in that order) at `tempo`."""
        found = dict(self._connection().execute("SELECT uid, frames FROM tempo_latents WHERE tempo=?", (tempo,)))
        return np.array([found[uid] for uid in uids], dtype=np.int32)

    def latents(self, uid, tempo):
        """Raw float32 [frames, C] latents of one row at one tempo."""
        row = self._connection().execute(
            "SELECT shard, offset, frames FROM tempo_latents WHERE uid=? AND tempo=?", (uid, tempo)
        ).fetchone()
        if row is None:
            raise KeyError(f"No x{tempo / 1000} latents for {uid} in {self.directory}")
        shard, offset, frames = row
        if shard not in self._maps:
            self._maps[shard] = np.memmap(self.directory / shard, dtype="<f2", mode="r").reshape(-1, self.channels)
            if len(self._maps) > 32:
                self._maps.popitem(last=False)
        self._maps.move_to_end(shard)
        return torch.from_numpy(np.array(self._maps[shard][offset : offset + frames], dtype=np.float32))
