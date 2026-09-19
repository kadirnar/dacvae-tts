import hashlib
import io
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .codec import Codec, check_compatibility, file_digest, read_audio
from .data import SCHEMA, ShardWriter, jsonl, save_stats, speaker_split
from .text import normalize


def source_manifest_digest(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted([*path.rglob("*.jsonl"), *path.rglob("*.parquet")])
    digest = hashlib.sha256()
    for file in files:
        stat = file.stat()
        digest.update(json.dumps([str(file), stat.st_size, stat.st_mtime_ns]).encode())
    return digest.hexdigest()


def source_rows(path, shard_index=0, num_shards=1):
    path = Path(path)
    if path.is_dir():
        files = sorted([*path.rglob("*.parquet"), *path.rglob("*.jsonl")])
        if not files:
            raise ValueError("Dataset directory contains no Parquet or JSONL files")
        for file_index, file in enumerate(files):
            if len(files) >= num_shards:
                if file_index % num_shards != shard_index:
                    continue
                iterator = source_rows(file)
            else:
                iterator = source_rows(file, shard_index, num_shards)
            for row in iterator:
                # Preserve explicit corpus IDs, otherwise namespace the generated row index.
                if row.get("_generated_id"):
                    row["id"] = f"{file.relative_to(path)}:{row['id']}"
                yield row
        return
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        source = pq.ParquetFile(path)
        offset = 0
        for group in range(source.num_row_groups):
            count = source.metadata.row_group(group).num_rows
            if source.num_row_groups >= num_shards:
                if group % num_shards != shard_index:
                    offset += count
                    continue
                partition_rows = False
            else:
                partition_rows = True
            index = offset
            for batch in source.iter_batches(batch_size=256, row_groups=[group]):
                for row in batch.to_pylist():
                    if not partition_rows or index % num_shards == shard_index:
                        if "id" not in row:
                            row.update(id=str(index), _generated_id=True)
                        yield row
                    index += 1
            offset += count
    else:
        for index, row in enumerate(jsonl(path)):
            if index % num_shards == shard_index:
                if "id" not in row:
                    row.update(id=str(index), _generated_id=True)
                yield row


def prepare(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "index.sqlite").exists():
        raise ValueError("Output already contains a cache; use a new partition directory")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    codec = Codec(args.codec, args.device)
    writer = ShardWriter(out, codec.latent_dim)
    sums = torch.zeros(codec.latent_dim, dtype=torch.float64)
    squares, count = sums.clone(), 0
    accepted, rejected = 0, 0
    source_path = Path(args.manifest).resolve()
    root = source_path if source_path.is_dir() else source_path.parent
    db = sqlite3.connect(out / "index.sqlite")
    db.executescript(SCHEMA)
    try:
        with open(out / "rejected.jsonl", "w") as failures:
            for index, row in enumerate(
                tqdm(source_rows(args.manifest, args.shard_index, args.num_shards), desc="Encode")
            ):
                missing = {args.text_column, args.audio_column, args.speaker_column} - row.keys()
                if missing:
                    raise ValueError(f"Missing required dataset columns: {sorted(missing)}")
                uid = str(row.get("id", index))
                try:
                    text = normalize(
                        row[args.text_column],
                        getattr(args, "text_normalization", "unicode-v1"),
                        row.get("spoken_text"),
                    )
                    speaker = str(row[args.speaker_column]).strip()
                    if not speaker or row[args.speaker_column] is None:
                        raise ValueError("Missing speaker identity")
                    language = row.get("language", "en")
                    if language not in {"en", "eng", "English", "english", "en-US", "en-GB"}:
                        raise ValueError(f"Non-English language tag: {language}")
                    source = row[args.audio_column]
                    if isinstance(source, dict):
                        source = io.BytesIO(source["bytes"]) if source.get("bytes") else source["path"]
                    if isinstance(source, (str, Path)):
                        source = Path(source)
                        source = source if source.is_absolute() else root / source
                    audio = read_audio(source, codec.sample_rate)
                    duration = len(audio) / codec.sample_rate
                    if not args.min_seconds <= duration <= args.max_seconds:
                        raise ValueError(f"Duration {duration:.3f}s outside accepted range")
                    if audio.square().mean().sqrt() < 1e-5:
                        raise ValueError("Silent audio")
                    digest = hashlib.sha256(audio.numpy().tobytes()).hexdigest()
                    if db.execute("SELECT 1 FROM samples WHERE uid=? OR digest=?", (uid, digest)).fetchone():
                        raise ValueError("Duplicate audio or ID")
                    split = row.get("split") or speaker_split(speaker, args.seed)
                    if split not in {"train", "val", "test"}:
                        raise ValueError("split must be train, val or test")
                    z = codec.encode(audio).cpu()
                    shard, offset = writer.write(z.numpy())
                    db.execute(
                        "INSERT INTO samples VALUES (NULL,?,?,?,?,?,?,?,?,?,?)",
                        (
                            uid,
                            speaker,
                            text,
                            str(source) if not isinstance(source, io.BytesIO) else "embedded",
                            shard,
                            offset,
                            len(z),
                            split,
                            len(audio),
                            digest,
                        ),
                    )
                    db.execute(
                        "INSERT INTO provenance VALUES (?,?,?,?,?,?,?)",
                        (
                            uid,
                            row[args.text_column],
                            text,
                            row.get("session_id"),
                            row.get("source_recording"),
                            row.get("start_seconds"),
                            row.get("end_seconds"),
                        ),
                    )
                    if split == "train":
                        z64 = z.double()
                        sums += z64.sum(0)
                        squares += z64.square().sum(0)
                        count += len(z)
                    accepted += 1
                    if accepted % 1000 == 0:
                        db.commit()
                except (ValueError, KeyError, OSError, RuntimeError) as exc:
                    # CUDA OOM or a broken codec is not a dataset error to silently skip 4M times.
                    if isinstance(exc, RuntimeError):
                        raise
                    failures.write(json.dumps({"id": uid, "reason": str(exc)}) + "\n")
                    rejected += 1
    finally:
        writer.close()
        db.commit()
        db.close()
    metadata = {
        **codec.metadata,
        "source": str(Path(args.manifest).resolve()),
        "seed": args.seed,
        "partition": args.shard_index,
        "partitions": args.num_shards,
        "accepted": accepted,
        "rejected": rejected,
        "complete": True,
        "text_normalization": getattr(args, "text_normalization", "unicode-v1"),
        "source_inventory_sha256": source_manifest_digest(args.manifest),
        "pairing_version": "distinct_utterance_same_speaker_v1",
        "split_version": "speaker_sha256_98_1_1_v1",
        "filtering": {
            "min_seconds": args.min_seconds,
            "max_seconds": args.max_seconds,
            "silence_rms_min": 1e-5,
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    save_stats(out / "stats.pt", count, sums, squares)
    print(json.dumps(metadata, indent=2))
    if not accepted:
        raise ValueError("No accepted samples; inspect rejected.jsonl")


def merge(args):
    """Merge partitions without copying large binary shards; recompute retained train statistics."""
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "index.sqlite").exists():
        raise ValueError("Merged output already exists")
    db = sqlite3.connect(out / "index.sqlite")
    db.executescript(SCHEMA)
    meta, duplicates, rejected = None, 0, 0
    partitions = {}
    fields = "uid,speaker,text,audio,shard,offset,frames,split,samples,digest"
    try:
        for directory in args.inputs:
            directory = Path(directory).resolve()
            current = json.loads((directory / "metadata.json").read_text())
            if not current.get("complete"):
                raise ValueError(f"Incomplete partition: {directory}")
            rejected += current.get("rejected", 0)
            if "source" in current and "partition" in current:
                key = (current["source"], current["partitions"])
                partitions.setdefault(key, set()).add(current["partition"])
            if meta is None:
                meta = current.copy()
            check_compatibility(meta, current)
            if meta.get("text_normalization", "unicode-v1") != current.get(
                "text_normalization", "unicode-v1"
            ):
                raise ValueError("Cannot merge different text-normalization versions")
            for key in ("checkpoint", "latent_dim", "sample_rate", "hop_length", "posterior", "seed"):
                if meta[key] != current[key]:
                    raise ValueError(f"Incompatible partitions: {key}")
            with sqlite3.connect(directory / "index.sqlite") as source:
                for row in source.execute(f"SELECT {fields} FROM samples"):
                    # IDs must be globally unique. Duplicate content is dropped; ID collisions are errors.
                    exists = db.execute("SELECT digest FROM samples WHERE uid=?", (row[0],)).fetchone()
                    if exists and exists[0] != row[-1]:
                        raise ValueError(f"ID collision with different audio: {row[0]}")
                    duplicate = db.execute(
                        "SELECT speaker,text,split FROM samples WHERE digest=?", (row[-1],)
                    ).fetchone()
                    if duplicate and duplicate != (row[1], row[2], row[7]):
                        raise ValueError(
                            f"Duplicate audio has inconsistent speaker/transcript/split labels: {row[0]}"
                        )
                    before = db.total_changes
                    db.execute(f"INSERT OR IGNORE INTO samples ({fields}) VALUES (?,?,?,?,?,?,?,?,?,?)", row)
                    duplicates += db.total_changes == before
                if source.execute("SELECT 1 FROM sqlite_master WHERE name='provenance'").fetchone():
                    db.executemany(
                        "INSERT OR IGNORE INTO provenance VALUES (?,?,?,?,?,?,?)",
                        source.execute("SELECT * FROM provenance"),
                    )
            db.commit()
        for (source, expected), observed in partitions.items():
            if observed != set(range(expected)):
                raise ValueError(
                    f"Missing partitions for {source}: expected {expected}, found {sorted(observed)}"
                )
        leakage = db.execute(
            "SELECT speaker FROM samples GROUP BY speaker HAVING count(DISTINCT split)>1 LIMIT 1"
        ).fetchone()
        if leakage:
            raise ValueError(f"Speaker appears across splits: {leakage[0]}")
        before = db.total_changes
        db.execute(
            "DELETE FROM samples WHERE speaker IN (SELECT speaker FROM samples GROUP BY speaker HAVING count(*)<2)"
        )
        singletons = db.total_changes - before
        db.execute("DELETE FROM provenance WHERE uid NOT IN (SELECT uid FROM samples)")
        db.commit()
        channels = meta["latent_dim"]
        sums = torch.zeros(channels, dtype=torch.float64)
        squares, count = sums.clone(), 0
        previous_shard, mapped = None, None
        for shard, offset, frames in tqdm(
            db.execute("SELECT shard,offset,frames FROM samples WHERE split='train' ORDER BY shard,offset"),
            desc="Statistics",
        ):
            if shard != previous_shard:
                mapped = np.memmap(shard, dtype="<f2", mode="r").reshape(-1, channels)
                previous_shard = shard
            z = torch.from_numpy(np.array(mapped[offset : offset + frames], dtype=np.float64))
            sums += z.sum(0)
            squares += z.square().sum(0)
            count += frames
        save_stats(out / "stats.pt", count, sums, squares)
        splits = dict(db.execute("SELECT split,count(*) FROM samples GROUP BY split").fetchall())
        meta.update(
            {
                "complete": True,
                "merged": True,
                "split_counts": splits,
                "accepted": sum(splits.values()),
                "rejected": rejected,
                "duplicates_removed": duplicates,
                "singleton_rows_removed": singletons,
                "inputs": [str(Path(p).resolve()) for p in args.inputs],
                "index_sha256": file_digest(out / "index.sqlite"),
            }
        )
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
    finally:
        db.close()
