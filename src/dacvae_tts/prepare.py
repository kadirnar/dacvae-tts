import hashlib
import io
import json
import multiprocessing
import sqlite3
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing
from functools import partial
from itertools import islice
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from .codec import Codec, backend_options, check_compatibility, file_digest, read_audio
from .data import SCHEMA, ShardWriter, save_stats, speaker_split
from .parallel import initialize_worker
from .speakers import assign_split, check_split_groups, compile_split_key, load_split_map, split_group
from .text import encode_ids, normalize

ENGLISH_TAGS = {"en", "eng", "English", "english", "en-US", "en-GB"}
UNRECORDED_SOURCE = "(unrecorded)"


def source_summary(counts):
    """{source: {"licenses", "prepared_rows"}} from accepted-row counts keyed by the manifest's (source, license).

    Empty when no row has either column, so caches of manifests without them keep their metadata. The licenses
    (CC0-1.0, MIT, CC-BY-3.0 in scripts/data/) live in metadata only: the index schema has no per-row column.
    """
    if all(source is None and license is None for source, license in counts):
        return {}
    summary = {}
    for (source, license), rows in counts.items():
        entry = summary.setdefault(UNRECORDED_SOURCE if source is None else source, {"licenses": [], "prepared_rows": 0})
        entry["prepared_rows"] += rows
        if license is not None and license not in entry["licenses"]:
            entry["licenses"].append(license)
    return {name: {**entry, "licenses": sorted(entry["licenses"])} for name, entry in sorted(summary.items())}


def merge_source_summaries(summaries):
    """Union of licenses and sum of prepared rows over the merge inputs' `sources` summaries."""
    merged = {}
    for summary in summaries:
        for name, entry in summary.items():
            target = merged.setdefault(name, {"licenses": set(), "prepared_rows": 0})
            target["licenses"].update(entry["licenses"])
            target["prepared_rows"] += entry["prepared_rows"]
    return {name: {**entry, "licenses": sorted(entry["licenses"])} for name, entry in sorted(merged.items())}


def source_manifest_digest(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted([*path.rglob("*.jsonl"), *path.rglob("*.parquet")])
    digest = hashlib.sha256()
    for file in files:
        stat = file.stat()
        digest.update(json.dumps([str(file), stat.st_size, stat.st_mtime_ns]).encode())
    return digest.hexdigest()


def source_rows(path, shard_index=0, num_shards=1):
    if not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard index")
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
        # Scan lines on each rank, but deserialize only this rank's rows. Preserve
        # the legacy row numbering (blank lines do not consume an index).
        index = 0
        with open(path, "rb") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                selected = index % num_shards == shard_index
                if selected:
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeDecodeError) as exc:
                        raise ValueError(f"{path}:{line_number}: {exc}") from exc
                    if "id" not in row:
                        row.update(id=str(index), _generated_id=True)
                    yield row
                index += 1


def ordered_prefetch(function, rows, workers, prefetch, backend="thread", worker_threads=1):
    """Bounded CPU decoding; results retain manifest order on every rank."""
    if workers < 0 or prefetch < 1 or worker_threads < 1 or backend not in {"thread", "process"}:
        raise ValueError("Invalid CPU prefetch configuration")
    if workers == 0:
        yield from map(function, rows)
        return
    executor = (
        ThreadPoolExecutor(max_workers=workers, thread_name_prefix="audio-decode")
        if backend == "thread"
        else ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=partial(initialize_worker, threads=worker_threads),
        )
    )
    pending = deque()
    rows = iter(rows)
    try:
        for row in islice(rows, prefetch):
            pending.append(executor.submit(function, row))
        while pending:
            result = pending.popleft().result()
            # Refill before yielding so CPU decoding overlaps the consumer's GPU work.
            for row in islice(rows, 1):
                pending.append(executor.submit(function, row))
            yield result
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def prepare_record(row, args, root, sample_rate):
    missing = {args.text_column, args.audio_column, args.speaker_column} - row.keys()
    if missing:
        raise ValueError(f"Missing required dataset columns: {sorted(missing)}")
    uid = str(row["id"])
    try:
        # Cheap metadata filters run before any audio is decoded.
        quality_column = getattr(args, "quality_column", None)
        if quality_column and getattr(args, "min_quality", None) is not None:
            quality = row.get(quality_column)
            if quality is None or quality < args.min_quality:
                raise ValueError(f"Quality {quality} below {args.min_quality}")
        if getattr(args, "reject_digits", False) and any(c.isdigit() for c in row[args.text_column] or ""):
            raise ValueError("Transcript contains digits")
        text = normalize(
            row[args.text_column], getattr(args, "text_normalization", "unicode-v1"), row.get("spoken_text")
        )
        speaker = str(row[args.speaker_column]).strip()
        if not speaker or row[args.speaker_column] is None:
            raise ValueError("Missing speaker identity")
        accepted = getattr(args, "languages", None) or ENGLISH_TAGS
        language = row.get("language", next(iter(accepted)))
        if accepted != {"any"} and language not in accepted:
            raise ValueError(f"Language tag not accepted: {language}")
        # --split-key hashes an episode/program key instead of the label, so whole episodes are held out.
        split_key = getattr(args, "split_key", None)
        split = row.get("split") or speaker_split(split_group(speaker, split_key), args.seed)
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        source = row[args.audio_column]
        if isinstance(source, dict):
            source = io.BytesIO(source["bytes"]) if source.get("bytes") else source["path"]
        if isinstance(source, (str, Path)):
            source = Path(source)
            source = source if source.is_absolute() else root / source
        audio = read_audio(source, sample_rate, getattr(args, "loudness", None))
        duration = len(audio) / sample_rate
        if not args.min_seconds <= duration <= args.max_seconds:
            raise ValueError(f"Duration {duration:.3f}s outside accepted range")
        array = audio.numpy()
        if np.sqrt(np.mean(np.square(array))) < 1e-5:
            raise ValueError("Silent audio")
        digest = hashlib.sha256(memoryview(array)).hexdigest()
        return dict(
            uid=uid,
            text=text,
            text_bytes=text.encode("utf-8"),
            token_ids=encode_ids(text.encode("utf-8")).tobytes(),
            speaker=speaker,
            split=split,
            audio=audio,
            digest=digest,
            provenance=(
                uid,
                row[args.text_column],
                text,
                row.get("session_id"),
                row.get("source_recording"),
                row.get("start_seconds"),
                row.get("end_seconds"),
            ),
            source=str(source) if not isinstance(source, io.BytesIO) else "embedded",
            # scripts/data/ manifests name each row's corpus and license; other datasets may have neither.
            corpus=tuple(None if row.get(key) is None else str(row[key]) for key in ("source", "license")),
        )
    except (ValueError, KeyError, OSError, sf.LibsndfileError) as exc:
        return {"uid": uid, "error": str(exc)}


def encode_records(codec, records, batch_size, batch_seconds, precision, counters):
    """Bucket within a bounded window, then restore source order before persistence."""
    groups = defaultdict(list)
    for i, record in enumerate(records):
        frames = (len(record["audio"]) + codec.hop_length - 1) // codec.hop_length
        groups[frames].append(i)
    results = [None] * len(records)

    def run(indices):
        counters["encoder_calls"] += 1
        try:
            output = codec.encode_batch([records[i]["audio"] for i in indices], precision)
        except torch.cuda.OutOfMemoryError:
            if len(indices) == 1:
                raise
            counters["oom_retries"] += 1
        else:
            for i, latent in zip(indices, output, strict=True):
                results[i] = latent
            return
        # Release the failed call's traceback before retrying smaller batches.
        torch.cuda.empty_cache()
        middle = len(indices) // 2
        run(indices[:middle])
        run(indices[middle:])

    for frames, indices in groups.items():
        limit = min(batch_size, int(batch_seconds * codec.sample_rate / (frames * codec.hop_length)))
        if limit < 1:
            raise ValueError("--batch-seconds is smaller than one hop-rounded recording")
        for start in range(0, len(indices), limit):
            run(indices[start : start + limit])
    return results


def prepare(args):
    workers = getattr(args, "workers", 4)
    prefetch = getattr(args, "prefetch", 16)
    batch_size = getattr(args, "batch_size", 8)
    bucket_size = getattr(args, "bucket_size", 256)
    batch_seconds = getattr(args, "batch_seconds", 120.0)
    precision = getattr(args, "precision", "fp32")
    fold = not getattr(args, "no_fold_weight_norm", False)
    if workers < 0 or min(prefetch, batch_size, bucket_size, batch_seconds) <= 0:
        raise ValueError("workers must be nonnegative; prefetch and batching limits must be positive")
    if not np.isfinite(batch_seconds) or not 0 < args.min_seconds <= args.max_seconds < float("inf"):
        raise ValueError("Duration and batch-seconds limits must be finite and positive")
    if precision not in {"fp32", "bf16"}:
        raise ValueError("Encoder precision must be fp32 or bf16")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    split_key = compile_split_key(getattr(args, "split_key", None))  # fail before loading the codec
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "index.sqlite").exists():
        raise ValueError("Output already contains a cache; use a new partition directory")
    started = time.perf_counter()
    options = backend_options(args)
    if options.get("backend") == "fast" and not fold:
        raise ValueError("Fast codec requires folded weight normalization")
    codec = Codec(
        args.codec,
        args.device,
        encoder_only=True,
        fold_weight_norm=fold,
        loudness=getattr(args, "loudness", None),
        **options,
    )
    if precision == "bf16" and (codec.device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 cache encoding requires a BF16-capable CUDA device")
    setup_seconds = time.perf_counter() - started
    processing_started = time.perf_counter()
    writer = ShardWriter(out, codec.latent_dim)
    sums = torch.zeros(codec.latent_dim, dtype=torch.float64)
    squares, count = sums.clone(), 0
    accepted, rejected, committed, audio_seconds = 0, 0, 0, 0.0
    counters = {"encoder_calls": 0, "oom_retries": 0}
    timings = {"input_wait_seconds": 0.0, "encode_seconds": 0.0, "write_seconds": 0.0}
    corpora = Counter()  # accepted rows per manifest (source, license)
    source_path = Path(args.manifest).resolve()
    root = source_path if source_path.is_dir() else source_path.parent
    db = sqlite3.connect(out / "index.sqlite")
    db.executescript(SCHEMA)
    rows = source_rows(args.manifest, args.shard_index, args.num_shards)
    records = ordered_prefetch(
        partial(prepare_record, args=args, root=root, sample_rate=codec.sample_rate),
        rows,
        workers,
        prefetch,
        getattr(args, "worker_backend", "thread"),
        getattr(args, "worker_threads", 1),
    )
    try:
        with closing(records), open(out / "rejected.jsonl", "w") as failures, tqdm(desc="Encode") as bar:
            while True:
                tick = time.perf_counter()
                window = list(islice(records, bucket_size))
                timings["input_wait_seconds"] += time.perf_counter() - tick
                if not window:
                    break
                selected, seen_ids, seen_digests = [], set(), set()
                for record in window:
                    uid = record["uid"]
                    if "error" not in record:
                        digest = record["digest"]
                        if (
                            uid in seen_ids
                            or digest in seen_digests
                            or db.execute(
                                "SELECT 1 FROM samples WHERE uid=? OR digest=?", (uid, digest)
                            ).fetchone()
                        ):
                            record["error"] = "Duplicate audio or ID"
                    if "error" in record:
                        failures.write(json.dumps({"id": uid, "reason": record["error"]}) + "\n")
                        rejected += 1
                        # Release rejected waveforms before GPU work.
                        record.pop("audio", None)
                    else:
                        selected.append(record)
                        seen_ids.add(uid)
                        seen_digests.add(record["digest"])
                tick = time.perf_counter()
                latents = encode_records(codec, selected, batch_size, batch_seconds, precision, counters)
                timings["encode_seconds"] += time.perf_counter() - tick
                tick = time.perf_counter()
                for record, z in zip(selected, latents, strict=True):
                    uid, split = record["uid"], record["split"]
                    shard, offset = writer.write(z.numpy())
                    db.execute(
                        "INSERT INTO samples VALUES (NULL,?,?,?,?,?,?,?,?,?,?)",
                        (
                            uid,
                            record["speaker"],
                            record["text"],
                            record["source"],
                            shard,
                            offset,
                            len(z),
                            split,
                            len(record["audio"]),
                            record["digest"],
                        ),
                    )
                    db.execute("INSERT INTO text_tokens VALUES (?,?)", (uid, record["text_bytes"]))
                    db.execute("INSERT INTO token_ids VALUES (?,?)", (uid, record["token_ids"]))
                    db.execute(
                        "INSERT INTO provenance VALUES (?,?,?,?,?,?,?)",
                        record["provenance"],
                    )
                    if split == "train":
                        # Compute statistics on the stored representation, as merge does.
                        z64 = z.half().double()
                        sums += z64.sum(0)
                        squares += z64.square().sum(0)
                        count += len(z)
                    accepted += 1
                    corpora[record["corpus"]] += 1
                    audio_seconds += len(record["audio"]) / codec.sample_rate
                if accepted - committed >= 1000:
                    db.commit()
                    committed = accepted
                timings["write_seconds"] += time.perf_counter() - tick
                bar.update(len(window))
                bar.set_postfix(
                    accepted=accepted, rejected=rejected, calls=counters["encoder_calls"], refresh=False
                )
                # Avoid retaining the previous window while filling the next one.
                del window, selected, latents
    finally:
        writer.close()
        db.commit()
        db.close()
    processing_seconds = time.perf_counter() - processing_started
    if not accepted:
        raise ValueError("No accepted samples; inspect rejected.jsonl")
    metadata = {
        **codec.metadata,
        "source": str(source_path),
        "seed": args.seed,
        "partition": args.shard_index,
        "partitions": args.num_shards,
        "accepted": accepted,
        "rejected": rejected,
        "complete": True,
        "text_normalization": getattr(args, "text_normalization", "unicode-v1"),
        "text_tokenizer": "utf8-bytes-v1",
        "token_ids": "bos-bytes+4-eos-uint16-v1",
        "languages": sorted(getattr(args, "languages", None) or ENGLISH_TAGS),
        "encoder_precision": precision,
        "source_inventory_sha256": source_manifest_digest(args.manifest),
        "pairing_version": "distinct_utterance_same_speaker_v1",
        "split_version": "speaker_sha256_98_1_1_v1",
        "filtering": {
            "min_seconds": args.min_seconds,
            "max_seconds": args.max_seconds,
            "silence_rms_min": 1e-5,
            "min_quality": getattr(args, "min_quality", None),
            "quality_column": getattr(args, "quality_column", None),
            "reject_digits": getattr(args, "reject_digits", False),
        },
        "preparation": {
            "workers": workers,
            "worker_backend": getattr(args, "worker_backend", "thread"),
            "worker_threads": getattr(args, "worker_threads", 1),
            "codec_runtime": codec.metadata.get("codec_runtime", {"backend": "reference"}),
            "codec_graphs": codec._fast.graph_statistics() if getattr(codec, "_fast", None) else {},
            "prefetch": prefetch,
            "batch_size": batch_size,
            "bucket_size": bucket_size,
            "batch_seconds": batch_seconds,
            "fold_weight_norm": fold,
            "cudnn_allow_tf32": False,
            "batching": "exact_hop_length_v1",
            "setup_seconds": setup_seconds,
            "processing_seconds": processing_seconds,
            "audio_seconds": audio_seconds,
            "audio_seconds_per_wall_second": audio_seconds / processing_seconds,
            "rows_per_second": accepted / processing_seconds,
            **timings,
            **counters,
        },
    }
    if split_key:
        metadata.update(split_key=split_key.pattern, split_version="split_key_sha256_98_1_1_v1")
    sources = source_summary(corpora)
    if sources:
        metadata["sources"] = sources
    save_stats(out / "stats.pt", count, sums, squares)
    # A completion marker is written only after both the index and statistics exist.
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


def merge(args):
    """Merge partitions without copying large binary shards; recompute retained train statistics.

    --split-key/--split-map re-split every row while merging (no re-encoding): the map entry of the label,
    else of its split key, else the split-key hash, else the partition's split. Statistics follow the new
    train split."""
    out = Path(args.output).resolve()
    # Validated before the output index exists, so a bad option never leaves a half-created cache.
    split_key = compile_split_key(getattr(args, "split_key", None))
    split_map = load_split_map(args.split_map) if getattr(args, "split_map", None) else None
    resplit, reassigned = split_key is not None or split_map is not None, 0
    out.mkdir(parents=True, exist_ok=True)
    if (out / "index.sqlite").exists():
        raise ValueError("Merged output already exists")
    db = sqlite3.connect(out / "index.sqlite")
    # Millions of rows go through three indexed lookups each; durability is not needed while merging.
    db.execute("PRAGMA journal_mode=MEMORY")
    db.execute("PRAGMA synchronous=OFF")
    db.executescript(SCHEMA)
    meta, duplicates, rejected, conflicts = None, 0, 0, 0
    partitions = {}
    preparation_partitions, source_summaries = [], []
    fields = "uid,speaker,text,audio,shard,offset,frames,split,samples,digest"
    try:
        for directory in args.inputs:
            directory = Path(directory).resolve()
            current = json.loads((directory / "metadata.json").read_text())
            if "preparation" in current:
                preparation_partitions.append({"directory": str(directory), **current["preparation"]})
            if not current.get("complete"):
                raise ValueError(f"Incomplete partition: {directory}")
            rejected += current.get("rejected", 0)
            source_summaries.append((current.get("sources"), current.get("accepted", 0)))
            if "source" in current and "partition" in current:
                key = (current["source"], current["partitions"])
                partitions.setdefault(key, set()).add(current["partition"])
            if meta is None:
                meta = current.copy()
            check_compatibility(meta, current)
            if meta.get("encoder_precision", "fp32") != current.get("encoder_precision", "fp32"):
                raise ValueError("Cannot merge different encoder precisions")
            if meta.get("text_tokenizer", "utf8-bytes-v1") != current.get("text_tokenizer", "utf8-bytes-v1"):
                raise ValueError("Cannot merge different text tokenizers")
            if meta.get("text_normalization", "unicode-v1") != current.get(
                "text_normalization", "unicode-v1"
            ):
                raise ValueError("Cannot merge different text-normalization versions")
            for key in ("checkpoint", "latent_dim", "sample_rate", "hop_length", "posterior", "seed"):
                if meta[key] != current[key]:
                    raise ValueError(f"Incompatible partitions: {key}")
            if not resplit and meta.get("split_key") != current.get("split_key"):
                raise ValueError("Incompatible partitions: split_key (re-split them with --split-key)")
            with sqlite3.connect(directory / "index.sqlite") as source:
                for row in source.execute(f"SELECT {fields} FROM samples"):
                    if resplit:
                        split = assign_split(row[1], row[7], meta["seed"], split_key, split_map)
                        reassigned += split != row[7]
                        row = (*row[:7], split, *row[8:])
                    # IDs must be globally unique. Duplicate content is dropped; ID collisions are errors.
                    exists = db.execute("SELECT digest FROM samples WHERE uid=?", (row[0],)).fetchone()
                    if exists and exists[0] != row[-1]:
                        raise ValueError(f"ID collision with different audio: {row[0]}")
                    duplicate = db.execute(
                        "SELECT speaker,text,split FROM samples WHERE digest=?", (row[-1],)
                    ).fetchone()
                    if duplicate and duplicate != (row[1], row[2], row[7]):
                        if getattr(args, "drop_conflicting_duplicates", False):
                            # Web-scale corpora repeat jingles/adverts under different labels: keep the first.
                            conflicts += 1
                            continue
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
                for table in ("text_tokens", "token_ids"):
                    if source.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
                        db.executemany(
                            f"INSERT OR IGNORE INTO {table} VALUES (?,?)",
                            source.execute(f"SELECT * FROM {table}"),
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
        group_key = split_key if resplit else compile_split_key(meta.get("split_key"))
        if group_key is not None:
            check_split_groups(db, group_key)
        if getattr(args, "drop_uids", None):
            # Quality filtering after encoding (ASR CER / DNSMOS tails): a JSON list or one uid per line.
            text = Path(args.drop_uids).read_text()
            uids = json.loads(text) if text.lstrip().startswith("[") else [u.strip() for u in text.splitlines() if u.strip()]
            db.executemany("DELETE FROM samples WHERE uid=?", [(u,) for u in uids])
            dropped_uids = len(uids)
        else:
            dropped_uids = 0
        before = db.total_changes
        if not getattr(args, "keep_singletons", False):
            # Cross-utterance pairing needs a second recording; within-utterance prompting does not.
            db.execute(
                "DELETE FROM samples WHERE speaker IN "
                "(SELECT speaker FROM samples GROUP BY speaker HAVING count(*)<2)"
            )
        singletons = db.total_changes - before
        db.execute("DELETE FROM provenance WHERE uid NOT IN (SELECT uid FROM samples)")
        db.execute("DELETE FROM text_tokens WHERE uid NOT IN (SELECT uid FROM samples)")
        db.execute("DELETE FROM token_ids WHERE uid NOT IN (SELECT uid FROM samples)")
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
        # A merged cache must not present the first rank's timing as aggregate throughput.
        meta.pop("preparation", None)
        meta.update(
            {
                "complete": True,
                "merged": True,
                "split_counts": splits,
                "accepted": sum(splits.values()),
                "rejected": rejected,
                "duplicates_removed": duplicates,
                "conflicting_duplicates_dropped": conflicts,
                "singleton_rows_removed": singletons,
                "dropped_uids": dropped_uids,
                "drop_uids_file": str(Path(args.drop_uids).resolve()) if getattr(args, "drop_uids", None) else None,
                "singletons_kept": bool(getattr(args, "keep_singletons", False)),
                "inputs": [str(Path(p).resolve()) for p in args.inputs],
                "index_sha256": file_digest(out / "index.sqlite"),
                "preparation_partitions": preparation_partitions,
            }
        )
        if any(summary for summary, _ in source_summaries):
            # Per-source licenses for attribution; counts are before merge-time drops (duplicates, --drop-uids,
            # singletons), which the index cannot attribute to a source.
            meta["sources"] = merge_source_summaries(
                summary or {UNRECORDED_SOURCE: {"licenses": [], "prepared_rows": accepted}}
                for summary, accepted in source_summaries
            )
        if resplit:
            meta.update(
                split_key=split_key.pattern if split_key else None,
                split_map=str(Path(args.split_map).resolve()) if split_map is not None else None,
                split_map_sha256=file_digest(args.split_map) if split_map is not None else None,
                split_version="split_map_v1" if split_map is not None else "split_key_sha256_98_1_1_v1",
                split_reassigned_rows=reassigned,
            )
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
    finally:
        db.close()
