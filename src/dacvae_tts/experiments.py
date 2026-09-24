"""Read-only cache audit, codec reconstruction, frozen evaluation cases and bounded sweeps."""

import itertools
import json
import math
import random
import sqlite3
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from .codec import Codec, backend_options, check_compatibility, file_digest, read_audio
from .data import LatentDataset, jsonl, load_stats
from .inference import Synthesizer
from .metrics import Evaluator, summarize
from .speakers import read_speaker_list
from .text import normalize


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


def audit_cache(args):
    root = Path(args.cache).resolve()
    meta = json.loads((root / "metadata.json").read_text())
    with sqlite3.connect(f"file:{root / 'index.sqlite'}?mode=ro", uri=True) as db:
        speakers = db.execute(
            "SELECT speaker,split,count(*),sum(samples) FROM samples GROUP BY speaker,split ORDER BY count(*) DESC"
        ).fetchall()
        total = sum(row[2] for row in speakers)
        leakage = db.execute(
            "SELECT speaker FROM samples GROUP BY speaker HAVING count(DISTINCT split)>1"
        ).fetchall()
        duplicates = db.execute(
            "SELECT digest,count(*) FROM samples GROUP BY digest HAVING count(*)>1"
        ).fetchall()
        has_provenance = db.execute("SELECT 1 FROM sqlite_master WHERE name='provenance'").fetchone()
        overlaps, invalid_intervals, session_rows = [], [], 0
        if has_provenance:
            session_rows = db.execute(
                "SELECT count(*) FROM provenance WHERE session IS NOT NULL AND session!=''"
            ).fetchone()[0]
            previous_source, active = None, []
            # Sweep intervals rather than a potentially quadratic whole-table SQL self-join.
            for source, uid, start, end in db.execute(
                "SELECT source_recording,uid,start_seconds,end_seconds FROM provenance WHERE source_recording IS NOT NULL AND start_seconds IS NOT NULL AND end_seconds IS NOT NULL ORDER BY source_recording,start_seconds"
            ):
                if not math.isfinite(start) or not math.isfinite(end) or end <= start or start < 0:
                    invalid_intervals.append(uid)
                    continue
                active = [
                    (other, stop) for other, stop in active if source == previous_source and stop > start
                ]
                overlaps.extend([other, uid] for other, stop in active)
                active.append((uid, end))
                previous_source = source
        errors = []
        if args.scan_latents:
            channels = meta["latent_dim"]
            previous, mapped = None, None
            for uid, shard, offset, frames in db.execute(
                "SELECT uid,shard,offset,frames FROM samples ORDER BY shard,offset"
            ):
                try:
                    if shard != previous:
                        mapped = np.memmap(shard, dtype="<f2", mode="r").reshape(-1, channels)
                        previous = shard
                    z = mapped[offset : offset + frames]
                    if offset < 0 or z.shape != (frames, channels) or frames < 1 or not np.isfinite(z).all():
                        errors.append(uid)
                except (OSError, ValueError):
                    errors.append(uid)
    stats = load_stats(root)
    report = {
        "rows": total,
        "speakers": len({r[0] for r in speakers}),
        "hours": sum(r[3] for r in speakers) / meta["sample_rate"] / 3600,
        "top_speakers": [
            {"speaker": r[0], "split": r[1], "rows": r[2], "fraction": r[2] / max(total, 1)}
            for r in speakers[:20]
        ],
        "cross_split_speakers": leakage,
        "duplicate_recordings": duplicates,
        "overlapping_clip_pairs": overlaps,
        "invalid_intervals": invalid_intervals,
        "session_metadata_rows": session_rows,
        "corrupt_latent_rows": errors,
        "latent_scan_performed": args.scan_latents,
        "minimum_std": float(stats["std"].min()),
        "index_sha256": file_digest(root / "index.sqlite"),
        "transcript_correctness": "not established; review original-audio ASR disagreements and listen",
        "near_duplicate_detection": "not implemented; exact waveform hashes and supplied interval metadata only",
        "filters": meta.get("filtering"),
        "removed_duplicates": meta.get("duplicates_removed"),
        "removed_singletons": meta.get("singleton_rows_removed"),
        "rejected_rows": meta.get("rejected"),
    }
    report["verified_integrity_checks_passed"] = not (
        leakage or duplicates or overlaps or invalid_intervals or errors
    )
    write_report(args.output, report)
    print(json.dumps(report, indent=2))


def codec_reconstruct(args):
    root = Path(args.cache)
    meta = json.loads((root / "metadata.json").read_text())
    stats = load_stats(root)
    codec = Codec(
        meta["checkpoint"], args.device, loudness=meta.get("loudness_lufs"), **backend_options(args)
    )
    check_compatibility(codec.metadata, meta)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(args.manifest).resolve().parent
    with (
        open(output / "original.jsonl", "x") as original,
        open(output / "reconstructed.jsonl", "x") as reconstructed,
    ):
        for index, row in enumerate(jsonl(args.manifest)):
            if index >= args.limit:
                break
            path = Path(row["audio"])
            path = path if path.is_absolute() else source_root / path
            audio = read_audio(path, codec.sample_rate, codec.loudness)
            decoded, diagnostics = codec.reconstruct(audio, stats["mean"], stats["std"], args.cache_precision)
            orig_path, recon_path = output / f"{index:06d}-original.wav", output / f"{index:06d}-codec.wav"
            sf.write(orig_path, audio.numpy(), codec.sample_rate, subtype="FLOAT")
            sf.write(recon_path, decoded.numpy(), codec.sample_rate, subtype="FLOAT")
            row = {
                **row,
                "uid": str(row.get("id", index)),
                "text": normalize(
                    row["text"], meta.get("text_normalization", "unicode-v1"), row.get("spoken_text")
                ),
                "reference_audio": str(orig_path),
                "codec": codec.metadata,
                "cache_precision": args.cache_precision,
                "diagnostics": diagnostics,
            }
            original.write(json.dumps({**row, "audio": str(orig_path)}) + "\n")
            reconstructed.write(json.dumps({**row, "audio": str(recon_path)}) + "\n")


def make_cases(args):
    data = LatentDataset(args.cache, args.split, args.seed)
    db = data._connection()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    has_provenance = db.execute("SELECT 1 FROM sqlite_master WHERE name='provenance'").fetchone()
    if db.execute(
        "SELECT speaker FROM samples GROUP BY speaker HAVING count(DISTINCT split)>1 LIMIT 1"
    ).fetchone():
        raise ValueError("Speaker leakage across splits; repair the cache before exporting cases")
    # Held-out labels whose voice also appears in train (speaker_clusters.py leakage.json) are not unseen.
    exclude_file = getattr(args, "exclude_speakers", None)
    excluded = read_speaker_list(exclude_file) if exclude_file else set()
    rows, seen = [], set()
    seen_digests, selected_intervals, intervals = set(), [], {}

    def interval(uid):
        if uid not in intervals:
            row = (
                db.execute(
                    "SELECT source_recording,start_seconds,end_seconds FROM provenance WHERE uid=?", (uid,)
                ).fetchone()
                if has_provenance
                else None
            )
            if row and all(v is not None for v in row):
                if not math.isfinite(row[1]) or not math.isfinite(row[2]) or not 0 <= row[1] < row[2]:
                    raise ValueError(f"Invalid source interval for {uid}")
                intervals[uid] = row
            else:
                intervals[uid] = None
        return intervals[uid]

    def overlaps(a, b):
        return a is not None and b is not None and a[0] == b[0] and a[1] < b[2] and b[1] < a[2]

    # Exhaustive search is opt-in bounded by the split's index; export only complete original audio paths.
    order = np.random.default_rng(args.seed).permutation(len(data))
    for index in order:
        uid = db.execute("SELECT uid FROM samples WHERE id=?", (int(data.ids[index]),)).fetchone()[0]
        target = db.execute(
            "SELECT uid,speaker,text,audio,frames,digest FROM samples WHERE uid=?", (uid,)
        ).fetchone()
        if target[1] in excluded:
            continue
        candidates = db.execute(
            "SELECT uid,text,audio,digest FROM samples WHERE speaker=? AND split=? AND uid!=? ORDER BY uid",
            (target[1], args.split, uid),
        ).fetchall()
        if args.cross_session:
            if not has_provenance:
                raise ValueError("Cross-session cases require session metadata in the cache")
            session = db.execute("SELECT session FROM provenance WHERE uid=?", (uid,)).fetchone()
            session = session[0] if session else None
            if not session:
                continue
            candidates = [
                r
                for r in candidates
                if (lambda s: s and s[0] and s[0] != session)(
                    db.execute("SELECT session FROM provenance WHERE uid=?", (r[0],)).fetchone()
                )
            ]
        candidates = [
            r
            for r in candidates
            if Path(r[2]).is_file()
            and r[3] != target[5]
            and r[0] not in seen
            and r[3] not in seen_digests
            and not overlaps(interval(uid), interval(r[0]))
            and not any(overlaps(interval(r[0]), old) for old in selected_intervals)
        ]
        if (
            not candidates
            or not Path(target[3]).is_file()
            or uid in seen
            or target[5] in seen_digests
            or any(overlaps(interval(uid), old) for old in selected_intervals)
        ):
            continue
        # No recording is reused as another case's target or reference.
        ref = candidates[random.Random(args.seed + int(index)).randrange(len(candidates))]
        seen.update((uid, ref[0]))
        seen_digests.update((target[5], ref[3]))
        selected_intervals.extend((interval(uid), interval(ref[0])))
        rows.append(
            {
                "uid": uid,
                "reference_uid": ref[0],
                "speaker": target[1],
                "split": args.split,
                "text": target[2],
                "reference_text": ref[1],
                "reference_audio": ref[2],
                "target_audio": target[3],
                "ground_truth_frames": target[4],
                "ground_truth_seconds": target[4] * data.meta["hop_length"] / data.meta["sample_rate"],
                "tags": [],
                "cross_session": args.cross_session,
            }
        )
        if len(rows) >= args.limit:
            break
    if not rows:
        raise ValueError(
            "No eligible original-audio pairs; supply accessible paths/session metadata or author cases JSONL explicitly"
        )
    with open(output, "x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    exclusion = (
        {
            "speakers": len({row["speaker"] for row in rows}),
            "excluded_speakers": len(excluded),
            "exclude_speakers_file": str(Path(exclude_file).resolve()),
            "exclude_speakers_sha256": file_digest(exclude_file),
        }
        if exclude_file
        else {}
    )
    write_report(
        output.with_suffix(".metadata.json"),
        {
            "cases": len(rows),
            "seed": args.seed,
            "index_sha256": file_digest(data.db_path),
            "cases_sha256": file_digest(output),
            "reference_target_reuse": False,
            "known_interval_overlap": False,
            "unknown_intervals_and_near_duplicates": "not verified",
            **exclusion,
        },
    )


def run_eval(args):
    config = yaml.safe_load(Path(args.config).read_text())
    cases = list(itertools.islice(jsonl(config["cases"]), config.get("max_cases", 32)))
    if not cases:
        raise ValueError("No evaluation cases")
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tts = Synthesizer(
        config["checkpoint"],
        config.get("device", "cuda"),
        config.get("precision", "bf16"),
        profile=True,
        codec_options=config.get("codec_options"),
    )
    if "reference_paths" in config:
        if config["reference_paths"] not in {"both", "full", "summary"}:
            raise ValueError("Unsupported reference path ablation")
        tts.model.cfg.reference_paths = config["reference_paths"]
    judge = config.get("judges")
    evaluator = Evaluator(**judge) if judge else None
    sampling = config["sampling"]
    variants = list(
        itertools.product(
            sampling["steps"],
            sampling["guidance"],
            sampling["sway"],
            sampling["duration"],
            sampling["duration_scale"],
            sampling["seed"],
        )
    )
    references = {}
    cases_sha256 = file_digest(config["cases"])
    for steps, guidance, sway, duration_mode, scale, seed in variants:
        if duration_mode not in {"predicted", "ground_truth"}:
            raise ValueError("duration must be predicted or ground_truth")
        name = f"n{steps}-g{guidance}-s{sway}-d{duration_mode}-x{scale}-seed{seed}"
        run = output / name
        run.mkdir(exist_ok=True)
        results = []
        with open(run / "manifest.jsonl", "x") as stream:
            for index, case in enumerate(cases):
                key = (case["reference_audio"], case["reference_text"])
                reused = key in references
                if not reused:
                    references[key] = tts.prepare_reference(*key)
                ref = references[key]
                seconds = case["ground_truth_seconds"] if duration_mode == "ground_truth" else None
                result = tts.synthesize(
                    case["text"],
                    reference=ref,
                    output=run / f"{index:06d}.wav",
                    seconds=seconds,
                    duration_scale=scale,
                    steps=steps,
                    guidance=guidance,
                    seed=seed,
                    sway=sway,
                )
                row = {
                    **case,
                    **result.metadata,
                    "audio": str(run / f"{index:06d}.wav"),
                    "duration_mode": duration_mode,
                    "duration_scale": scale,
                    "candidate_count": 1,
                    "selection": "none",
                    "reference_cache_hit": reused,
                    "duration_absolute_error_seconds": abs(
                        result.metadata["audio_seconds"] - case["ground_truth_seconds"]
                    ),
                    "duration_relative_error": abs(
                        result.metadata["audio_seconds"] / case["ground_truth_seconds"] - 1
                    ),
                    "cases_sha256": cases_sha256,
                    "checkpoint": str(Path(config["checkpoint"]).resolve()),
                    "reference_preparation_charged_seconds": 0 if reused else sum(ref.timings.values()),
                }
                if evaluator:
                    row.update(evaluator.score(row["audio"], row["text"], row["reference_audio"]))
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                results.append(row)
        summary = (
            summarize(results) if evaluator else {"count": len(results), "quality_metrics": "not scored"}
        )
        summary.update(
            duration_mae_seconds=float(np.mean([r["duration_absolute_error_seconds"] for r in results])),
            mean_generation_rtf=float(np.mean([r["rtf"] for r in results])),
            candidate_count=1,
            selection="none",
            config=config,
            variant=name,
        )
        write_report(run / "summary.json", summary)
    write_report(output / "run_config.json", config)
