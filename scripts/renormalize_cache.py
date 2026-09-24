"""Re-normalize the transcripts of a prepared latent cache with another text normalization, without re-encoding audio.

  python scripts/renormalize_cache.py --cache data/cache-tr --output data/cache-tr-v2 --text-normalization turkish-v2

Only the index moves: samples.shard holds absolute paths into the source partitions, so the new directory gets a copy
of index.sqlite, metadata.json and stats.pt and keeps reading the same latent shards. Every row is normalized again
from provenance.original_text, the transcript as it was before normalization. A row whose stored text did not come
from original_text with the cache's recorded version (a spoken_text override, a hand edit) or that has no provenance
is normalized from samples.text instead, and both cases are counted. samples.text, provenance.normalized_text,
text_tokens.utf8 and token_ids.ids are rewritten and metadata.json records the new `text_normalization`.

Rows the new version rejects are dropped and listed in OUTPUT/rejected.jsonl; a speaker left with a single utterance
by those drops loses it too (cross-utterance pairing needs two; --keep-singletons, or a source merged with
--keep-singletons, keeps it). stats.pt is copied byte for byte unless train rows were dropped, in which case the
train statistics are recomputed from the retained rows as `merge` does. The source cache is opened read-only.
"""

import argparse
import json
import shutil
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from dacvae_tts.codec import file_digest
from dacvae_tts.data import save_stats
from dacvae_tts.text import TEXT_VERSIONS, encode_ids, normalize


def train_statistics(db, channels):
    sums = torch.zeros(channels, dtype=torch.float64)
    squares, count = sums.clone(), 0
    previous, mapped = None, None
    for shard, offset, frames in db.execute(
        "SELECT shard,offset,frames FROM samples WHERE split='train' ORDER BY shard,offset"
    ):
        if shard != previous:
            mapped = np.memmap(shard, dtype="<f2", mode="r").reshape(-1, channels)
            previous = shard
        z = torch.from_numpy(np.array(mapped[offset : offset + frames], dtype=np.float64))
        sums += z.sum(0)
        squares += z.square().sum(0)
        count += frames
    return count, sums, squares


def renormalize(cache, output, version, keep_singletons=None, examples=10, batch=10000):
    if version not in TEXT_VERSIONS:
        raise ValueError(f"Unknown normalization version: {version}")
    source, out = Path(cache).resolve(), Path(output).resolve()
    meta = json.loads((source / "metadata.json").read_text())
    if not meta.get("complete"):
        raise ValueError(f"Incomplete cache: {source}")
    if out == source or (out.exists() and any(out.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    previous_version = meta.get("text_normalization", "unicode-v1")
    keep_singletons = meta.get("singletons_kept", False) if keep_singletons is None else keep_singletons
    src = sqlite3.connect(f"{(source / 'index.sqlite').as_uri()}?mode=ro", uri=True)
    db = sqlite3.connect(out / "index.sqlite")
    counts, rejected, changed_examples, dropped = Counter(), [], [], []
    try:
        src.backup(db)
        db.execute("CREATE TABLE IF NOT EXISTS text_tokens (uid TEXT PRIMARY KEY, utf8 BLOB NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS token_ids (uid TEXT PRIMARY KEY, ids BLOB NOT NULL)")
        has_provenance = src.execute("SELECT 1 FROM sqlite_master WHERE name='provenance'").fetchone() is not None
        query = (
            "SELECT s.uid,s.speaker,s.split,s.text,p.original_text FROM samples s LEFT JOIN provenance p ON p.uid=s.uid"
            if has_provenance
            else "SELECT uid,speaker,split,text,NULL FROM samples"
        )
        pending = []

        def flush():
            db.executemany("UPDATE samples SET text=? WHERE uid=?", [(t, u) for u, t, _, _ in pending])
            if has_provenance:
                db.executemany(
                    "UPDATE provenance SET normalized_text=? WHERE uid=?", [(t, u) for u, t, _, _ in pending]
                )
            db.executemany("INSERT OR REPLACE INTO text_tokens VALUES (?,?)", [(u, b) for u, _, b, _ in pending])
            db.executemany("INSERT OR REPLACE INTO token_ids VALUES (?,?)", [(u, i) for u, _, _, i in pending])
            pending.clear()

        for uid, speaker, split, text, original in src.execute(query):
            counts["rows"] += 1
            origin = "no_provenance"
            if original is not None:
                try:
                    origin = "original_text" if normalize(original, previous_version) == text else "stored_text"
                except ValueError:
                    origin = "stored_text"
            counts[f"from_{origin}"] += 1
            try:
                new = normalize(original if origin == "original_text" else text, version)
            except ValueError as exc:
                rejected.append({"uid": uid, "speaker": speaker, "split": split, "text": text, "error": str(exc)})
                dropped.append((uid, speaker, split))
                continue
            if new != text:
                counts["changed"] += 1
                if len(changed_examples) < examples:
                    changed_examples.append({"uid": uid, "before": text, "after": new})
            utf8 = new.encode("utf-8")
            pending.append((uid, new, utf8, encode_ids(utf8).tobytes()))
            if len(pending) >= batch:
                flush()
        flush()

        def delete(rows):
            for table in ("samples", "provenance", "text_tokens", "token_ids"):
                if table != "provenance" or has_provenance:
                    db.executemany(f"DELETE FROM {table} WHERE uid=?", [(uid,) for uid, _, _ in rows])

        delete(dropped)
        singletons = []
        for speaker, split in sorted({(speaker, split) for _, speaker, split in dropped}):  # a speaker has one split
            remaining = db.execute("SELECT uid FROM samples WHERE split=? AND speaker=?", (split, speaker)).fetchall()
            if len(remaining) == 1 and not keep_singletons:
                singletons.append((remaining[0][0], speaker, split))
        delete(singletons)
        dropped += singletons
        counts["singleton_rows_removed"] = len(singletons)
        db.commit()
        if any(split == "train" for _, _, split in dropped):
            save_stats(out / "stats.pt", *train_statistics(db, meta["latent_dim"]))
            stats = "recomputed"
        else:
            shutil.copy2(source / "stats.pt", out / "stats.pt")
            stats = "copied"
        splits = dict(db.execute("SELECT split,count(*) FROM samples GROUP BY split").fetchall())
        shards = [shard for (shard,) in db.execute("SELECT DISTINCT shard FROM samples")]
    finally:
        src.close()
        db.close()
    with open(out / "rejected.jsonl", "w") as stream:
        for row in rejected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": str(source),
        "output": str(out),
        "source_text_normalization": previous_version,
        "text_normalization": version,
        "rows": counts["rows"],
        "changed": counts["changed"],
        "unchanged": counts["rows"] - counts["changed"] - len(rejected),
        "from_original_text": counts["from_original_text"],
        "from_stored_text": counts["from_stored_text"],
        "no_provenance": counts["from_no_provenance"],
        "rejected": len(rejected),
        "singleton_rows_removed": counts["singleton_rows_removed"],
        "retained": sum(splits.values()),
        "stats": stats,
        "shards": len(shards),
        "missing_shards": sum(not Path(shard).exists() for shard in shards),
    }
    meta.update(
        {
            "text_normalization": version,
            "split_counts": splits,
            "accepted": sum(splits.values()),
            "index_sha256": file_digest(out / "index.sqlite"),
            "renormalized": {key: value for key, value in summary.items() if key != "output"},
        }
    )
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return {**summary, "examples": changed_examples, "rejected_examples": rejected[:examples]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True, help="Prepared (usually merged) cache directory; never modified")
    parser.add_argument("--output", required=True, help="New directory for index.sqlite, metadata.json, stats.pt")
    parser.add_argument("--text-normalization", choices=sorted(TEXT_VERSIONS), default="turkish-v2")
    parser.add_argument(
        "--keep-singletons",
        action="store_true",
        default=None,
        help="Keep a speaker's last utterance when the others were rejected (default: as the source cache)",
    )
    parser.add_argument("--examples", type=int, default=10, help="Changed and rejected rows shown in the summary")
    args = parser.parse_args()
    summary = renormalize(args.cache, args.output, args.text_normalization, args.keep_singletons, args.examples)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["missing_shards"]:
        print(f"warning: {summary['missing_shards']} latent shards referenced by the index are missing", file=sys.stderr)


if __name__ == "__main__":
    main()
