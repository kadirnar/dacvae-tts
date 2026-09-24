import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest
import torch

from dacvae_tts.data import LatentDataset, load_stats
from dacvae_tts.text import decode_ids, encode_ids, normalize

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "renormalize_cache.py"
spec = importlib.util.spec_from_file_location("renormalize_cache", SCRIPT)
renormalize_cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renormalize_cache)

PLAIN = "Bugün hava çok güzel."
# uid -> (provenance.original_text or None for a row without provenance, samples.text as prepared with turkish-v1)
SPECIAL = {
    "train-0-0": ("Toplantı 4'e ertelendi.", "Toplantı dörte ertelendi."),
    "train-0-1": ("T.C. vatandaşıyım.", "T.C. vatandaşıyım."),
    "val-0-0": ("%4'ü geldi.", "yüzde dörtü geldi."),
    "train-1-0": ("4 kişi", "dört kişi geldi"),  # a spoken_text override: the stored text did not come from original
    "train-1-1": (None, "Dr. Ahmet geldi."),  # a row without provenance
    "train-2-0": ("Merhaba ☃", "Merhaba ☃"),  # the new version rejects it
    "train-3-0": ("Kardan adam ☃", "Kardan adam ☃"),
    "train-3-1": ("İki ☃", "İki ☃"),  # ... and speaker train-3 keeps a single utterance
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def turkish_cache(path, special=SPECIAL):
    """The synthetic merged cache of conftest, turned into a turkish-v1 cache with provenance and token ids."""
    with sqlite3.connect(path / "index.sqlite") as db:
        for (uid,) in db.execute("SELECT uid FROM samples").fetchall():
            original, text = special.get(uid, (PLAIN, normalize(PLAIN, "turkish-v1")))
            db.execute("UPDATE samples SET text=? WHERE uid=?", (text, uid))
            if original is not None:
                db.execute("INSERT INTO provenance VALUES (?,?,?,NULL,NULL,NULL,NULL)", (uid, original, text))
            db.execute("INSERT INTO text_tokens VALUES (?,?)", (uid, text.encode()))
            db.execute("INSERT INTO token_ids VALUES (?,?)", (uid, encode_ids(text.encode()).tobytes()))
    meta = json.loads((path / "metadata.json").read_text())
    meta["text_normalization"] = "turkish-v1"
    (path / "metadata.json").write_text(json.dumps(meta))
    return path


def test_renormalize_rewrites_text_tokens_and_metadata_without_touching_the_source(cache, tmp_path):
    source = turkish_cache(cache)
    before = {name: digest(source / name) for name in ("index.sqlite", "metadata.json", "stats.pt")}
    out = tmp_path / "cache-v2"
    summary = renormalize_cache.renormalize(source, out, "turkish-v2")

    assert {name: digest(source / name) for name in before} == before
    assert summary["rows"] == 36 and summary["rejected"] == 3 and summary["singleton_rows_removed"] == 1
    assert summary["changed"] == 4 and summary["retained"] == 32
    assert (summary["from_original_text"], summary["from_stored_text"], summary["no_provenance"]) == (31, 4, 1)
    assert {e["uid"]: e["after"] for e in summary["examples"]} == {
        "train-0-0": "Toplantı dörde ertelendi.",
        "train-0-1": "te ce vatandaşıyım.",
        "train-1-1": "Doktor Ahmet geldi.",
        "val-0-0": "yüzde dördü geldi.",
    }
    with sqlite3.connect(out / "index.sqlite") as db:
        texts = dict(db.execute("SELECT uid,text FROM samples"))
        assert texts["train-1-0"] == "dört kişi geldi"  # the override is kept, not replaced by "dört kişi"
        assert not {"train-2-0", "train-3-0", "train-3-1", "train-3-2"} & texts.keys()
        for uid, text, normalized, utf8, ids in db.execute(
            "SELECT s.uid,s.text,p.normalized_text,t.utf8,k.ids FROM samples s LEFT JOIN provenance p ON p.uid=s.uid "
            "JOIN text_tokens t ON t.uid=s.uid JOIN token_ids k ON k.uid=s.uid"
        ):
            assert normalized in (text, None) and utf8 == text.encode()
            assert decode_ids(ids).tolist() == encode_ids(text.encode()).tolist()
        for table in ("provenance", "text_tokens", "token_ids"):
            assert db.execute(f"SELECT count(*) FROM {table} WHERE uid NOT IN (SELECT uid FROM samples)").fetchone()[0] == 0
        train_frames = db.execute("SELECT sum(frames) FROM samples WHERE split='train'").fetchone()[0]
    rejected = [json.loads(line) for line in (out / "rejected.jsonl").read_text().splitlines()]
    assert sorted(r["uid"] for r in rejected) == ["train-2-0", "train-3-0", "train-3-1"]
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["text_normalization"] == "turkish-v2" and meta["renormalized"]["source_text_normalization"] == "turkish-v1"
    assert meta["index_sha256"] == digest(out / "index.sqlite")
    assert meta["split_counts"] == {"train": 8, "val": 12, "test": 12} and meta["renormalized"]["stats"] == "recomputed"
    stats = load_stats(out)
    assert stats["count"] == train_frames and not torch.equal(stats["mean"], load_stats(source)["mean"])
    data = LatentDataset(out, "train")
    item = data[0]
    assert len(data) == 8 and bytes((item["token_ids"][1:-1] - 4).astype("uint8").tolist()).decode() == item["text"]
    with pytest.raises(ValueError, match="new or empty"):
        renormalize_cache.renormalize(source, out, "turkish-v2")
    kept = renormalize_cache.renormalize(source, tmp_path / "kept", "turkish-v2", keep_singletons=True)
    assert kept["singleton_rows_removed"] == 0 and kept["retained"] == 33


def test_renormalize_without_drops_copies_stats_and_runs_from_the_command_line(cache, tmp_path, monkeypatch, capsys):
    source = turkish_cache(cache, {"train-0-0": SPECIAL["train-0-0"]})
    out = tmp_path / "cache-v2"
    monkeypatch.setattr(sys, "argv", ["renormalize_cache.py", "--cache", str(source), "--output", str(out)])
    renormalize_cache.main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["changed"] == 1 and summary["rejected"] == 0 and summary["stats"] == "copied"
    assert summary["missing_shards"] == 0 and summary["shards"] > 0
    assert digest(out / "stats.pt") == digest(source / "stats.pt")
    assert len(LatentDataset(out, "train")) == len(LatentDataset(source, "train")) == 12
