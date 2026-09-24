"""Leak-free prompt sets: scripts/data/make_prompt_set.py and eval_sentences.py --prompt-set."""

import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clip(seconds, rate=48000, seed=0):
    rng = np.random.default_rng(seed)
    audio = (0.1 * rng.standard_normal(int(seconds * rate))).astype(np.float32)
    stream = io.BytesIO()
    sf.write(stream, audio, rate, format="WAV")
    return stream.getvalue()


def cv_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist([
        {"client_id": r["client"], "path": f"/x/{r['clip']}", "audio": {"bytes": clip(r["seconds"], seed=i), "path": r["clip"]},
         "sentence": r["sentence"], "up_votes": r.get("up", 2), "down_votes": r.get("down", 0),
         "gender": r.get("gender", "")}
        for i, r in enumerate(rows)
    ])
    pq.write_table(table, path)


def test_make_prompt_set_one_clean_clip_per_speaker(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS / "data"))
    module = load("make_prompt_set_under_test", SCRIPTS / "data" / "make_prompt_set.py")
    freya = tmp_path / "freya.jsonl"
    freya.write_text(json.dumps({"id": "0", "text": "Çocuğun kıvırcık saçları ve kocaman kara gözleri vardı."}) + "\n")
    rows = [
        # speaker a: the longest eligible clip wins (no DNSMOS: ranked by duration)
        dict(client="a" * 20, clip="a1.mp3", seconds=4.0, sentence="Bugün hava çok güzel ve güneşli.", gender="male_masculine"),
        dict(client="a" * 20, clip="a2.mp3", seconds=6.0, sentence="Yarın sabah erkenden yola çıkacağız.", gender="male_masculine"),
        # speaker b: its only clip reads a Freya sentence (other casing and punctuation): excluded
        dict(client="b" * 20, clip="b1.mp3", seconds=5.0, sentence="çocuğun kıvırcık saçları ve kocaman kara gözleri vardı",
             gender="female_feminine"),
        # speaker c: digits, a down-vote, too short, too few words: all excluded
        dict(client="c" * 20, clip="c1.mp3", seconds=5.0, sentence="Saat 5'te buluşalım mı acaba?", gender="female_feminine"),
        dict(client="c" * 20, clip="c2.mp3", seconds=5.0, sentence="Kitapları rafa geri koydum sonunda.", down=1),
        dict(client="c" * 20, clip="c3.mp3", seconds=2.0, sentence="Kapıyı kapatmayı unutma lütfen.", gender="female_feminine"),
        dict(client="c" * 20, clip="c4.mp3", seconds=5.0, sentence="Evet, tamam.", gender="female_feminine"),
        # speakers d and e: eligible female and untagged voices
        dict(client="d" * 20, clip="d1.mp3", seconds=5.0, sentence="Akşam yemeğinde balık pişirmeyi düşünüyorum.",
             gender="female_feminine"),
        dict(client="e" * 20, clip="e1.mp3", seconds=7.0, sentence="Kütüphanede sessizce kitap okumayı severim."),
    ]
    source = tmp_path / "cv"
    source.mkdir()
    cv_parquet(source / "test-00000.parquet", rows)
    summary = module.main(["--parquet", str(source), "--exclude-sentences", str(freya), "--speakers", "2",
                           "--output", str(tmp_path / "out")])
    prompts = json.loads((tmp_path / "out" / "prompts.json").read_text())
    assert summary["eligible_speakers"] == 3 and summary["removed"]["evaluation_sentence"] == 1
    assert summary["removed"] == {"evaluation_sentence": 1, "text": 2, "votes": 1, "duration": 1}
    # one male and one female voice first (gender balance), each with its own best clip
    assert [(p["speaker"], p["uid"], p["gender"]) for p in prompts] == [
        ("cv/" + "a" * 16, "a2.mp3", "male"), ("cv/" + "d" * 16, "d1.mp3", "female")]
    assert sf.info(str(tmp_path / "out" / "prompt-00.wav")).duration == pytest.approx(6.0)


def test_eval_sentences_reads_a_prompt_set(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    module = load("eval_sentences_prompt_set", SCRIPTS / "eval_sentences.py")
    sf.write(tmp_path / "p0.wav", np.zeros(48000, np.float32), 48000)
    sf.write(tmp_path / "p1.wav", np.zeros(48000, np.float32), 48000)
    entries = [{"audio": "p0.wav", "text": "Birinci ses.", "speaker": "cv/a", "uid": "clip-a"},
               {"audio": str(tmp_path / "p1.wav"), "text": "İkinci ses.", "speaker": "cv/b"}]
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(entries, ensure_ascii=False))
    cases = module.load_prompt_set(path)
    assert [c["prompt_uid"] for c in cases] == ["clip-a", "p1.wav"]
    assert cases[0]["prompt_audio"] == str(tmp_path / "p0.wav") and cases[1]["speaker"] == "cv/b"
    assert len(module.load_prompt_set(path, 1)) == 1
    # SIM-o's original recording is the prompt-set file itself, without --prompt-audio
    args = type("Args", (), {"prompt_audio": None})()
    find = module.original_prompt_finder(args, [(None, tmp_path / "prompt-00.wav", "cv/a")], cases[:1])
    assert find("prompt-00.wav") == tmp_path / "p0.wav" and find("prompt-01.wav") is None
    path.write_text(json.dumps(entries + [entries[0]]))
    with pytest.raises(SystemExit, match="unique"):
        module.load_prompt_set(path)
    path.write_text(json.dumps([{"audio": "missing.wav", "text": "x", "speaker": "s"}]))
    with pytest.raises(SystemExit, match="missing prompt recording"):
        module.load_prompt_set(path)
