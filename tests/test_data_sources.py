"""Turkish source preparers (scripts/data/) on tiny synthetic YODAS / Common Voice / ISSAI TSC layouts, and the
manifest -> transcribe_corpus uid -> prepare -> merge --drop-uids path."""

import io
import json
import sqlite3
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from test_manifest_pipeline import FREYA, freya_file, manifest_rows, noise, sentence, wav_bytes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "data"))
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_common_voice  # noqa: E402
import prepare_issai_tsc  # noqa: E402
import prepare_yodas  # noqa: E402


def add_file(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def test_yodas_utterance_shards(tmp_path):
    local = tmp_path / "yodas"
    utts = {
        "Y1RQ3T-lVN8-00000-00000001-00000258": "Herkese merhaba ben &quot;Sümeyra&quot;.",
        "Y1RQ3T-lVN8-00001-00000258-00000658": sentence(1, 5),
        "Y1RQ3T-lVN8-00002-00000661-00000900": "[Müzik]",
        "Yabc_def123-00000-00000100-00000400": FREYA[0],
    }
    (local / "data/tr000/text").mkdir(parents=True)
    (local / "data/tr000/duration").mkdir(parents=True)
    (local / "data/tr000/audio").mkdir(parents=True)
    (local / "data/tr000/text/00000000.txt").write_text("".join(f"{k} {v}\n" for k, v in utts.items()))
    durations = {k: (int(k.rsplit("-", 1)[1]) - int(k.rsplit("-", 2)[1])) / 100 for k in utts}
    (local / "data/tr000/duration/00000000.txt").write_text("".join(f"{k} {v:.2f}\n" for k, v in durations.items()))
    with tarfile.open(local / "data/tr000/audio/00000000.tar.gz", "w:gz") as tar:
        for i, (utt, seconds) in enumerate(durations.items()):
            add_file(tar, f"00000000/{utt}.wav", wav_bytes(noise(seconds, seed=i)))
    out = tmp_path / "out"
    summary = prepare_yodas.main([
        "--variant", "yodas", "--subset", "tr000", "--local-dir", str(local), "--shards", "all",
        "--output", str(out), "--freya-sentences", str(freya_file(tmp_path)), "--cps-iqr-k", "0",
    ])
    assert summary["removed"] == {"caption_markup": 1, "freya_exact": 1}
    assert summary["origin"]["shards"] == [0] and summary["license"] == "CC-BY-3.0"
    rows = manifest_rows(out)
    assert [r["id"] for r in rows] == ["yodas/tr000/Y1RQ3T-lVN8-00000-00000001-00000258",
                                       "yodas/tr000/Y1RQ3T-lVN8-00001-00000258-00000658"]
    assert rows[0]["text"] == 'Herkese merhaba ben "Sümeyra".'  # HTML entities unescaped
    assert {r["speaker"] for r in rows} == {"yodas/Y1RQ3T-lVN8"}  # video ids may contain '-'
    assert (rows[1]["start_seconds"], rows[1]["end_seconds"], rows[1]["session_id"]) == (2.58, 6.58, "Y1RQ3T-lVN8")
    assert (local / "data/tr000/audio/00000000.tar.gz").exists()  # --local-dir files are never deleted


def longform_audio(seconds, sample_rate=24000):
    return noise(seconds, sample_rate=sample_rate, seed=5)


def test_yodas2_longform_segments(tmp_path):
    local = tmp_path / "yodas2"
    audio = longform_audio(12.0)
    utterances = {"Vid-eo1-00000-00000050-00000350": sentence(0, 5), "Vid-eo1-00001-00000400-00000460": "Evet.",
                  "Vid-eo1-00002-00000500-00000900": sentence(2, 6)}
    (local / "data/tr000/text").mkdir(parents=True)
    (local / "data/tr000/audio").mkdir(parents=True)
    (local / "data/tr000/text/00000000.json").write_text(json.dumps([{"audio_id": "Vid-eo1", "text": utterances}]))
    with tarfile.open(local / "data/tr000/audio/00000000.tar.gz", "w:gz") as tar:
        add_file(tar, "Vid-eo1.flac", wav_bytes(audio, 24000, "FLAC"))
    out = tmp_path / "out"
    summary = prepare_yodas.main([
        "--subset", "tr000", "--local-dir", str(local), "--shards", "0", "--speaker-mode", "utterance",
        "--output", str(out), "--freya-sentences", str(freya_file(tmp_path)), "--cps-iqr-k", "0",
    ])
    assert summary["removed"] == {"duration": 1}  # the 0.6 s "Evet." is cut before decoding
    rows = manifest_rows(out)
    assert [r["id"] for r in rows] == ["yodas2/tr000/Vid-eo1-00000-00000050-00000350",
                                       "yodas2/tr000/Vid-eo1-00002-00000500-00000900"]
    segment, sample_rate = sf.read(io.BytesIO(rows[0]["audio"]["bytes"]), dtype="float32")
    assert sample_rate == 24000 and len(segment) == 3.0 * 24000
    np.testing.assert_allclose(segment, audio[int(0.5 * 24000) : int(3.5 * 24000)], atol=1e-4)  # 16-bit FLAC
    assert rows[1]["speaker"] == "yodas/Vid-eo1-00002-00000500-00000900"
    assert rows[1]["duration_seconds"] == pytest.approx(4.0)


def test_long_recording_spans_seek_or_decode(monkeypatch):
    blob = wav_bytes(longform_audio(6.0), 24000, "FLAC")
    spans = [(0.5, 2.0), (5.5, 7.0), (3.0, 3.0)]  # the second runs past the end of the recording
    seeked = prepare_yodas.read_spans(blob, spans)
    monkeypatch.setattr(prepare_yodas.sf.SoundFile, "seekable", lambda self: False)
    decoded = prepare_yodas.read_spans(blob, spans)
    assert [len(a) for a, _ in seeked] == [36000, 12000, 0] == [len(a) for a, _ in decoded]
    for (a, rate_a), (b, rate_b) in zip(seeked, decoded):
        assert rate_a == rate_b == 24000
        np.testing.assert_array_equal(a, b)


def test_sidon_webdataset_pairs(tmp_path):
    local = tmp_path / "sidon"
    (local / "tr000").mkdir(parents=True)
    columnar = {"video_id": "VidA", "utterances": {"utt_id": ["VidA-00000-00000000-00000300"], "text": [sentence(0, 5)],
                                                   "start": [0.0], "end": [3.0]}}
    rows_format = {"video_id": "VidB", "utterances": [{"utt_id": "VidB-00000-00000100-00000400",
                                                        "text": FREYA[1], "start": 1.0, "end": 4.0}]}
    with tarfile.open(local / "tr000/train-00000.tar.gz", "w:gz") as tar:
        add_file(tar, "000001.metadata.json", json.dumps(columnar).encode())  # metadata before audio is fine
        add_file(tar, "000001.flac", wav_bytes(longform_audio(4.0), 24000, "FLAC"))
        add_file(tar, "000002.flac", wav_bytes(longform_audio(5.0), 24000, "FLAC"))
        add_file(tar, "000002.metadata.json", json.dumps(rows_format).encode())
    out = tmp_path / "out"
    summary = prepare_yodas.main([
        "--variant", "sidon", "--subset", "tr000", "--local-dir", str(local), "--shards", "all",
        "--output", str(out), "--freya-sentences", str(freya_file(tmp_path)), "--cps-iqr-k", "0",
    ])
    assert summary["removed"] == {"freya_exact": 1} and summary["kept"] == 1
    (row,) = manifest_rows(out)
    assert row["id"] == "sidon/tr000/VidA-00000-00000000-00000300" and row["speaker"] == "yodas/VidA"
    assert row["sample_rate"] == 24000 and row["source"] == "sidon-tr000"


def test_yodas_refuses_evaluation_sets(tmp_path):
    with pytest.raises(ValueError, match="evaluation set"):
        prepare_yodas.main(["--repo", "google/fleurs", "--shards", "0", "--output", str(tmp_path / "o"),
                            "--no-freya-check"])


def write_common_voice(tmp_path):
    cv = tmp_path / "cv-corpus-27.0-2026-09-01" / "tr"
    (cv / "clips").mkdir(parents=True)
    header = "client_id\tpath\tsentence_id\tsentence\tsentence_domain\tup_votes\tdown_votes\tage\tgender\taccents\t" \
             "variant\tlocale\tsegment\n"
    lines, durations = [], ["clip\tduration[ms]\n"]
    texts = [sentence(i, 4 + i % 3) for i in range(8)] + ['Ona "tamam" dedim ve çıktım.', FREYA[0], sentence(9, 5)]
    for i, text in enumerate(texts):
        clip = f"common_voice_tr_{i}.wav"
        seconds = 2.0 + 0.1 * i
        (cv / "clips" / clip).write_bytes(wav_bytes(noise(seconds, 48000, seed=i), 48000))
        down = 1 if i == 10 else 0
        lines.append(f"client{i % 3}\t{clip}\ts{i}\t{text}\t\t2\t{down}\t\t\t\t\ttr\t\n")
        durations.append(f"{clip}\t{int(seconds * 1000)}\n")
    (cv / "validated.tsv").write_text(header + "".join(lines))
    (cv / "clip_durations.tsv").write_text("".join(durations))
    return cv


def test_common_voice_release(tmp_path):
    cv = write_common_voice(tmp_path)
    out = tmp_path / "out"
    summary = prepare_common_voice.main(["--cv-dir", str(cv), "--output", str(out),
                                         "--freya-sentences", str(freya_file(tmp_path)), "--cps-iqr-k", "0"])
    assert summary["removed"] == {"freya_exact": 1, "cv_down_votes": 1}
    assert summary["origin"]["release"] == "cv-corpus-27.0-2026-09-01" and summary["license"] == "CC0-1.0"
    rows = manifest_rows(out)
    assert len(rows) == 9 and {r["speaker"] for r in rows} == {"cv/client0", "cv/client1", "cv/client2"}
    quoted = next(r for r in rows if r["id"] == "common-voice/tr/common_voice_tr_8")
    assert quoted["text"] == 'Ona "tamam" dedim ve çıktım.'  # unquoted TSV: '"' is text
    assert json.loads(quoted["source_meta"])["up_votes"] == "2"
    assert quoted["sample_rate"] == 48000 and quoted["audio"]["bytes"] == (cv / "clips" / "common_voice_tr_8.wav").read_bytes()


def test_issai_tsc_folders(tmp_path):
    root = tmp_path / "ISSAI_TSC_218"
    for split, count in (("Train", 4), ("Dev", 2), ("Test", 2)):
        (root / split).mkdir(parents=True)
        for i in range(count):
            stem = f"rec{i % 2}_{split.lower()}{i}"
            sf.write(root / split / f"{stem}.wav", noise(1.5 + 0.1 * i, seed=i), 16000)
            if not (split == "Dev" and i == 1):
                (root / split / f"{stem}.txt").write_text(sentence(i, 4) + "\n")
    out = tmp_path / "out"
    summary = prepare_issai_tsc.main(["--root", str(root), "--output", str(out), "--speaker-regex", r"^(rec\d)_",
                                      "--freya-sentences", str(freya_file(tmp_path)), "--cps-iqr-k", "0"])
    assert summary["removed"] == {"missing_transcript": 1} and summary["license"] == "MIT"
    rows = manifest_rows(out)
    assert len(rows) == 5 and not any("/Test/" in r["id"] for r in rows)  # Test is held out by default
    assert {r["speaker"] for r in rows} == {"tsc/rec0", "tsc/rec1"}
    assert rows[0]["id"] == "issai-tsc/Train/rec0_train0"


def test_manifest_prepares_and_merges_with_explicit_uids(tmp_path, monkeypatch):
    """The manifest is read by `prepare` unchanged; uids survive to `merge --drop-uids` via transcribe_corpus."""
    from test_prepare_fast import ToyCodec, args_for
    from transcribe_corpus import row_audio, row_uid

    import dacvae_tts.prepare as module

    cv = write_common_voice(tmp_path)
    out = tmp_path / "cv"
    prepare_common_voice.main(["--cv-dir", str(cv), "--output", str(out), "--freya-sentences",
                               str(freya_file(tmp_path)), "--cps-iqr-k", "0"])
    rows = manifest_rows(out)
    part = sorted((out / "manifest").glob("*.parquet"))[0]
    assert row_uid("data/x.parquet", 0, rows[0]) == rows[0]["id"]
    assert row_uid("data/x.parquet", 3, {"text": "t"}) == "data/x.parquet:3"
    assert row_audio(rows[0], part.parent) == rows[0]["audio"]["bytes"]
    codec = ToyCodec()
    monkeypatch.setattr(module, "Codec", lambda *args, **kwargs: codec)
    cache = tmp_path / "cache"
    module.prepare(args_for(out / "manifest", cache, speaker_column="speaker", text_normalization="turkish-v1",
                            languages={"tr"}, min_seconds=1.0, max_seconds=20.0, batch_seconds=60, loudness=-16))
    meta = json.loads((cache / "metadata.json").read_text())
    assert (meta["accepted"], meta["rejected"]) == (len(rows), 0)
    drop = tmp_path / "drop.json"
    drop.write_text(json.dumps([rows[0]["id"]]))
    merged = tmp_path / "merged"
    module.merge(SimpleNamespace(inputs=[cache], output=merged, drop_uids=str(drop), keep_singletons=True))
    with sqlite3.connect(merged / "index.sqlite") as db:
        uids = {uid for (uid,) in db.execute("SELECT uid FROM samples")}
        splits = {split for (split,) in db.execute("SELECT DISTINCT split FROM samples")}
    assert uids == {r["id"] for r in rows[1:]} and splits <= {"train", "val", "test"}
