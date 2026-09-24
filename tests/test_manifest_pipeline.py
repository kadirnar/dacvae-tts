"""Shared manifest builder of the Turkish source preparers (scripts/data/manifest_pipeline.py) on synthetic clips."""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))

import manifest_pipeline  # noqa: E402

FREYA = ["İstanbul'da yarın öğleden sonra yağmur bekleniyor.", "Kâğıt kesiği çok acıtır."]
WORDS = "bugün pazara gidip taze sebze meyve aldım akşam yemeğini birlikte hazırlarız sonra çay içeriz".split()


def noise(seconds, sample_rate=16000, seed=0, cutoff=None):
    audio = np.random.default_rng(seed).normal(0, 0.1, int(seconds * sample_rate)).astype(np.float32)
    if cutoff:
        spectrum = np.fft.rfft(audio)
        spectrum[np.fft.rfftfreq(len(audio), 1 / sample_rate) > cutoff] = 0
        audio = np.fft.irfft(spectrum, len(audio)).astype(np.float32)
    return audio


def wav_bytes(audio, sample_rate=16000, format="WAV"):
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format=format)
    return buffer.getvalue()


def sentence(i, words):
    return " ".join(WORDS[(i + k) % len(WORDS)] for k in range(words)).capitalize() + "."


def freya_file(tmp_path):
    path = tmp_path / "freya_tr_eval.jsonl"
    path.write_text("".join(json.dumps({"id": i, "text": s}, ensure_ascii=False) + "\n" for i, s in enumerate(FREYA)))
    return path


def builder_args(tmp_path, *extra):
    parser = argparse.ArgumentParser()
    manifest_pipeline.add_arguments(parser)
    return parser.parse_args(
        ["--output", str(tmp_path / "out"), "--freya-sentences", str(freya_file(tmp_path)), "--workers", "2", *extra]
    )


def good_candidates(count, rate=12.0):
    """Clips whose duration follows the text at ~`rate` chars/s, as correctly aligned speech does."""
    from dacvae_tts.datafilters import text_characters

    rows = []
    for i in range(count):
        text = sentence(i, 3 + i % 4)
        seconds = max(1.1, text_characters(text) / rate * (1 + 0.04 * ((i % 5) - 2)))
        rows.append(dict(id=f"test/{i}", text=text, speaker=f"spk/{i % 6}", audio_bytes=wav_bytes(noise(seconds, seed=i))))
    return rows


def manifest_rows(output):
    return [row for part in sorted((Path(output) / "manifest").glob("*.parquet")) for row in pq.read_table(part).to_pylist()]


def test_builder_filters_report_and_schema(tmp_path):
    args = builder_args(tmp_path, "--min-bandwidth", "3000")
    rows = good_candidates(30)
    bad = [
        dict(id="bad/freya", text="ISTANBUL'DA YARIN ÖĞLEDEN SONRA YAĞMUR BEKLENİYOR", speaker="spk/x",
             audio_bytes=wav_bytes(noise(3))),
        dict(id="bad/short", text="Merhaba.", speaker="spk/x", audio_bytes=wav_bytes(noise(0.5))),
        dict(id="bad/meta-long", text="Merhaba.", speaker="spk/x", duration=25.0, path="never/read.wav"),
        dict(id="bad/fast", text=" ".join(WORDS * 3), speaker="spk/x", audio_bytes=wav_bytes(noise(1.5, seed=3))),
        dict(id="bad/script", text="Привет мир", speaker="spk/x", audio_bytes=wav_bytes(noise(2))),
        dict(id="bad/empty", text=" ... ", speaker="spk/x", audio_bytes=wav_bytes(noise(2))),
        dict(id="bad/silent", text=sentence(1, 4), speaker="spk/x", audio_bytes=wav_bytes(np.zeros(32000, np.float32))),
        dict(id="bad/corrupt", text=sentence(2, 4), speaker="spk/x", audio_bytes=b"not audio at all"),
        dict(id="bad/votes", text=sentence(3, 4), speaker="spk/x", reject="cv_down_votes", audio_bytes=b""),
        dict(id="bad/narrow", text=sentence(4, 4), speaker="spk/x", audio_bytes=wav_bytes(noise(2.2, cutoff=1500))),
        dict(id="bad/pauses", text=sentence(5, 4), speaker="spk/x",
             audio_bytes=wav_bytes(np.concatenate([noise(1.2, seed=9), np.zeros(19200, np.float32)]))),
        dict(id="bad/nospeaker", text=sentence(6, 4), speaker=" ", audio_bytes=wav_bytes(noise(2))),
    ]
    builder = manifest_pipeline.ManifestBuilder(
        args, source="test-source", license="CC0-1.0", vad=lambda audio, sr: float(np.mean(np.abs(audio) > 1e-4))
    )
    summary = builder.run(rows[:15] + bad + rows[15:])
    expected = {
        "freya_exact": 1, "duration": 2, "chars_per_second": 1, "text_unnormalizable": 1, "empty_text": 1,
        "silent": 1, "decode_error": 1, "cv_down_votes": 1, "bandwidth": 1, "speech_ratio": 1, "missing_speaker": 1,
    }
    assert summary["removed"] == expected
    assert summary["evaluated"] == 30 + len(bad) and summary["kept"] == 30
    assert summary["removed_fraction"] == pytest.approx(len(bad) / (30 + len(bad)))
    assert len(summary["warnings"]) == 1 and "8.0 kHz" in summary["warnings"][0]  # 12/42 removed: within budget
    assert summary["chars_per_second_mode"] == "iqr"
    low, high = summary["chars_per_second_bounds"]
    assert low < 12 < high
    assert summary["speakers"] == 6 and summary["speakers_with_2plus_clips"] == 6
    assert summary["kept_bandwidth_histogram"]["4-8k"] == 30  # 16 kHz clips: <= 8 kHz
    assert summary["kept_sample_rates"] == {"16000": 30}
    assert summary["freya_sentences"] == 2
    rejected = [json.loads(line) for line in (tmp_path / "out" / "rejected.jsonl").read_text().splitlines()]
    assert sorted(r["id"] for r in rejected) == sorted(r["id"] for r in bad)
    assert next(r for r in rejected if r["id"] == "bad/narrow")["bandwidth_hz"] < 3000
    kept = manifest_rows(tmp_path / "out")
    assert [r["id"] for r in kept] == [r["id"] for r in rows]  # source order survives staging and prefetch
    assert list(kept[0]) == manifest_pipeline.schema().names
    first = kept[0]
    assert (first["language"], first["source"], first["license"]) == ("tr", "test-source", "CC0-1.0")
    assert first["audio"]["bytes"] == rows[0]["audio_bytes"]  # original bytes, not transcoded
    assert first["speech_ratio"] > 0.9 and 7000 < first["bandwidth_hz"] <= 8000
    assert first["chars_per_second"] == pytest.approx(12, rel=0.1)
    assert json.loads(first["source_meta"]) == {}
    assert not (tmp_path / "out" / ".staging").exists()
    with pytest.raises(ValueError, match="already exists"):
        manifest_pipeline.ManifestBuilder(args, source="test-source", license="CC0-1.0")


def test_fixed_bounds_cap_sampling_limit_and_restoration(tmp_path):
    rows = good_candidates(40)
    args = builder_args(tmp_path, "--cps-bounds", "0,11.9", "--max-per-speaker", "3", "--workers", "0")
    summary = manifest_pipeline.ManifestBuilder(args, source="s", license="MIT").run(rows)
    assert summary["chars_per_second_mode"] == "fixed" and summary["chars_per_second_bounds"] == [0, 11.9]
    kept = manifest_rows(tmp_path / "out")
    assert all(r["chars_per_second"] <= 11.9 for r in kept)
    assert max(sum(r["speaker"] == s for r in kept) for s in {r["speaker"] for r in kept}) <= 3
    assert summary["removed"]["chars_per_second"] + summary["removed"].get("speaker_cap", 0) + len(kept) == 40
    assert any("Raon-OpenTTS" in w for w in summary["warnings"])  # more than 30% removed

    def picked(fraction, limit=0, name="sample"):
        extra = ["--sample-fraction", str(fraction), "--output", str(tmp_path / name), "--cps-iqr-k", "0"]
        args = builder_args(tmp_path, *extra, *(["--limit", str(limit)] if limit else []))
        summary = manifest_pipeline.ManifestBuilder(args, source="s", license="MIT").run(iter(rows))
        return summary, [r["id"] for r in manifest_rows(tmp_path / name)]

    summary, ids = picked(0.5, name="half")
    assert summary["sampled_out"] + summary["evaluated"] == 40 and 8 < len(ids) < 32
    assert picked(0.5, name="half-again")[1] == ids  # deterministic id-hash sample
    summary, limited = picked(1.0, limit=7, name="limited")
    assert summary["evaluated"] == 7 and limited == [r["id"] for r in rows[:7]]

    def restore(audio, sample_rate):
        spectrum = np.fft.rfft(audio)
        spectrum[np.fft.rfftfreq(len(audio), 1 / sample_rate) > 3000] = 0
        return np.fft.irfft(spectrum, len(audio)).astype(np.float32), sample_rate

    args = builder_args(tmp_path, "--output", str(tmp_path / "restored"), "--cps-iqr-k", "0", "--limit", "3")
    manifest_pipeline.ManifestBuilder(args, source="s", license="MIT", restore=restore).run(rows)
    for row in manifest_rows(tmp_path / "restored"):
        assert row["restored"] == "restore" and row["bandwidth_hz"] < 3300
        audio, sample_rate = sf.read(io.BytesIO(row["audio"]["bytes"]))
        assert sf.info(io.BytesIO(row["audio"]["bytes"])).format == "FLAC" and sample_rate == 16000


def test_freya_file_is_required(tmp_path):
    parser = argparse.ArgumentParser()
    manifest_pipeline.add_arguments(parser)
    args = parser.parse_args(["--output", str(tmp_path / "x")])
    with pytest.raises(ValueError, match="freya"):
        manifest_pipeline.ManifestBuilder(args, source="s", license="MIT")
    args = parser.parse_args(["--output", str(tmp_path / "x"), "--no-freya-check", "--cps-iqr-k", "0"])
    summary = manifest_pipeline.ManifestBuilder(args, source="s", license="MIT").run(good_candidates(2))
    assert any("Freya" in w for w in summary["warnings"])
