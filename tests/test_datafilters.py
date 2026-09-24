import json
import math

import numpy as np
import pytest

from dacvae_tts.datafilters import (
    FreyaExclusion,
    SileroVAD,
    bandwidth_hz,
    chars_per_second,
    check_training_source,
    duration_ok,
    iqr_bounds,
    iqr_outliers,
    load_callable,
    load_sentences,
    match_key,
    speech_ratio_from_segments,
    text_characters,
)


def brickwall(signal, sample_rate, cutoff):
    spectrum = np.fft.rfft(signal)
    spectrum[np.fft.rfftfreq(len(signal), 1 / sample_rate) > cutoff] = 0
    return np.fft.irfft(spectrum, len(signal))


@pytest.mark.parametrize("cutoff", [4000, 8000, 12000])
def test_bandwidth_of_lowpassed_white_noise(cutoff):
    noise = np.random.default_rng(cutoff).normal(0, 0.1, 48000 * 3)
    estimate = bandwidth_hz(brickwall(noise, 48000, cutoff), 48000)
    # ~23 Hz Welch bins; Hann leakage past a brick wall stays within a few bins of the cutoff.
    assert cutoff <= estimate + 50 and estimate - cutoff <= 250


def test_bandwidth_limits_and_threshold():
    rng = np.random.default_rng(0)
    assert bandwidth_hz(rng.normal(0, 0.1, 48000), 48000) == 24000  # full band reaches Nyquist
    assert bandwidth_hz(rng.normal(0, 0.1, 16000), 16000) == 8000  # a 16 kHz YODAS clip cannot exceed 8 kHz
    assert bandwidth_hz(np.zeros(48000), 48000) == 0.0
    assert bandwidth_hz(np.ones(8), 48000) == 0.0
    # A 6 dB/octave tilt: a tighter threshold reports a lower bandwidth; DC offset cannot set the reference.
    spectrum = np.fft.rfft(rng.normal(0, 0.1, 48000 * 2))
    freqs = np.fft.rfftfreq(48000 * 2, 1 / 48000)
    tilted = np.fft.irfft(spectrum * 100 / np.maximum(freqs, 100), 48000 * 2)
    assert bandwidth_hz(tilted, 48000, threshold_db=20) < bandwidth_hz(tilted, 48000, threshold_db=50)
    assert abs(bandwidth_hz(brickwall(tilted, 48000, 8000) + 5.0, 48000) - 8000) <= 250


FREYA = [
    "Işık hızı saniyede yaklaşık 300.000 kilometredir.",
    "İstanbul'da yarın öğleden sonra yağmur bekleniyor.",
    "Kâğıt kesiği çok acıtır.",
]


def test_freya_exact_matches_turkish_casing_punctuation_and_numbers():
    freya = FreyaExclusion(FREYA)
    assert len(freya) == 3
    assert freya.match("IŞIK HIZI SANİYEDE YAKLAŞIK ÜÇ YÜZ BİN KİLOMETREDİR") == "freya_exact"
    assert freya.match("istanbulda yarın öğleden sonra yağmur bekleniyor!") == "freya_exact"
    assert freya.match("İSTANBUL’DA  YARIN, öğleden sonra yağmur bekleniyor…") == "freya_exact"
    assert freya.match("Istanbul'da yarın öğleden sonra yağmur bekleniyor.") == "freya_exact"  # ASCII I
    assert freya.match("Kağıt kesiği çok acıtır") == "freya_exact"  # circumflex optional
    assert match_key("Işık 3'te!") == match_key("işik üçte") == "işik üçte"
    assert freya.match("Bugün hava çok güzel.") is None
    assert freya.match("...") is None


def test_freya_contained_and_near_duplicates():
    freya = FreyaExclusion(FREYA)
    long_caption = "Hava durumuna göre İstanbul'da yarın öğleden sonra yağmur bekleniyor, dikkatli olun."
    assert freya.match(long_caption) == "freya_contains"
    edited = "Işık hızı saniyede yaklaşık üç yüz bin kilometre."
    assert freya.match(edited) is None
    near = FreyaExclusion(FREYA, near_threshold=0.8)
    assert near.match(edited) == "freya_near"
    assert near.match("Işık hızı çok yüksektir.") is None  # one shared trigram of a short text is not enough
    assert near.match("Kâğıt kesiği") is None  # contained evaluation sentences need >= 5 words; no trigram
    with pytest.raises(ValueError):
        FreyaExclusion(FREYA, near_threshold=1.5)


def test_load_sentences_formats(tmp_path):
    (tmp_path / "freya.jsonl").write_text(
        "\n".join(json.dumps({"id": i, "text": s}, ensure_ascii=False) for i, s in enumerate(FREYA)) + "\n"
    )
    (tmp_path / "freya.txt").write_text("\n".join(FREYA) + "\n\n")
    (tmp_path / "freya.tsv").write_text("id\tsentence\n" + "".join(f"{i}\t{s}\n" for i, s in enumerate(FREYA)))
    (tmp_path / "freya.json").write_text(json.dumps(FREYA, ensure_ascii=False))
    for name in ("freya.jsonl", "freya.txt", "freya.tsv", "freya.json"):
        assert load_sentences(tmp_path / name) == FREYA
    (tmp_path / "empty.txt").write_text("\n")
    with pytest.raises(ValueError):
        load_sentences(tmp_path / "empty.txt")


def test_chars_per_second_and_iqr_cut():
    assert text_characters("Saat 3'te, geldi!") == len("saatüçtegeldi")
    assert chars_per_second("Saat 3'te, geldi!", 2.0) == 6.5
    with pytest.raises(ValueError):
        chars_per_second("a", 0)
    values = [10, 11, 12, 13, 14, 15, 16, 40, 1]
    low, high = iqr_bounds(values)
    q1, q3 = np.percentile(values, [25, 75])
    assert (low, high) == pytest.approx((q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)))
    assert iqr_outliers(values).tolist() == [False] * 7 + [True, True]
    assert iqr_bounds([]) == (-math.inf, math.inf)
    # Per-source fences: 20 chars/s is an outlier for a slow source and normal for a fast one.
    slow, fast = [9.0, 10.0, 10.5, 11.0, 12.0, 20.0], [18.0, 19.0, 20.0, 21.0, 22.0, 20.5]
    mask = iqr_outliers(slow + fast, ["slow"] * 6 + ["fast"] * 6)
    assert mask.tolist() == [False] * 5 + [True] + [False] * 6


def test_duration_bounds():
    assert [duration_ok(d) for d in (0.99, 1.0, 12.3, 20.0, 20.01, None)] == [False, True, True, True, False, False]
    assert duration_ok(25.0, 1.0, 30.0)


def test_speech_ratio_from_segments_merges_and_clips():
    segments = [{"start": 0, "end": 100}, {"start": 50, "end": 150}, {"start": 300, "end": 400}]
    assert speech_ratio_from_segments(segments, 1000) == 0.25
    assert speech_ratio_from_segments([(900, 1200), (-10, 0)], 1000) == 0.1
    assert speech_ratio_from_segments([], 1000) == 0.0
    assert speech_ratio_from_segments([(0, 5)], 0) == 0.0


def test_silero_vad_is_lazy(monkeypatch):
    import torch

    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("loaded eagerly")))
    SileroVAD(threshold=0.6)  # nothing is downloaded until the first clip is measured


def test_training_source_policy():
    for name in ("google/fleurs", "/data/MediaSpeech/tr", "antalia-tr", "data/eval/freya_tr_eval.jsonl", "KIRAAT"):
        with pytest.raises(ValueError, match="training manifest"):
            check_training_source(name)
    check_training_source("espnet/yodas2", "issai/Turkish_Speech_Corpus", None, "raw/cv-corpus-27.0/tr")


def test_load_callable():
    assert load_callable("math:hypot")(3, 4) == 5
    for spec in ("math", "math:pi", ":hypot"):
        with pytest.raises(ValueError):
            load_callable(spec)
