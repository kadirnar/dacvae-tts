"""Waveform helpers for serving: silence trimming, fades, joining sentence chunks, output loudness.

These act on float numpy waveforms outside the model. `trim_silence` is used on voice prompts (long leading or
trailing silence would lower the measured speaking rate and add a pause before the new sentence) and on generated
chunks before they are joined with controlled pauses; `speech_timing` measures a prompt's edge silence, pauses and
net speaking time for the `articulation` duration rule (duration.py). `finalize` gives every output the same
presentation: short fades against clicks (generated speech starts on the first frame), a little padding, -16 LUFS
integrated loudness (the training data's level; high guidance otherwise makes outputs ~2-3 dB louder) with the peak
kept under -1 dBFS.
"""

import numpy as np


def frame_levels(audio, sample_rate, frame_ms=10):
    """RMS level per frame in dBFS."""
    size = max(int(sample_rate * frame_ms / 1000), 1)
    count = max(len(audio) // size, 1)
    frames = np.asarray(audio[: count * size], dtype=np.float64).reshape(count, -1) if len(audio) >= size else \
        np.asarray(audio, dtype=np.float64)[None]
    return 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-9), size


def trim_silence(audio, sample_rate, below_peak_db=40.0, floor_db=-60.0, margin_start_ms=100, margin_end_ms=150):
    """Cut leading/trailing frames quieter than max(peak - below_peak_db, floor_db); keep small margins.

    Returns (trimmed audio, start sample, end sample). Audio without any loud frame is returned unchanged.
    """
    audio = np.asarray(audio, dtype=np.float32)
    levels, size = frame_levels(audio, sample_rate)
    threshold = max(levels.max() - below_peak_db, floor_db)
    loud = np.flatnonzero(levels > threshold)
    if not len(loud):
        return audio, 0, len(audio)
    start = max(loud[0] * size - int(sample_rate * margin_start_ms / 1000), 0)
    end = min((loud[-1] + 1) * size + int(sample_rate * margin_end_ms / 1000), len(audio))
    return audio[start:end], start, end


def _runs(mask):
    """[start, end) frame indices of the True runs of a boolean array, shape (runs, 2)."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return edges.reshape(-1, 2)


def speech_timing(audio, sample_rate, below_peak_db=35.0, floor_db=-60.0, min_pause_ms=200, min_speech_ms=50,
                  percentile=95.0, frame_ms=10):
    """Edge silence, internal pauses and net speaking time of a voice prompt (inputs of the `articulation` rule).

    A 10 ms frame is speech when its RMS level is above max(p - below_peak_db, floor_db), p the `percentile`-th
    frame level (a robust peak: one click cannot raise the threshold the way the maximum does; 35 dB below it is
    about F5-TTS's -42 dBFS edge trim for its -20 dBFS RMS prompts). Speech runs shorter than `min_speech_ms`
    (clicks, lip smacks) count as silence. Leading/trailing silence is cut, and silent runs of at least
    `min_pause_ms` between speech are pauses: 200 ms is above stop closures and ordinary word gaps, which belong to
    articulation, and below phrase pauses. Speaking time = trimmed span - pauses. Loudness-invariant apart from
    the floor. Returns a dict of plain floats (JSON metadata); speech_seconds is 0 when no frame is loud.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    total = len(audio) / sample_rate
    levels, size = frame_levels(audio, sample_rate, frame_ms)
    step = size / sample_rate
    reference = float(np.percentile(levels, percentile))
    threshold = max(reference - below_peak_db, floor_db)
    loud = levels > threshold
    for start, end in _runs(loud):
        if (end - start) * step < min_speech_ms / 1000:
            loud[start:end] = False
    runs = _runs(loud)
    timing = {"total_seconds": total, "reference_db": reference, "threshold_db": threshold, "min_pause_ms": min_pause_ms}
    if not len(runs):
        return {**timing, "speech_seconds": 0.0, "pause_seconds": 0.0, "pauses": 0, "leading_silence_seconds": total,
                "trailing_silence_seconds": 0.0}
    gaps = runs[1:, 0] - runs[:-1, 1]
    pauses = gaps[gaps * step >= min_pause_ms / 1000 - 1e-9]
    first, last = int(runs[0, 0]), int(runs[-1, 1])
    return {
        **timing,
        "speech_seconds": float((last - first - pauses.sum()) * step),
        "pause_seconds": float(pauses.sum() * step),
        "pauses": int(len(pauses)),
        "leading_silence_seconds": first * step,
        "trailing_silence_seconds": max(total - last * step, 0.0),
    }


def fade(audio, sample_rate, fade_in_ms=10, fade_out_ms=20):
    """Raised-cosine fade in/out (removes the click of speech that starts on the first sample)."""
    audio = np.array(audio, dtype=np.float32, copy=True)
    for length, reverse in ((int(sample_rate * fade_in_ms / 1000), False), (int(sample_rate * fade_out_ms / 1000), True)):
        length = min(length, len(audio) // 2)
        if length < 2:
            continue
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, length))).astype(np.float32)
        if reverse:
            audio[-length:] *= ramp[::-1]
        else:
            audio[:length] *= ramp
    return audio


def join(segments, sample_rate, pauses):
    """Concatenate chunks with `pauses[i]` seconds of silence after chunk i (the last pause is ignored)."""
    if not segments:
        return np.zeros(0, dtype=np.float32)
    parts = []
    for index, segment in enumerate(segments):
        parts.append(fade(segment, sample_rate))
        if index + 1 < len(segments):
            parts.append(np.zeros(int(sample_rate * max(pauses[index], 0.0)), dtype=np.float32))
    return np.concatenate(parts)


def loudness(audio, sample_rate):
    import pyloudnorm

    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) < 0.4 * sample_rate:
        return float("nan")
    value = pyloudnorm.Meter(sample_rate).integrated_loudness(audio)
    return float(value) if np.isfinite(value) else float("nan")


def finalize(audio, sample_rate, target_lufs=-16.0, peak_dbfs=-1.0, pad_start_ms=120, pad_end_ms=200):
    """Fades, padding and loudness normalization with a peak ceiling. Returns (audio, info)."""
    audio = fade(np.asarray(audio, dtype=np.float32), sample_rate)
    clipped = float(np.mean(np.abs(audio) >= 0.999)) if len(audio) else 0.0
    measured = loudness(audio, sample_rate)
    gain_db = 0.0 if not np.isfinite(measured) else target_lufs - measured
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    ceiling = 10 ** (peak_dbfs / 20)
    gain = 10 ** (gain_db / 20)
    if peak * gain > ceiling and peak > 0:
        gain = ceiling / peak
    audio = audio * np.float32(gain)
    audio = np.concatenate([np.zeros(int(sample_rate * pad_start_ms / 1000), np.float32), audio,
                            np.zeros(int(sample_rate * pad_end_ms / 1000), np.float32)])
    return audio, {"input_lufs": measured, "gain_db": float(20 * np.log10(max(gain, 1e-9))),
                   "clipped_input_fraction": clipped}


def to_pcm16(audio):
    return (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).round().astype(np.int16)
