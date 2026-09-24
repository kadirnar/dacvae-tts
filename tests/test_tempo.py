"""WSOLA tempo change (dacvae_tts.tempo) on synthetic signals: length, pitch, level, identity."""

import numpy as np
import pytest

from dacvae_tts.tempo import time_stretch

RATE = 48000


def tones(seconds=1.0, frequencies=(200.0, 730.0), amplitudes=(0.5, 0.2)):
    t = np.arange(int(seconds * RATE)) / RATE
    return sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(frequencies, amplitudes)).astype(np.float32)


def spectrum_peak(audio, low, high):
    magnitude = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    frequencies = np.fft.rfftfreq(len(audio), 1 / RATE)
    band = (frequencies >= low) & (frequencies <= high)
    return frequencies[band][np.argmax(magnitude[band])]


@pytest.mark.parametrize("rate", [0.75, 0.8, 0.9, 1.1, 1.25, 1.33])
def test_tempo_changes_duration_but_keeps_pitch_and_level(rate):
    audio = tones()
    out = time_stretch(audio, rate, RATE)
    assert out.dtype == np.float32 and len(out) == round(len(audio) / rate)
    # Both partials stay where they were (a resampling "speed" change would move them by `rate`).
    assert spectrum_peak(out, 100, 400) == pytest.approx(200, abs=3)
    assert spectrum_peak(out, 500, 1000) == pytest.approx(730, abs=4)
    middle = out[len(out) // 4 : 3 * len(out) // 4]
    assert np.sqrt(np.mean(middle**2)) == pytest.approx(np.sqrt(np.mean(audio**2)), rel=0.03)


def test_rate_one_is_identity_and_inputs_are_checked():
    audio = tones(0.3)
    assert np.array_equal(time_stretch(audio, 1.0, RATE), audio)
    for rate in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            time_stretch(audio, rate, RATE)
    with pytest.raises(ValueError):
        time_stretch(np.array([np.nan], np.float32), 1.1, RATE)


def test_silence_and_speech_like_bursts_keep_their_order():
    # Two bursts in silence start and end where the time map puts them (input position / rate), within the
    # WSOLA search tolerance plus one frame; the silence around them stays silent.
    audio = np.zeros(RATE, np.float32)
    audio[4800:9600] = tones(0.1)
    audio[38400:43200] = tones(0.1, (300.0,), (0.4,))
    rate, slack = 1.25, int(0.04 * RATE)  # 10 ms tolerance + 30 ms frame
    out = time_stretch(audio, rate, RATE)
    loud = np.abs(out) > 1e-3
    for start, end in ((4800, 9600), (38400, 43200)):
        lo, hi = int(start / rate) - slack, int(end / rate) + slack
        support = np.flatnonzero(loud[lo:hi]) + lo
        assert abs(support[0] - start / rate) < slack and abs(support[-1] - end / rate) < slack
    assert not loud[: int(4800 / rate) - slack].any() and not loud[int(9600 / rate) + slack : int(38400 / rate) - slack].any()


@pytest.mark.parametrize("rate", [0.8, 1.25])
def test_constant_stays_constant_pulses_scale_and_output_is_deterministic(rate):
    constant = np.full(RATE // 2, 0.3, np.float32)
    assert np.abs(time_stretch(constant, rate, RATE) - 0.3).max() < 1e-5  # no dip at the edges or the seams
    pulses = np.zeros(RATE, np.float32)
    pulses[::480] = 1.0  # 100 Hz glottal-like pulse train
    out = time_stretch(pulses, rate, RATE)
    peaks = np.flatnonzero((out[1:-1] > 0.5) & (out[1:-1] >= out[:-2]) & (out[1:-1] >= out[2:]))
    assert abs(len(peaks) - round(100 / rate)) <= 1 and np.median(np.diff(peaks)) == 480  # period kept
    assert np.array_equal(out, time_stretch(pulses, rate, RATE))
    assert np.isfinite(time_stretch(np.zeros(4800, np.float32), rate, RATE)).all()
