import numpy as np

from dacvae_tts.audio import fade, finalize, join, loudness, to_pcm16, trim_silence

RATE = 48000


def tone(seconds, amplitude=0.3, frequency=220.0):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def test_trim_silence_keeps_margins():
    audio = np.concatenate([np.zeros(RATE), tone(1.0), np.zeros(2 * RATE)])
    trimmed, start, end = trim_silence(audio, RATE, margin_start_ms=100, margin_end_ms=150)
    assert abs(start - int(0.9 * RATE)) <= RATE // 100 and abs(end - int(2.15 * RATE)) <= RATE // 100
    assert len(trimmed) == end - start
    silent = np.zeros(RATE, dtype=np.float32)
    assert trim_silence(silent, RATE)[0] is not None


def test_fade_join_and_finalize():
    audio = tone(0.5)
    faded = fade(audio, RATE)
    assert faded[0] == 0 and abs(faded[-1]) < 1e-3 and np.allclose(faded[RATE // 10], audio[RATE // 10])
    joined = join([tone(0.5), tone(0.5)], RATE, [0.3, 0.0])
    assert len(joined) == int(1.3 * RATE)
    loud = np.clip(tone(2.0, amplitude=1.2), -1, 1)
    out, info = finalize(loud, RATE, target_lufs=-16.0, peak_dbfs=-1.0)
    assert np.abs(out).max() <= 10 ** (-1 / 20) + 1e-4
    assert info["clipped_input_fraction"] > 0 and info["gain_db"] < 0
    quiet, _ = finalize(tone(2.0, amplitude=0.01), RATE, target_lufs=-16.0)
    assert abs(loudness(quiet[int(0.12 * RATE):-int(0.2 * RATE)], RATE) + 16.0) < 0.5
    assert to_pcm16(np.array([2.0, -2.0, 0.5])).tolist() == [32767, -32767, 16384]
