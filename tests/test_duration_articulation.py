import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dacvae_tts.audio import speech_timing
from dacvae_tts.duration import DurationPredictor, articulation_seconds, pause_budget, rule_frames, units
from dacvae_tts.inference import Synthesizer, VoiceReference

RATE = 24000
SYLLABLE, GAP = 0.15, 0.05  # one "syllable": a 150 ms tone burst; bursts of a phrase are 50 ms apart


def phrase(syllables, rng):
    parts = []
    for _ in range(syllables):
        t = np.arange(int(SYLLABLE * RATE)) / RATE
        parts.append(0.3 * np.sin(2 * np.pi * rng.uniform(120, 260) * t) * np.hanning(len(t)) ** 0.25)
        parts.append(np.zeros(int(GAP * RATE)))
    return np.concatenate(parts[:-1])


def prompt(phrases, lead=0.0, trail=0.0, pause=0.0, seed=0):
    """Tone-burst 'speech' with edge silences and `pause` seconds between phrases, over a -75 dBFS noise floor."""
    rng = np.random.default_rng(seed)
    parts = [np.zeros(int(lead * RATE))]
    for index, count in enumerate(phrases):
        if index:
            parts.append(np.zeros(int(pause * RATE)))
        parts.append(phrase(count, rng))
    parts.append(np.zeros(int(trail * RATE)))
    audio = np.concatenate(parts)
    return (audio + rng.normal(0, 10 ** (-75 / 20), len(audio))).astype(np.float32)


def speech_span(phrases):
    return sum(count * SYLLABLE + (count - 1) * GAP for count in phrases)


TRANSCRIPT = "Bir iki üç dört beş altı yedi sekiz dokuz on bir iki."  # 18 vowels = 18 syllables
PHRASES = [6, 6, 6]


def test_speech_timing_ignores_edge_silence_and_long_pauses():
    assert units(TRANSCRIPT)["syllables"] == sum(PHRASES)
    plain = speech_timing(prompt(PHRASES, pause=GAP), RATE)
    edges = speech_timing(prompt(PHRASES, lead=1.0, trail=2.0, pause=GAP), RATE)
    pausy = speech_timing(prompt(PHRASES, lead=0.7, trail=0.4, pause=0.6), RATE)
    span = speech_span(PHRASES)  # the phrases; a 50 ms gap between them is articulation, a 600 ms one a pause
    for timing, gaps in ((plain, 2 * GAP), (edges, 2 * GAP), (pausy, 0.0)):
        assert timing["speech_seconds"] == pytest.approx(span + gaps, abs=0.03)
    assert plain["pauses"] == edges["pauses"] == 0 and plain["pause_seconds"] == 0
    assert edges["leading_silence_seconds"] == pytest.approx(1.0, abs=0.03)
    assert edges["trailing_silence_seconds"] == pytest.approx(2.0, abs=0.03)
    assert pausy["pauses"] == 2 and pausy["pause_seconds"] == pytest.approx(1.2, abs=0.04)
    # a click in the leading silence is not speech; a short (150 ms) gap is articulation, not a pause
    clicky = prompt(PHRASES, lead=1.0, pause=0.15)
    clicky[RATE // 2 : RATE // 2 + RATE // 500] = 0.9
    timing = speech_timing(clicky, RATE)
    assert timing["leading_silence_seconds"] == pytest.approx(1.0, abs=0.03) and timing["pauses"] == 0
    assert timing["speech_seconds"] == pytest.approx(speech_span(PHRASES) + 0.3, abs=0.04)
    silent = speech_timing(np.zeros(RATE, np.float32), RATE)
    assert silent["speech_seconds"] == 0 and json.dumps(silent)


def test_articulation_rate_is_insensitive_to_edge_silence_and_pauses():
    text = "Bugün hava çok güzel dışarı çıkalım mı"  # no internal punctuation
    seconds, rules = [], []
    for kwargs, gaps in (({}, 2 * GAP), ({"lead": 1.0, "trail": 2.0}, 2 * GAP),
                         ({"lead": 0.5, "trail": 0.5, "pause": 0.7}, 0.0)):
        audio = prompt(PHRASES, **{"pause": GAP, **kwargs})
        value, profile = articulation_seconds(speech_timing(audio, RATE), TRANSCRIPT, text)
        seconds.append(value)
        rules.append(rule_frames(len(audio) / RATE * 25, TRANSCRIPT, text) / 25)
        assert profile["duration_articulation_rate"] == pytest.approx(18 / (speech_span(PHRASES) + gaps), rel=0.01)
        assert profile["duration_prompt_timing"]["speech_seconds"] > 0 and not profile["duration_floor_applied"]
    assert seconds[1] == pytest.approx(seconds[0], rel=0.01)  # the articulation length does not move ...
    assert seconds[2] == pytest.approx(seconds[0], rel=0.04)  # (only the two 50 ms word gaps became pauses)
    assert rules[1] / rules[0] > 1.5 and rules[2] / rules[0] > 1.4  # ... where frames per byte grows by half


def test_pause_budget_counts_internal_punctuation_only():
    assert pause_budget("Evet.") == (0.0, 0, 0)
    seconds, commas, stops = pause_budget("Merhaba, nasılsın? İyiyim; sen...")
    assert (commas, stops) == (2, 1) and seconds == pytest.approx(2 * 0.15 + 0.3)
    assert pause_budget('Dedi ki: "Geliyorum."')[1:] == (1, 0)
    assert pause_budget("Ali-Veli geldi - sonra gitti,")[1:] == (1, 0)
    assert pause_budget("Bir. İki. Üç!", comma_pause=0.1, stop_pause=0.5) == (1.0, 0, 2)
    timing = {"speech_seconds": 3.0}
    plain = articulation_seconds(timing, TRANSCRIPT, "Evet hayır belki yarın gel")[0]
    punctuated, profile = articulation_seconds(timing, TRANSCRIPT, "Evet, hayır. Belki yarın gel.")
    assert punctuated - plain == pytest.approx(0.45) and profile["duration_pause_budget_seconds"] == pytest.approx(0.45)


def test_short_text_floor_rate_clamp_and_fallbacks():
    timing = {"speech_seconds": 3.0}  # 18 syllables in 3 s: 6 syllables/s
    seconds, profile = articulation_seconds(timing, TRANSCRIPT, "Evet.")
    assert seconds == 0.5 and profile["duration_floor_applied"] and profile["duration_target_syllables"] == 2
    assert articulation_seconds(timing, TRANSCRIPT, "Evet.", min_seconds=0.2)[0] == pytest.approx(2 / 6)
    fast, profile = articulation_seconds({"speech_seconds": 0.5}, TRANSCRIPT, "Bugün hava çok güzel.")
    assert profile["duration_articulation_measured_rate"] == 36 and profile["duration_articulation_rate"] == 8.5
    assert fast == pytest.approx(7 / 8.5)
    slow = articulation_seconds({"speech_seconds": 30.0}, TRANSCRIPT, "Bugün hava çok güzel.")[1]
    assert slow["duration_articulation_rate"] == 3.0
    for timing, reason in ((None, "no prompt waveform"), ({"speech_seconds": 0.0}, "no speech detected")):
        value, profile = articulation_seconds(timing, TRANSCRIPT, "Bugün hava çok güzel.")
        assert value is None and profile["duration_articulation_fallback"] == f"rule ({reason})"
    assert articulation_seconds({"speech_seconds": 3.0}, "Hmm.", "Bugün.")[1]["duration_articulation_fallback"] == \
        "rule (no prompt vowels)"


def bare_synthesizer():
    tts = Synthesizer.__new__(Synthesizer)
    tts.text_version, tts.duration_model = "turkish-v1", None
    tts.codec = SimpleNamespace(sample_rate=48000, hop_length=1920)
    return tts


def test_existing_duration_modes_are_unchanged():
    """Frames of every pre-existing mode, recorded before the articulation rule was added."""
    tts = bare_synthesizer()
    cases = [(60, "Hızlı hızlı konuşan bir referans kaydı burada duruyor.", "Bu biraz daha uzun ikinci bir cümle.",
              {"rule": 37, "clamp": 44, "syllable": 41, "predictor": 62, "auto": 44}),
             (150, "Yavaş, dinlendirici bir sesle okunan kısa bir cümle.", "Merhaba dünya.",
              {"rule": 41, "clamp": 41, "syllable": 42, "predictor": 41, "auto": 41}),
             (100, "Normal hızda konuşulmuş bir cümle.", "Evet, bugün hava çok güzel; dışarı çıkalım mı?",
              {"rule": 147, "clamp": 147, "syllable": 145, "predictor": 130, "auto": 130})]
    for frames, reference, text, expected in cases:
        for mode, value in expected.items():
            got, profile = tts.target_frames(frames, reference, text, duration_mode=mode)
            assert got == value and profile["duration_rule"] == "reference_frames_per_byte"
            assert not any(key.startswith(("duration_articulation", "duration_prompt")) for key in profile)
    assert tts.target_frames(60, cases[0][1], cases[0][2], duration_scale=1.1, duration_mode="clamp")[0] == 49


def test_target_frames_articulation_and_rule_fallback():
    tts = bare_synthesizer()
    reference, text = TRANSCRIPT, "Bugün hava çok güzel, dışarı çıkalım mı?"
    syllables = units(text)["syllables"]  # 14; 18 prompt syllables in 3 s of speech: 6 per second
    frames, profile = tts.target_frames(100, reference, text, duration_mode="articulation", timing={"speech_seconds": 3.0})
    assert frames == round((syllables / 6 + 0.15) * 25)
    assert profile["duration_rule"] == "prompt_syllables_per_speaking_second"
    assert tts.target_frames(100, reference, text, duration_scale=1.2, duration_mode="articulation",
                             timing={"speech_seconds": 3.0})[0] == round((syllables / 6 + 0.15) * 25 * 1.2)
    fallback, profile = tts.target_frames(100, reference, text, duration_mode="articulation")
    assert fallback == tts.target_frames(100, reference, text)[0]
    assert profile["duration_articulation_fallback"] == "rule (no prompt waveform)"
    tts.articulation_options = {"comma_pause": 0.55}
    assert tts.target_frames(100, reference, text, duration_mode="articulation",
                             timing={"speech_seconds": 3.0})[0] == round((syllables / 6 + 0.55) * 25)


class ToneCodec:
    """Fake codec: frames whose first (denormalized) channel exceeds 0.5 decode to a tone, the others to silence."""

    def __init__(self, checkpoint, device, **_):
        self.latent_dim, self.sample_rate, self.hop_length = 4, 24000, 512
        self.metadata = dict(checkpoint="test-codec", sample_rate=24000, hop_length=512, latent_dim=4,
                             posterior="mean", weights_sha256="fixture", preprocessing="fixture")

    def decode(self, z):
        t = torch.arange(len(z) * 512) / 24000
        gate = (z.float()[:, 0] > 0.5).float().repeat_interleave(512)
        noise = 1e-4 * torch.sin(z.float().sum(-1)).repeat_interleave(512)  # latent-dependent, below the -60 dB floor
        return 0.3 * torch.sin(2 * math.pi * 220 * t) * gate + noise


def tiny_synthesizer(monkeypatch, cache, tmp_path):
    import dacvae_tts.inference as module
    from dacvae_tts.config import Config, ModelConfig
    from dacvae_tts.data import LatentDataset
    from dacvae_tts.model import FlowTTS

    data = LatentDataset(cache)
    config = Config(ModelConfig(latent_dim=4, width=16, depth=1, heads=2, text_depth=1, text_layout="joined",
                                duration="rule", positions="rope", prediction="edm"))
    path = tmp_path / "model.pt"
    model = FlowTTS(config.model)
    torch.save({"model": model.state_dict(), "ema": model.state_dict(), "config": config.to_dict(),
                "codec": {**data.meta, "text_normalization": "turkish-v1"}, "mean": data.mean, "std": data.std}, path)
    monkeypatch.setattr(module, "Codec", ToneCodec)
    return Synthesizer(path, device="cpu", precision="fp32")


def gated_latents(tts, pattern):
    """Normalized latents that ToneCodec decodes to speech (1) / silence (0) per frame."""
    raw = torch.zeros(len(pattern), 4)
    raw[:, 0] = torch.tensor(pattern, dtype=torch.float32)
    return (raw - tts.mean) / tts.std


def test_articulation_through_synthesizer_decodes_cached_prompts(monkeypatch, cache, tmp_path):
    tts = tiny_synthesizer(monkeypatch, cache, tmp_path)
    fps = 24000 / 512
    pattern = [0] * 30 + [1] * 70 + [0] * 20 + [1] * 70 + [0] * 40  # 0.64 s edge, 427 ms pause, 0.85 s edge
    voice = VoiceReference(gated_latents(tts, pattern), TRANSCRIPT, "cache", {})
    texts = ["Merhaba dünya, nasılsın?", "Bu biraz daha uzun ikinci bir cümle."]
    results, _ = tts.synthesize_many(texts, voice, candidates=2, steps=1, guidance=1.0, duration_mode="articulation")
    timing = voice.speech_timing
    assert timing["source"] == "decoded_latents" and timing["pauses"] == 1
    assert timing["speech_seconds"] == pytest.approx(140 / fps, abs=0.03)
    assert timing["leading_silence_seconds"] == pytest.approx(30 / fps, abs=0.03)
    for text, row in zip(texts, results):
        expected = round(articulation_seconds(timing, TRANSCRIPT, text)[0] * fps)
        assert all(r["frames"] == expected and r["duration"]["duration_mode"] == "articulation" for r in row)
        assert row[0]["duration"]["duration_articulation_rate"] == pytest.approx(18 / (140 / fps), rel=0.02)
    rule, _ = tts.synthesize_many(texts[:1], voice, steps=1, guidance=1.0)
    assert rule[0][0]["frames"] > results[0][0]["frames"]  # pauses and edges no longer count as speech
    single = tts.synthesize(texts[1], reference=voice, steps=1, guidance=1.0, duration_mode="articulation")
    assert single.metadata["duration_prompt_timing"] == timing
    assert len(single.audio) == results[1][0]["frames"] * 512
    # prompts prepared from a waveform are measured on it, before any decoding
    monkeypatch.setattr(tts, "encode_reference", lambda audio, rate: torch.zeros(20, 4))
    prepared = tts.prepare_reference((prompt(PHRASES, lead=1.0, pause=0.6), RATE), TRANSCRIPT)
    assert prepared.speech_timing["source"] == "waveform" and prepared.speech_timing["pauses"] == 2
    assert tts.prompt_timing(prepared) is prepared.speech_timing
    # an unreadable/absent file keeps the old behaviour: no timing, articulation falls back to decoding
    monkeypatch.setattr(tts, "reference", lambda path: gated_latents(tts, pattern))
    assert tts.prepare_reference(tmp_path / "missing.wav", TRANSCRIPT).speech_timing is None


def test_duration_factors_cycle_over_candidates(monkeypatch, cache, tmp_path):
    tts = tiny_synthesizer(monkeypatch, cache, tmp_path)
    voice = VoiceReference(torch.randn(60, 4), "Hızlı hızlı konuşan bir referans kaydı burada duruyor.", "test", {})
    texts = ["Merhaba dünya.", "Bu biraz daha uzun ikinci bir cümle."]
    plain, plain_meta = tts.synthesize_many(texts, voice, candidates=3, steps=2, guidance=2.0, max_rows=4)
    same, same_meta = tts.synthesize_many(texts, voice, candidates=3, steps=2, guidance=2.0, max_rows=4,
                                          duration_factors=[1.0])
    for a, b in zip(sum(plain, []), sum(same, [])):
        assert np.array_equal(a["audio"], b["audio"]) and a["frames"] == b["frames"] and a["duration"] == b["duration"]
    assert "duration_factor" not in plain[0][0] and "candidate_factors" not in plain_meta
    assert same_meta["candidate_factors"] == [1.0, 1.0, 1.0]
    factors = [1.0, 0.9, 1.1]
    diverse, meta = tts.synthesize_many(texts, voice, candidates=6, steps=2, guidance=2.0, max_rows=4,
                                        duration_factors=factors, duration_mode="clamp")
    assert meta["duration_factors"] == factors and meta["candidate_factors"] == factors * 2 and meta["rows"] == 12
    for text, row in zip(texts, diverse):
        assert [r["duration_factor"] for r in row] == factors * 2
        for r in row:
            expected = tts.target_frames(60, voice.transcript, text, duration_scale=r["duration_factor"],
                                         duration_mode="clamp")[0]
            assert r["frames"] == expected and r["audio"].shape == (expected * 512,)
        assert row[1]["frames"] < row[0]["frames"] < row[2]["frames"]
        assert not np.array_equal(row[0]["audio"], row[3]["audio"])  # same length, different noise
    for bad in ([1.0, 0.9, 1.1, 1.2], [1.0, 0.0], [float("nan")], []):
        with pytest.raises(ValueError, match="duration_factors"):
            tts.synthesize_many(texts, voice, candidates=3, steps=1, duration_factors=bad)


def load_fit_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fit_duration_best_factor.py"
    spec = importlib.util.spec_from_file_location("fit_duration_best_factor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WORDS = "bugün hava çok güzel dışarı çıkalım mı yarın okula gideceğim kitap okumayı severim deniz kenarında".split()


def synthetic_pairs(count, seed=0):
    rng = np.random.default_rng(seed)
    pairs = []
    for index in range(count):
        prompt_text = " ".join(rng.choice(WORDS, rng.integers(5, 14))) + "."
        text = " ".join(rng.choice(WORDS, rng.integers(2, 16))) + "."
        rate = rng.uniform(10, 21)  # prompt speaking rate, characters per second
        pairs.append({"pair": index, "speaker": f"s{index % 7}", "prompt_index": index, "prompt_text": prompt_text,
                      "prompt_frames": len(prompt_text) / rate * 25, "text": text})
    return pairs


class FakeTTS:
    """synthesize_many stand-in: the 'audio' of a candidate is its factor, frames follow the byte rule."""

    text_version = "turkish-v1"
    codec = SimpleNamespace(sample_rate=16000)

    def __init__(self):
        self.calls = 0

    def synthesize_many(self, texts, voice, *, candidates, duration_factors, duration_mode, seed, **synthesis):
        self.calls += 1
        base = rule_frames(len(voice.latents), voice.transcript, texts[0])
        row = [{"audio": np.full(4, duration_factors[k % len(duration_factors)], np.float32),
                "frames": round(base * duration_factors[k % len(duration_factors)]),
                "duration_factor": duration_factors[k % len(duration_factors)]} for k in range(candidates)]
        return [row], {}


def fast_prompt_scorer(best_fast=1.15, best_other=1.0):
    """Fake scorer: CER grows with the distance to 1.15 for prompts of > 17 chars/s and to 1.0 otherwise."""

    def score(text, audios, sample_rate, prompt=None):
        best = best_fast if score.fast else best_other
        return [{"cer": abs(float(a[0]) - best), "score": abs(float(a[0]) - best)} for a in audios]

    score.fast = False
    return score


def test_fit_script_search_picks_best_factor_resumes_and_refits(tmp_path):
    from dacvae_tts.duration import speaking_rate

    fit_script = load_fit_script()
    factors = fit_script.parse_factors("1.0,0.85,1.15")
    with pytest.raises(ValueError):
        fit_script.parse_factors("1.0,1.0")
    pairs = synthetic_pairs(80)
    tts, scorer = FakeTTS(), fast_prompt_scorer()

    def latents(pair):  # prompt_frames rounded to whole latent frames; flags the scorer's regime for this pair
        scorer.fast = speaking_rate(round(pair["prompt_frames"]), pair["prompt_text"]) > 17
        return torch.zeros(round(pair["prompt_frames"]), 4)

    records_path = tmp_path / "records.jsonl"
    records = fit_script.search(tts, pairs[:50], latents, scorer, factors, "rule", records_path, seeds=2)
    assert tts.calls == 50 and len(fit_script.read_records(records_path)) == 50
    records = fit_script.search(tts, pairs, latents, scorer, factors, "rule", records_path, seeds=2)
    assert tts.calls == 80 and len(records) == 80  # the first 50 pairs were resumed, not synthesized again
    with pytest.raises(ValueError, match="another base mode or factor set"):
        fit_script.search(tts, pairs, latents, scorer, [1.0, 0.9], "rule", records_path)
    for record in records:
        fast = speaking_rate(record["prompt_frames"], record["prompt_text"]) > 17
        assert record["best_factor"] == (1.15 if fast else 1.0)
        assert record["target_frames"] == record["factor_frames"][str(record["best_factor"])]
        assert record["base_frames"] == record["factor_frames"]["1.0"] and len(record["candidates"]) == 6
    predictor, report = fit_script.refit(records, holdout=0.2)
    assert report["fit_pairs"] == 64 and report["holdout_pairs"] == 16
    assert report["fast_prompts_mean_log_best_factor"] == pytest.approx(math.log(1.15))
    assert report["normal_prompts_mean_log_best_factor"] == 0.0
    assert report["fit_refit_mae_log"] < report["fit_base_mae_log"]
    path = tmp_path / "predictor.json"
    predictor.save(path)
    loaded = DurationPredictor.load(path)
    assert loaded.weights == predictor.weights and loaded.metadata["best_factor_counts"]


def test_fit_script_refit_recovers_a_uniform_best_factor():
    fit_script = load_fit_script()
    records = []
    for pair in synthetic_pairs(200, seed=3):
        base = rule_frames(pair["prompt_frames"], pair["prompt_text"], pair["text"])
        records.append({**pair, "best_factor": 1.1, "base_frames": base, "target_frames": base * 1.1})
    records.append({"pair": 999, "error": "Target duration outside .25–30 s"})
    predictor, report = fit_script.refit(records, ridge=1e-6)
    assert report["failed"] == 1 and report["changed_fraction"] == 1.0
    for record in records[:-1]:
        predicted = predictor.predict(record["prompt_frames"], record["prompt_text"], record["text"])
        assert predicted == pytest.approx(record["target_frames"], rel=0.02)


def test_fit_script_samples_same_speaker_pairs(cache):
    from dacvae_tts.data import LatentDataset

    fit_script = load_fit_script()
    data = LatentDataset(cache, "train", pairing="within", layout="joined")
    pairs = fit_script.sample_pairs(data, "train", 20, seed=1, min_prompt=1, max_prompt=100)
    assert len(pairs) == 12  # every train recording is a target once (4 speakers x 3 recordings)
    assert len({p["uid"] for p in pairs}) == 12
    for pair in pairs:
        row = data.row(pair["prompt_index"])
        assert pair["prompt_uid"] != pair["uid"] and row["uid"] == pair["prompt_uid"]
        assert row["speaker"] == pair["speaker"] == pair["uid"].rsplit("-", 1)[0]
        assert pair["prompt_frames"] == len(row["latents"])
