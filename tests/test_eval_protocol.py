"""Evaluation protocol v2 (issue #3): pure metrics, SIM-o loader checks and evaluator wiring, without heavy models."""

import argparse
import importlib.metadata
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from dacvae_tts import eval_protocol, metrics, sim_o
from dacvae_tts.eval_protocol import (
    KNOWN_HALLUCINATIONS,
    ProtocolOptions,
    ProtocolScorer,
    add_protocol_args,
    band_limit,
    bandwidth_hz,
    clipped_fraction,
    find_hallucinations,
    manifest_prompts,
    protocol_from_args,
    signal_stats,
    summary_extras,
    trim_trailing_silence,
)
from dacvae_tts.metrics import error_counts, freya_error_counts, metric_text, summarize


def band_limited_noise(cutoff, rate, seconds=2.0, seed=0):
    noise = np.random.default_rng(seed).standard_normal(int(rate * seconds))
    spectrum = np.fft.rfft(noise)
    spectrum[np.fft.rfftfreq(len(noise), 1 / rate) > cutoff] = 0
    return np.fft.irfft(spectrum, len(noise))


def speech_like(rate=16000, seconds=1.0, silence=1.0, seed=0):
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(int(rate * seconds)) / rate)
    tone += 0.05 * np.random.default_rng(seed).standard_normal(len(tone))
    return np.concatenate([tone, np.zeros(int(rate * silence))]).astype(np.float32)


# ------------------------------------------------------------------------------------------------ pure functions


def test_bandwidth_of_band_limited_noise():
    assert abs(bandwidth_hz(band_limited_noise(6000, 48000), 48000) - 6000) < 150
    assert abs(bandwidth_hz(band_limited_noise(3000, 16000), 16000) - 3000) < 150
    assert bandwidth_hz(np.random.default_rng(1).standard_normal(48000), 48000) == 24000
    assert bandwidth_hz(np.zeros(4000), 16000) == 0.0
    assert bandwidth_hz(np.ones(10), 16000) >= 0.0  # shorter than one FFT frame: padded, no crash


def test_clipped_fraction_counts_every_channel():
    assert clipped_fraction(np.array([1.0, -1.0, 0.5, 0.9995])) == 0.75
    assert clipped_fraction(np.array([[0.999, 0.0], [0.1, -0.2]])) == 0.25
    assert clipped_fraction(np.zeros(0)) == 0.0


def test_band_limit_8k_keeps_rate_length_and_removes_the_top_band():
    audio = np.random.default_rng(2).standard_normal(16000 * 2).astype(np.float32)
    limited = band_limit(audio)
    assert limited.shape == audio.shape and limited.dtype == np.float32
    assert bandwidth_hz(audio, 16000) == 8000
    assert bandwidth_hz(limited, 16000) < 5000  # 4 kHz Nyquist of the 8 kHz detour plus the filter's transition
    tensor = band_limit(torch.from_numpy(audio))
    assert torch.is_tensor(tensor) and tensor.shape == (32000,)
    np.testing.assert_allclose(tensor.numpy(), limited, atol=1e-6)
    odd = band_limit(np.ones(16001, dtype=np.float32))
    assert odd.shape == (16001,)


def test_trailing_silence_trim_keeps_the_start():
    audio = speech_like()
    trimmed, removed = trim_trailing_silence(audio)
    assert 0.8 * 16000 < removed < 16000 and len(trimmed) + removed == len(audio)
    np.testing.assert_array_equal(trimmed, audio[: len(trimmed)])
    tensor, removed_tensor = trim_trailing_silence(torch.from_numpy(audio))
    assert torch.is_tensor(tensor) and removed_tensor == removed


def test_hallucination_flag_removes_only_excess_phrases():
    filtered, removed = find_hallucinations("Merhaba dünya. Altyazı M.K.", "Merhaba dünya.")
    assert filtered == "merhaba dünya" and removed == ["Altyazı M.K."]
    assert find_hallucinations("ALTYAZI M. K", "Bir cümle.")[1] == ["Altyazı M.K."]  # case/punctuation-insensitive
    # Occurring in the reference makes the phrase legitimate; only the extra copy is removed.
    assert find_hallucinations("İzlediğiniz için teşekkürler", "İzlediğiniz için teşekkürler") == (
        "izlediğiniz için teşekkürler", []
    )
    filtered, removed = find_hallucinations(
        "İzlediğiniz için teşekkürler. İzlediğiniz için teşekkürler!", "İzlediğiniz için teşekkürler."
    )
    assert filtered == "izlediğiniz için teşekkürler" and removed == ["İzlediğiniz için teşekkürler"]
    assert find_hallucinations("Thanks for watching!", "hello", "english-unicode-v2") == ("", ["Thanks for watching"])
    assert find_hallucinations("teşekkürler", "Bir cümle.")[1] == []  # single words are never flagged
    assert all(len(phrase.split()) >= 2 for phrase in KNOWN_HALLUCINATIONS)


def test_signal_stats_on_a_file(tmp_path):
    path = tmp_path / "a.wav"
    audio = 0.5 * band_limited_noise(4000, 48000, seconds=1.0)
    audio[:10] = 1.0
    sf.write(path, np.stack([audio, audio * 0.5], 1), 48000, subtype="FLOAT")
    stats = signal_stats(path)
    assert stats["clipped_fraction_fullband"] == pytest.approx(10 / (2 * 48000))
    assert abs(stats["bandwidth_hz"] - 4000) < 150
    assert -40 < stats["loudness_lufs"] < 0
    sf.write(tmp_path / "short.wav", np.full(1000, 0.1), 48000)
    assert signal_stats(tmp_path / "short.wav")["loudness_lufs"] is None  # < 0.4 s: undefined, not a fake number


# ------------------------------------------------------------------------------------------------------ summaries


def test_summarize_keeps_v1_keys_and_adds_per_utterance_stats():
    a = error_counts("bir iki üç dört", "bir iki üç dört", "turkish-v1")
    b = error_counts("beş", "altı yedi", "turkish-v1")
    rows = [{**a, "dnsmos_ovrl": 3.0, "speaker_similarity": 0.9}, {**b, "dnsmos_ovrl": 2.0, "speaker_similarity": 0.8}]
    summary = summarize(rows)
    # v1 keys and values are unchanged (corpus rates, means of the v1 metrics).
    assert summary["count"] == 2
    assert summary["wer"] == pytest.approx(2 / 5)
    assert summary["cer"] == pytest.approx((a["char_edits"] + b["char_edits"]) / (a["chars"] + b["chars"]))
    assert summary["dnsmos_ovrl"] == 2.5 and summary["speaker_similarity"] == pytest.approx(0.85)
    # Added keys.
    assert summary["wer_mean"] == pytest.approx((0 + 2) / 2)
    assert summary["cer_mean"] == pytest.approx((a["cer"] + b["cer"]) / 2)
    assert summary["wer_over_half_fraction"] == 0.5 and summary["error_free_fraction"] == 0.5
    assert (summary["word_substitutions"], summary["word_deletions"], summary["word_insertions"]) == (1, 0, 1)
    assert not any(key.startswith(("sim_", "utmos", "hallucination")) for key in summary)


def test_summary_extras_partial_metrics_and_filtered_wer():
    a = {**error_counts("bir iki", "bir iki"), "sim_o": 0.6, "utmos": float("nan"), "loudness_lufs": None}
    b = {**error_counts("üç dört", "üç"), "sim_o": 0.4, "sim_r": 0.7}
    extras = summary_extras([a, b])
    assert extras["sim_o"] == pytest.approx(0.5) and "sim_o_rows" not in extras
    assert extras["sim_r"] == 0.7 and extras["sim_r_rows"] == 1  # partial coverage is reported, not hidden
    assert "utmos" not in extras and "loudness_lufs" not in extras
    assert "wer_filtered" not in extras
    for row, raw in ((a, "bir iki"), (b, "üç dört")):
        counts = error_counts(raw, raw)
        row.update(word_edits_filtered=counts["word_edits"], char_edits_filtered=counts["char_edits"],
                   hallucination=row is b)
    extras = summary_extras([a, b])
    assert extras["wer_filtered"] == 0 and extras["cer_filtered"] == 0 and extras["hallucination_rate"] == 0.5


def test_freya_metric_splits_at_apostrophes_and_counts_spaces():
    reference, hypothesis = "İstanbul'da 2010'da kaldık.", "İstanbul'da iki bin onda kaldık"
    ours = error_counts(reference, hypothesis, "turkish-v1")
    assert (ours["words"], ours["chars"], ours["wer"]) == (5, 26, 0)  # istanbulda iki bin onda kaldık
    freya = freya_error_counts(reference, hypothesis, "turkish-v1")
    # istanbul da iki bin onda kaldık: the apostrophe is a word break, CER counts the 5 spaces.
    assert (freya["freya_words"], freya["freya_chars"], freya["freya_wer"]) == (6, 31, 0)
    joined = freya_error_counts(reference, "istanbulda iki bin onda kaldık", "turkish-v1")
    assert joined["freya_word_edits"] == 2 and joined["freya_char_edits"] == 1  # 0 edits under our convention
    assert metric_text("Don't stop", apostrophe=" ") == "don t stop" and metric_text("Don't stop") == "don't stop"
    assert metric_text("İsveç'ten", "turkish-v2", " ") == "isveç ten" and metric_text("İsveç'ten", "turkish-v2") == (
        "isveçten"
    )
    with pytest.raises(ValueError, match="empty"):
        freya_error_counts("...", "bir")
    rows = [{**error_counts("bir iki", "bir iki"), **freya_error_counts("bir iki", "bir")},
            {**error_counts("üç", "üç"), **freya_error_counts("üç", "üç")}]
    extras = summary_extras(rows)
    assert extras["freya_wer"] == pytest.approx(1 / 3) and extras["freya_cer"] == pytest.approx(4 / 9)
    assert "freya_wer" not in summary_extras([error_counts("bir", "bir")])


# ---------------------------------------------------------------------------------------------- options / flags


def parse(*flags, dnsmos=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dnsmos")
    add_protocol_args(parser)
    return parser.parse_args([*flags, *(["--dnsmos", dnsmos] if dnsmos else [])])


def test_protocol_is_off_by_default():
    assert protocol_from_args(parse()) is None
    assert protocol_from_args(SimpleNamespace()) is None  # callers without the flags
    assert not ProtocolOptions().enabled


def test_protocol_v2_bundle_and_validation():
    options = protocol_from_args(parse("--protocol-v2"))
    assert options.asr_deterministic and options.sim_o and options.sim_speechbrain and options.utmos
    assert options.signal_stats and options.flag_hallucinations
    assert not (options.prompt_dnsmos or options.utmosv2 or options.band_limit_8k or options.asr_trim_silence)
    assert protocol_from_args(parse("--protocol-v2", dnsmos="x.onnx")).prompt_dnsmos
    options = protocol_from_args(parse("--band-limit-8k", "--sim-o", "--sim-o-backend", "s3prl"))
    assert options.band_limit_8k and options.sim_o and options.sim_o_backend == "s3prl" and not options.utmos
    # Any active protocol decodes deterministically; --no-asr-deterministic opts out, and alone changes nothing.
    assert options.deterministic_asr and options.asr_deterministic is None
    assert protocol_from_args(parse("--asr-deterministic")).deterministic_asr
    assert not protocol_from_args(parse("--band-limit-8k", "--no-asr-deterministic")).deterministic_asr
    assert not protocol_from_args(parse("--protocol-v2", "--no-asr-deterministic")).deterministic_asr
    assert protocol_from_args(parse("--no-asr-deterministic")) is None
    with pytest.raises(ValueError, match="DNSMOS"):
        protocol_from_args(parse("--prompt-dnsmos"))
    with pytest.raises(ValueError, match="sim-o"):
        protocol_from_args(parse("--sim-o-checkpoint", "x.pth"))
    with pytest.raises(ValueError, match="backend"):
        ProtocolOptions(sim_o=True, sim_o_backend="unknown")
    with pytest.raises(ValueError, match="disabled"):
        ProtocolOptions(sim_o_checkpoint="x.pth")
    assert eval_protocol.SIM_O_BACKBONES == sim_o.BACKBONES


def test_manifest_prompts_never_guess_the_reference_kind():
    row = {"reference_audio": "ref.wav"}
    assert manifest_prompts(row) == {"original_prompt": None, "codec_prompt": None}
    assert manifest_prompts(row, "original") == {"original_prompt": "ref.wav", "codec_prompt": None}
    assert manifest_prompts(row, "codec") == {"original_prompt": None, "codec_prompt": "ref.wav"}
    explicit = {**row, "original_prompt_audio": "orig.wav", "codec_prompt_audio": "codec.wav"}
    assert manifest_prompts(explicit, "original") == {"original_prompt": "orig.wav", "codec_prompt": "codec.wav"}
    with pytest.raises(ValueError):
        manifest_prompts(row, "maybe")


# ------------------------------------------------------------------------------------------------ scorer wiring


class StubSpeaker:
    """Embeds a file as its normalized first-8-sample vector; counts prompt computations."""

    identity = {"model": "stub"}

    def __init__(self):
        self.calls = []

    def file_embedding(self, path, cache=False):
        self.calls.append((str(path), cache))
        audio = sf.read(str(path), dtype="float32", always_2d=True)[0][:8, 0]
        return torch.nn.functional.normalize(torch.from_numpy(audio + 1.0), dim=0)


def write(path, audio, rate=16000):
    sf.write(path, audio, rate, subtype="FLOAT")
    return path


def test_scorer_reports_sim_o_and_sim_r_separately(tmp_path):
    generated = write(tmp_path / "gen.wav", speech_like(seed=1))
    original = write(tmp_path / "orig.wav", speech_like(seed=1))
    codec = write(tmp_path / "codec.wav", speech_like(seed=9))
    dnsmos_calls = []

    def dnsmos(audio):
        dnsmos_calls.append(len(audio))
        return {"dnsmos_sig": 3.0, "dnsmos_bak": 4.0, "dnsmos_ovrl": 2.5}

    options = ProtocolOptions(sim_o=True, sim_speechbrain=True, utmos=True, utmosv2=True, prompt_dnsmos=True,
                              signal_stats=True, flag_hallucinations=True)
    speaker, second = StubSpeaker(), StubSpeaker()
    scorer = ProtocolScorer(options, dnsmos=dnsmos, sim_o=speaker, speechbrain=second, utmos=lambda a: 4.1,
                            utmosv2=lambda a: 3.3)
    audio = torch.from_numpy(speech_like(seed=1))
    row = scorer.score(generated, audio, "Bir cümle.", "Bir cümle. Altyazı M.K.", "turkish-v1", original, codec)
    assert row["sim_o"] == pytest.approx(1.0) and row["sim_r"] < 1.0
    assert row["sim_o_speechbrain"] == pytest.approx(1.0) and "sim_r_speechbrain" in row
    assert (row["utmos"], row["utmosv2"]) == (4.1, 3.3)
    assert row["prompt_dnsmos_ovrl"] == 2.5 and "dnsmos_ovrl" not in row
    assert row["hallucination"] and row["hallucination_phrases"] == ["Altyazı M.K."]
    assert row["wer_filtered"] == 0 and row["hypothesis_filtered"] == "bir cümle"
    assert {"clipped_fraction_fullband", "loudness_lufs", "bandwidth_hz", "prompt_bandwidth_hz"} <= row.keys()
    scorer.score(generated, audio, "Bir cümle.", "Bir cümle.", "turkish-v1", original, codec)
    assert len(dnsmos_calls) == 1  # prompt DNSMOS is computed once per prompt
    assert all(cache for path, cache in speaker.calls if "gen" not in path)  # prompt embeddings cached
    alone = scorer.score(generated, audio, "Bir cümle.", "Bir cümle.", "turkish-v1")
    assert not any(k.startswith(("sim_", "prompt_")) for k in alone)  # no prompt, no SIM: never a fallback
    assert scorer.identity["options"]["sim_o"] and scorer.identity["sim_o"] == {"model": "stub"}
    with pytest.raises(ValueError, match="DNSMOS"):
        ProtocolScorer(ProtocolOptions(prompt_dnsmos=True))


def test_scorer_asr_input_and_decoding():
    audio = torch.from_numpy(speech_like())
    plain = ProtocolScorer(ProtocolOptions(signal_stats=True))
    assert plain.asr_audio(audio)[0] is audio and plain.whisper_kwargs() == eval_protocol.WHISPER_DETERMINISTIC
    sampling = ProtocolScorer(ProtocolOptions(signal_stats=True, asr_deterministic=False))
    assert sampling.whisper_kwargs() == eval_protocol.WHISPER_V1
    scorer = ProtocolScorer(ProtocolOptions(asr_deterministic=True, asr_trim_silence=True, band_limit_8k=True))
    trimmed, info = scorer.asr_audio(audio)
    assert len(trimmed) < len(audio) and 0.8 < info["asr_trimmed_seconds"] < 1.0
    assert scorer.whisper_kwargs() == dict(beam_size=5, vad_filter=False, condition_on_previous_text=False,
                                           temperature=0.0, without_timestamps=True)
    assert scorer.identity["band_limit_rate"] == 8000


# --------------------------------------------------------------------------------------------- evaluator wiring


class FakeWhisper:
    calls = []
    text = "Bir iki üç."

    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio, **kwargs):
        FakeWhisper.calls.append((len(audio), kwargs))
        return [SimpleNamespace(text=FakeWhisper.text)], None


@pytest.fixture
def fake_whisper(monkeypatch):
    FakeWhisper.calls = []
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisper))
    real = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda name: "0.0-test" if name in ("faster-whisper", "ctranslate2") else real(name))
    return FakeWhisper


V1_IDENTITY = {"asr_model", "faster_whisper_version", "speaker_model", "speaker_revision", "dnsmos_sha256",
               "metric_normalization", "language", "asr_backend", "device", "compute_type", "decoding"}
V1_ROW = {"word_edits", "words", "char_edits", "chars", "wer", "cer", "word_substitutions", "word_deletions",
          "word_insertions", "hypothesis", "audio_seconds", "clipped_fraction", "evaluator"}


def test_evaluator_without_protocol_is_unchanged(fake_whisper, tmp_path):
    path = write(tmp_path / "gen.wav", speech_like())
    evaluator = metrics.Evaluator("large-v3", None, None, "cpu", language="tr")
    assert set(evaluator.identity) == V1_IDENTITY and evaluator.protocol is None
    result = evaluator.score(path, "Bir iki üç.", original_prompt=path, codec_prompt=path)
    assert set(result) == V1_ROW and result["wer"] == 0
    length, kwargs = fake_whisper.calls[-1]
    assert length == 32000
    assert kwargs == dict(language="tr", beam_size=5, vad_filter=False, condition_on_previous_text=False)
    disabled = metrics.Evaluator("large-v3", None, None, "cpu", language="tr", protocol=ProtocolOptions())
    assert set(disabled.identity) == V1_IDENTITY  # an all-off protocol is v1
    # The compact identity kept in every script row: what changes WER/CER/SIM for the same audio.
    assert evaluator.row_identity == {
        "asr_backend": "faster-whisper", "asr_model": "large-v3", "language": "tr", "metric_normalization": "turkish-v1",
        "decoding": eval_protocol.WHISPER_V1, "compute_type": "int8", "device": "cpu", "speaker_model": None,
        "protocol_options": None,
    }
    assert metrics.row_identity(evaluator.row_identity) == evaluator.row_identity == metrics.row_identity(
        evaluator.identity
    )
    assert metrics.row_identity(None) is None


def test_evaluator_with_protocol(fake_whisper, tmp_path, monkeypatch):
    monkeypatch.setattr(sim_o, "SimO", lambda checkpoint, backend, device: StubSpeaker())
    monkeypatch.setattr(eval_protocol, "UTMOS22Strong", lambda device: (lambda audio: 3.9))
    path = write(tmp_path / "gen.wav", speech_like())
    original = write(tmp_path / "orig.wav", speech_like(seed=4))
    fake_whisper.text = "Bir iki üç. İzlediğiniz için teşekkürler."
    options = ProtocolOptions(asr_deterministic=True, asr_trim_silence=True, band_limit_8k=True,
                              flag_hallucinations=True, signal_stats=True, sim_o=True, utmos=True)
    evaluator = metrics.Evaluator("large-v3", None, None, "cpu", language="tr", protocol=options)
    protocol = evaluator.identity["protocol"]
    assert protocol["whisper_decoding"]["temperature"] == 0.0 and protocol["asr_snapshot"] is None
    assert protocol["ctranslate2_version"] == "0.0-test" and protocol["options"]["band_limit_8k"]
    assert evaluator.row_identity["protocol_options"]["band_limit_8k"]
    assert evaluator.row_identity["decoding"] == evaluator.identity["decoding"] == protocol["whisper_decoding"]
    result = evaluator.score(path, "Bir iki üç.", original_prompt=original)
    length, kwargs = fake_whisper.calls[-1]
    assert length < 32000 and kwargs["temperature"] == 0.0 and kwargs["without_timestamps"]
    assert V1_ROW <= set(result)
    assert result["wer"] > 0 and result["wer_filtered"] == 0 and result["hallucination"]  # raw WER kept
    assert result["utmos"] == 3.9 and "sim_o" in result and "sim_r" not in result
    assert result["asr_trimmed_seconds"] > 0.8


def test_any_protocol_decodes_deterministically(fake_whisper, tmp_path):
    """--band-limit-8k or --sim-o alone used to keep the sampling fallback (only --asr-deterministic/--protocol-v2
    pinned temperature 0); without a protocol the v1 decoding is unchanged."""
    path = write(tmp_path / "gen.wav", speech_like())
    for options in (ProtocolOptions(band_limit_8k=True), ProtocolOptions(signal_stats=True)):
        evaluator = metrics.Evaluator("large-v3", None, None, "cpu", language="tr", protocol=options)
        evaluator.score(path, "Bir iki üç.")
        assert fake_whisper.calls[-1][1]["temperature"] == 0.0 and fake_whisper.calls[-1][1]["without_timestamps"]
        assert evaluator.row_identity["decoding"] == eval_protocol.WHISPER_DETERMINISTIC
    evaluator = metrics.Evaluator("large-v3", None, None, "cpu", language="tr",
                                  protocol=ProtocolOptions(band_limit_8k=True, asr_deterministic=False))
    evaluator.score(path, "Bir iki üç.")
    assert "temperature" not in fake_whisper.calls[-1][1]
    metrics.Evaluator("large-v3", None, None, "cpu", language="tr").score(path, "Bir iki üç.")
    assert fake_whisper.calls[-1][1] == dict(language="tr", **eval_protocol.WHISPER_V1)


def test_cli_evaluate_with_protocol(fake_whisper, tmp_path, monkeypatch):
    from dacvae_tts import cli

    fake_whisper.text = "Bir iki üç."
    audio = write(tmp_path / "gen.wav", speech_like())
    reference = write(tmp_path / "ref.wav", speech_like(seed=2))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"audio": str(audio), "text": "Bir iki üç.", "reference_audio": str(reference)})
                        + "\n")
    output = tmp_path / "scores.jsonl"
    monkeypatch.setattr(sim_o, "SimO", lambda checkpoint, backend, device: StubSpeaker())
    monkeypatch.setattr(sys, "argv", ["dacvae-tts", "evaluate", "--manifest", str(manifest), "--output", str(output),
                                      "--language", "tr", "--no-speaker", "--sim-o", "--signal-stats",
                                      "--reference-kind", "original"])
    cli.main()
    row = json.loads(output.read_text())
    assert -1 <= row["sim_o"] <= 1 and "sim_r" not in row and "speaker_similarity" not in row
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["wer"] == 0 and summary["wer_mean"] == 0 and "sim_o" in summary and "bandwidth_hz" in summary
    assert summary["evaluator"]["protocol"]["options"] == {"sim_o": True, "sim_o_backend": "transformers",
                                                           "signal_stats": True}


def test_eval_sentences_rescore_with_protocol(fake_whisper, tmp_path, monkeypatch):
    """scripts/eval_sentences.py --rescore: sim_r vs the codec prompt WAV, sim_o only vs the exported original."""
    import importlib.util
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("eval_sentences_under_test", scripts / "eval_sentences.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = [{"speaker": "spk", "prompt_uid": "shard/a.parquet:7", "prompt_text": "Önceki cümle.", "prompt_index": 0,
              "uid": "shard/a.parquet:8", "text": "Hedef.", "target_index": 1, "ground_truth_seconds": 1.0}]
    data = SimpleNamespace(row=lambda index: {"latents": torch.zeros(5, 4)})
    monkeypatch.setattr(module, "select_cases", lambda cache, count, seed, exclude=(): (data, cases))
    monkeypatch.setattr(sim_o, "SimO", lambda checkpoint, backend, device: StubSpeaker())
    out, originals = tmp_path / "out", tmp_path / "originals"
    out.mkdir()
    originals.mkdir()
    sentences = tmp_path / "sentences.jsonl"
    rows = ({"id": "s0", "text": "Bir iki üç."}, {"id": "s1", "text": "Dört beş."})
    sentences.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write(out / "s0.wav", speech_like(48000), 48000)
    write(out / "s1.wav", speech_like(48000, seed=3), 48000)
    for name in ("s0", "s1"):  # synthesis sidecars of the first pass
        (out / f"{name}.json").write_text(json.dumps({"audio_seconds": 2.0, "rtf": 0.1}))
    write(out / "prompt-00.wav", speech_like(48000, seed=5), 48000)  # codec-decoded prompt of the first pass
    write(originals / "shard_a.parquet_7.wav", speech_like(48000), 48000)
    fake_whisper.text = "Bir iki üç."
    monkeypatch.setattr(sys, "argv", [
        "eval_sentences.py", "--checkpoint", "unused.pt", "--cache", "unused", "--sentences", str(sentences),
        "--output", str(out), "--prompts", "1", "--rescore", "--asr-device", "cpu", "--speaker-model", "",
        "--prompt-audio", str(originals), "--sim-o", "--signal-stats", "--flag-hallucinations",
        "--asr-deterministic", "--band-limit-8k",
    ])
    module.main()
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["s0", "s1"]
    assert all(r["prompt_original"].endswith("shard_a.parquet_7.wav") for r in rows)
    assert rows[0]["sim_o"] == pytest.approx(1.0) and rows[0]["sim_r"] < 1.0
    assert rows[0]["wer"] == 0 and rows[1]["wer"] > 0 and "wer_filtered" in rows[1]
    assert fake_whisper.calls[-1][1]["temperature"] == 0.0
    # Every row keeps the compact scorer identity, so compare_evals can refuse mismatched scorers.
    identity = rows[0]["evaluator"]
    assert identity == rows[1]["evaluator"] and identity["protocol_options"]["band_limit_8k"]
    assert (identity["metric_normalization"], identity["device"], identity["compute_type"]) == (
        "turkish-v1", "cpu", "int8"
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["protocol"]["band_limit_rate"] == 8000 and summary["prompt_audio"] == str(originals)
    assert {"sim_o", "sim_r", "wer_mean", "cer_mean", "bandwidth_hz", "per_sentence_wer_mean"} <= summary.keys()
    assert json.loads((out / "cases.json").read_text()) == cases


@pytest.fixture
def fake_transformers(monkeypatch):
    """transformers stand-ins for eval_sentences.score_hf: Whisper transcribes every clip as FakeWhisper.text (the
    clip lengths it was given go to `clips`), x-vectors are the clip's first samples."""
    clips = []

    class Processor:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return cls()

        def __call__(self, audios, sampling_rate, return_tensors):
            clips.extend(len(a) / sampling_rate for a in audios)
            return SimpleNamespace(input_features=torch.zeros(len(audios), 1))

        def batch_decode(self, ids, skip_special_tokens=True):
            return [FakeWhisper.text] * len(ids)

    class Model:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return cls()

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

        def generate(self, features, **kwargs):
            FakeWhisper.calls.append((features.shape[0], kwargs))
            return torch.zeros(features.shape[0], 1, dtype=torch.long)

    class Extractor(Processor):
        def __call__(self, audio, sampling_rate, return_tensors, padding):
            return {"input_values": torch.as_tensor(audio[:8] + 1.0)[None]}

    class XVector(Model):
        def __call__(self, input_values):
            return SimpleNamespace(embeddings=input_values)

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        WhisperForConditionalGeneration=Model, WhisperProcessor=Processor, AutoFeatureExtractor=Extractor,
        AutoModelForAudioXVector=XVector,
    ))
    return clips


EVAL_CASES = [{"speaker": "spk", "prompt_uid": "shard/a.parquet:7", "prompt_text": "Önceki cümle.", "prompt_index": 0,
               "uid": "shard/a.parquet:8", "text": "Hedef.", "target_index": 1, "ground_truth_seconds": 1.0}]


def run_eval_sentences(tmp_path, monkeypatch, *flags, texts=("Bir iki üç.", "Dört beş."), output="out",
                       cases=EVAL_CASES, seconds=None, synthesizer=None):
    """scripts/eval_sentences.py --rescore over WAVs of a first pass (one prompt), or a synthesis pass with the
    given `synthesizer` class; returns (rows, summary, out)."""
    import importlib.util
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("eval_sentences_under_test", scripts / "eval_sentences.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = SimpleNamespace(row=lambda index: {"latents": torch.zeros(5, 4)})
    monkeypatch.setattr(module, "select_cases", lambda cache, count, seed, exclude=(): (data, cases))
    if synthesizer is not None:
        monkeypatch.setattr(module, "Synthesizer", synthesizer)
    out = tmp_path / output
    out.mkdir(exist_ok=True)
    sentences = tmp_path / f"{output}.jsonl"
    sentences.write_text("".join(json.dumps({"id": f"s{i}", "text": t}, ensure_ascii=False) + "\n"
                                 for i, t in enumerate(texts)))
    for i in range(len(texts)):
        if not (out / f"s{i}.wav").exists():
            write(out / f"s{i}.wav", speech_like(48000, seconds or 1.0, seed=i), 48000)
            (out / f"s{i}.json").write_text(json.dumps({"audio_seconds": 2.0, "rtf": 0.1}))
    if not (out / "prompt-00.wav").exists():
        write(out / "prompt-00.wav", speech_like(48000, seed=5), 48000)
    monkeypatch.setattr(sys, "argv", [
        "eval_sentences.py", "--checkpoint", "unused.pt", "--cache", "unused", "--sentences", str(sentences),
        "--output", str(out), "--prompts", "1", *([] if synthesizer else ["--rescore"]), "--asr-device", "cpu",
        "--device", "cpu", *flags,
    ])
    module.main()
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    return rows, json.loads((out / "summary.json").read_text()), out


def test_eval_sentences_metric_normalization_reaches_both_asr_paths(fake_whisper, fake_transformers, tmp_path,
                                                                  monkeypatch):
    fake_whisper.text = "Saat dörde kadar."
    texts = ("Saat 4'e kadar.",)  # turkish-v1 reads the reference as "dörte", turkish-v2 as "dörde"
    rows, summary, _ = run_eval_sentences(tmp_path, monkeypatch, "--speaker-model", "", texts=texts, output="v1")
    assert rows[0]["wer"] > 0 and rows[0]["evaluator"]["metric_normalization"] == "turkish-v1"
    assert summary["metric_normalization"] == "turkish-v1"  # the default for --language tr is unchanged
    rows, summary, _ = run_eval_sentences(tmp_path, monkeypatch, "--speaker-model", "", "--metric-normalization",
                                          "turkish-v2", texts=texts, output="v2")
    assert rows[0]["wer"] == 0 and rows[0]["evaluator"]["metric_normalization"] == "turkish-v2"
    rows, summary, _ = run_eval_sentences(tmp_path, monkeypatch, "--asr-backend", "hf", "--metric-normalization",
                                          "turkish-v2", texts=texts, output="hf-v2")
    assert rows[0]["asr_backend"] == "hf-greedy" and rows[0]["wer"] == 0
    assert rows[0]["evaluator"]["metric_normalization"] == summary["metric_normalization"] == "turkish-v2"
    rows, summary, _ = run_eval_sentences(tmp_path, monkeypatch, "--asr-backend", "hf", texts=texts, output="hf")
    assert rows[0]["wer"] > 0 and rows[0]["evaluator"]["metric_normalization"] == "turkish-v1"
    # score_hf used turkish-v1 for every --language; it now follows the language like the faster-whisper path.
    fake_whisper.text = "Hello there."
    rows, summary, _ = run_eval_sentences(tmp_path, monkeypatch, "--asr-backend", "hf", "--language", "en",
                                          texts=("Hello there.",), output="hf-en")
    assert rows[0]["evaluator"]["metric_normalization"] == summary["metric_normalization"] == "english-unicode-v2"
    assert fake_whisper.calls[-1][1]["language"] == "en"


def test_eval_sentences_freya_metric(fake_whisper, fake_transformers, tmp_path, monkeypatch):
    fake_whisper.text = "İstanbul'da kaldık."
    texts = ("İstanbul'da kaldık.", "Ankara'ya gittik.")
    for backend in ("faster-whisper", "hf"):
        rows, summary, _ = run_eval_sentences(tmp_path, monkeypatch, "--speaker-model", "", "--asr-backend", backend,
                                              "--freya-metric", texts=texts, output=f"freya-{backend}")
        assert rows[0]["wer"] == rows[0]["freya_wer"] == 0
        assert (rows[0]["words"], rows[0]["freya_words"], rows[0]["freya_chars"]) == (2, 3, 18)
        assert summary["freya_wer"] > 0 and summary["wer"] > 0 and "freya_cer" in summary
    rows, summary, _ = run_eval_sentences(tmp_path, monkeypatch, "--speaker-model", "", texts=texts, output="plain")
    assert "freya_wer" not in rows[0] and "freya_wer" not in summary  # opt-in


OTHER_CASES = [{**EVAL_CASES[0], "speaker": "spk2", "prompt_uid": "shard/b.parquet:3"}]


def test_eval_sentences_rescore_refuses_other_prompts(fake_whisper, tmp_path, monkeypatch):
    """prompt-NN.wav is named by position only: a --rescore whose --seed/--prompts/--exclude-speakers select other
    prompts than the synthesis pass must stop instead of scoring SIM against the wrong voice."""
    fake_whisper.text = "Bir iki üç."
    rows, _, out = run_eval_sentences(tmp_path, monkeypatch, "--speaker-model", "")
    assert all(r["prompt_uid"] == "shard/a.parquet:7" for r in rows)
    record = (out / "cases.json").read_text()
    with pytest.raises(SystemExit, match="differ from .*cases.json"):
        run_eval_sentences(tmp_path, monkeypatch, "--speaker-model", "", cases=OTHER_CASES)
    assert (out / "cases.json").read_text() == record  # the synthesis pass's record is kept
    (out / "cases.json").unlink()  # e.g. an output written before cases.json existed: the rows still tell
    with pytest.raises(SystemExit, match="2 rows .* other prompts"):
        run_eval_sentences(tmp_path, monkeypatch, "--speaker-model", "", cases=OTHER_CASES)
    rows, _, _ = run_eval_sentences(tmp_path, monkeypatch, "--speaker-model", "")  # the right prompts still work
    assert [r["prompt_uid"] for r in rows] == ["shard/a.parquet:7"] * 2


def test_eval_sentences_synthesis_rewrites_prompt_wavs(fake_whisper, tmp_path, monkeypatch):
    class StubSynthesizer:
        def __init__(self, checkpoint, device="cpu"):
            self.device, self.std, self.mean = "cpu", 1.0, 0.0
            self.codec = SimpleNamespace(sample_rate=48000, decode=lambda latents: torch.full((4800,), 0.25))

        def synthesize(self, text, reference=None, output=None, **kwargs):
            write(output, speech_like(48000), 48000)
            return SimpleNamespace(metadata={"audio_seconds": 2.0, "rtf": 0.1})

    fake_whisper.text = "Bir iki üç."
    out = tmp_path / "out"
    out.mkdir()
    write(out / "prompt-00.wav", speech_like(48000, seed=7), 48000)  # left by a pass with other prompts
    rows, _, _ = run_eval_sentences(tmp_path, monkeypatch, "--speaker-model", "", cases=OTHER_CASES,
                                    synthesizer=StubSynthesizer)
    np.testing.assert_allclose(sf.read(str(out / "prompt-00.wav"))[0], 0.25, atol=1e-4)
    assert [(r["prompt"], r["prompt_uid"], r["speaker"]) for r in rows] == [
        ("prompt-00.wav", "shard/b.parquet:3", "spk2")
    ] * 2
    assert json.loads((out / "cases.json").read_text()) == OTHER_CASES

