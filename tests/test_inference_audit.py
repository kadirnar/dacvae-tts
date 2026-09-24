import json
import sqlite3
from types import SimpleNamespace

import pytest
import torch

from dacvae_tts.config import Config, ModelConfig
from dacvae_tts.data import BucketBatchSampler, LatentDataset
from dacvae_tts.experiments import audit_cache, make_cases
from dacvae_tts.inference import Synthesizer
from dacvae_tts.metrics import error_counts, metric_text
from dacvae_tts.model import FlowTTS
from dacvae_tts.prepare import merge
from dacvae_tts.text import normalize, tokenize


class FakeCodec:
    def __init__(self, checkpoint, device):
        self.latent_dim, self.sample_rate, self.hop_length = 4, 24000, 512
        self.metadata = dict(
            checkpoint="test-codec",
            sample_rate=24000,
            hop_length=512,
            latent_dim=4,
            posterior="mean",
            weights_sha256="fixture",
            preprocessing="fixture",
        )

    def decode(self, z):
        return torch.zeros(len(z) * 512)


def test_audio_only_api_no_speaker_id(monkeypatch, cache, tmp_path):
    import dacvae_tts.inference as module

    data = LatentDataset(cache)
    config = Config(ModelConfig(latent_dim=4, width=16, depth=1, heads=2, text_depth=1))
    model = FlowTTS(config.model)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "ema": model.state_dict(),
            "config": config.to_dict(),
            "codec": data.meta,
            "mean": data.mean,
            "std": data.std,
        },
        path,
    )
    monkeypatch.setattr(module, "Codec", FakeCodec)
    tts = Synthesizer(path, device="cpu", precision="fp32", profile=True)
    calls = []
    monkeypatch.setattr(tts, "reference", lambda path: torch.zeros(9, 4))
    monkeypatch.setattr(tts, "transcribe_reference", lambda path: calls.append(path) or "Reference words.")
    result = tts.synthesize(
        "Target words.", ref_audio="reference.wav", seconds=0.5, steps=2, output=tmp_path / "out.wav"
    )
    assert calls == ["reference.wav"]
    assert result.metadata["reference_transcript_source"].startswith("asr:")
    assert result.metadata["branch_evaluations"] == 4 and result.metadata["forward_calls"] == 2
    assert (tmp_path / "out.wav").exists()
    prepared = tts.prepare_reference("reference.wav", reference_text="Exact transcript.")
    tts.synthesize("New words.", reference=prepared, seconds=0.5, steps=1, guidance=1)
    assert calls == ["reference.wav"]
    assert "iterative_generation_seconds" in result.metadata["profiled_stages"]


def test_asr_dependency_is_lazy_and_empty_transcripts_fail(monkeypatch):
    tts = Synthesizer.__new__(Synthesizer)
    tts.device = torch.device("cpu")
    tts.asr_model = "test"
    monkeypatch.setattr(tts, "reference", lambda path: torch.zeros(3, 4))
    monkeypatch.setattr(tts, "transcribe_reference", lambda path: "")
    with pytest.raises(ValueError, match="nonempty"):
        tts.prepare_reference("any.wav")


def test_reference_asr_accepts_waveform_pairs(monkeypatch):
    # prepare_reference documents (waveform, rate) input, but its ASR fallback read the pair as a file path
    # (TypeError: Invalid file) whenever no transcript was given.
    import numpy as np

    heard = []

    class FakeWhisper:
        def transcribe(self, audio, **options):
            heard.append((audio.dtype, audio.shape, options["language"]))
            return [SimpleNamespace(text=" Merhaba dünya. ")], None

    tts = Synthesizer.__new__(Synthesizer)
    tts.device = torch.device("cpu")
    tts.asr_model, tts.asr_language, tts._asr = "fake", "tr", FakeWhisper()
    monkeypatch.setattr(tts, "encode_reference", lambda audio, rate: torch.zeros(3, 4))
    wave = (0.1 * np.sin(np.arange(48000) / 10)).astype(np.float32)  # 2 s at 24 kHz
    prepared = tts.prepare_reference((wave, 24000))
    assert prepared.transcript == "Merhaba dünya." and prepared.transcript_source == "asr:fake"
    assert heard == [(np.float32, (32000,), "tr")]  # resampled to Whisper's 16 kHz
    assert tts.transcribe_reference((torch.from_numpy(wave), 16000)) == "Merhaba dünya."
    assert heard[-1][:2] == (np.float32, (48000,))


def test_text_versions_and_accent_preservation():
    assert normalize("  José’s   café. ") == "José's café."
    assert metric_text("JOSÉ’S café!") == "josé's café"
    assert error_counts("José", "Jose")["wer"] == 1
    assert (
        normalize("12/04/2026", "english-explicit-v2", spoken_text="April twelfth, twenty twenty six.")
        == "April twelfth, twenty twenty six."
    )
    for text in ("12/04/2026", "$12", "10 kg", "Dr. Smith", "5:30 pm"):
        with pytest.raises(ValueError, match="Ambiguous"):
            normalize(text, "english-explicit-v2")
    a, b = tokenize("Reference.", "Target.")
    assert a[0] == 1 and a[-1] == 3 and (a == 2).sum() == 1
    assert torch.equal(a, tokenize("Reference.", "Target.", version="unicode-v1")[0])
    assert error_counts("a b c", "a x c d")["word_substitutions"] == 1
    assert error_counts("a b c", "a x c d")["word_insertions"] == 1


def test_audit_reports_supplied_overlaps_and_sessions(cache, tmp_path):
    with sqlite3.connect(cache / "index.sqlite") as db:
        db.executemany(
            "INSERT INTO provenance VALUES (?,?,?,?,?,?,?)",
            [
                ("train-0-0", "Original", "Normalized", "session-A", "recording-A", 0.0, 2.0),
                ("train-0-1", "Original", "Normalized", "session-B", "recording-A", 1.0, 3.0),
            ],
        )
    output = tmp_path / "audit.json"
    audit_cache(SimpleNamespace(cache=cache, output=output, scan_latents=True))
    report = json.loads(output.read_text())
    assert report["session_metadata_rows"] == 2
    assert report["overlapping_clip_pairs"] == [["train-0-0", "train-0-1"]]
    assert not report["verified_integrity_checks_passed"]
    assert not report["corrupt_latent_rows"]


def test_merge_rejects_duplicate_label_conflicts(cache, tmp_path):
    import shutil

    other = tmp_path / "other"
    shutil.copytree(cache, other)
    with sqlite3.connect(other / "index.sqlite") as db:
        db.execute("UPDATE samples SET speaker='incorrect' WHERE uid='train-0-0'")
    with pytest.raises(ValueError, match="inconsistent"):
        merge(SimpleNamespace(inputs=[cache, other], output=tmp_path / "merged"))


def test_speaker_balancing_is_reproducible_and_optional(cache):
    import numpy as np

    data = LatentDataset(cache)
    counts = np.array([2, 2] + [10] * 10)
    sampler = BucketBatchSampler(data.costs, 2, seed=17, speaker_counts=counts, speaker_balance=1)
    assert list(sampler) == list(sampler)
    assert sampler.weights[0] / sampler.weights[-1] == pytest.approx(5)


def test_case_export_rejects_known_overlapping_reference_targets(cache, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.touch()
    with sqlite3.connect(cache / "index.sqlite") as db:
        db.execute("UPDATE samples SET audio=?", (str(audio),))
        for (uid,) in db.execute("SELECT uid FROM samples").fetchall():
            db.execute(
                "INSERT INTO provenance VALUES (?,?,?,?,?,?,?)",
                (uid, "text", "text", "session", "recording", 0, 2),
            )
    with pytest.raises(ValueError, match="No eligible"):
        make_cases(
            SimpleNamespace(
                cache=cache,
                split="val",
                seed=42,
                output=tmp_path / "cases.jsonl",
                cross_session=False,
                limit=10,
            )
        )


def test_synthesize_many_duration_modes_and_speaker_guidance(monkeypatch, cache, tmp_path):
    import dacvae_tts.inference as module
    from dacvae_tts.inference import VoiceReference

    data = LatentDataset(cache)
    config = Config(ModelConfig(latent_dim=4, width=16, depth=1, heads=2, text_depth=1, text_layout="joined",
                                duration="rule", positions="rope", prediction="edm"))
    model = FlowTTS(config.model)
    path = tmp_path / "model.pt"
    meta = {**data.meta, "text_normalization": "turkish-v1"}
    torch.save({"model": model.state_dict(), "ema": model.state_dict(), "config": config.to_dict(), "codec": meta,
                "mean": data.mean, "std": data.std}, path)
    monkeypatch.setattr(module, "Codec", FakeCodec)
    tts = Synthesizer(path, device="cpu", precision="fp32")
    voice = VoiceReference(torch.zeros(60, 4), "Hızlı hızlı konuşan bir referans kaydı burada duruyor.", "test", {})
    texts = ["Merhaba dünya.", "Bu biraz daha uzun ikinci bir cümle."]
    results, metadata = tts.synthesize_many(texts, voice, candidates=3, steps=2, guidance=2.0, max_rows=4)
    assert [len(r) for r in results] == [3, 3] and metadata["rows"] == 6
    assert results[1][0]["frames"] > results[0][0]["frames"]
    assert all(r["audio"].shape == (r["frames"] * 512,) for row in results for r in row)
    rule = tts.target_frames(60, voice.transcript, texts[1])[0]
    clamp, profile = tts.target_frames(60, voice.transcript, texts[1], duration_mode="clamp")
    assert clamp > rule and profile["duration_clamp_factor"] > 1  # 54 characters in 2.4 s: a fast prompt
    assert tts.target_frames(60, voice.transcript, texts[1], duration_mode="predictor")[0] > 0
    assert tts.target_frames(60, voice.transcript, texts[1], duration_mode="syllable")[0] > 0
    guided, _ = tts.synthesize_many(texts[:1], voice, steps=2, guidance=2.0, speaker_guidance=1.0)
    assert guided[0][0]["frames"] == results[0][0]["frames"]
    with pytest.raises(TypeError):
        tts.synthesize_many(texts, voice, unknown_option=1)


def test_synthesize_many_sizes_duration_head_models_like_synthesize(monkeypatch, cache, tmp_path):
    # Best-of-N used to size head models with the byte rule (17 vs 82 frames for one text), so single and
    # best-of-N evaluations of configs/small.yaml / tiny.yaml models compared different duration methods.
    import math

    import dacvae_tts.inference as module
    from dacvae_tts.inference import VoiceReference

    data = LatentDataset(cache)
    config = Config(ModelConfig(latent_dim=4, width=16, depth=1, heads=2, text_depth=1, duration="head"))
    model = FlowTTS(config.model)
    torch.nn.init.zeros_(model.duration[-1].weight)
    torch.nn.init.constant_(model.duration[-1].bias, math.log(3.0))  # three frames per target byte
    path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "ema": model.state_dict(), "config": config.to_dict(),
                "codec": data.meta, "mean": data.mean, "std": data.std}, path)
    monkeypatch.setattr(module, "Codec", FakeCodec)
    tts = Synthesizer(path, device="cpu", precision="fp32")
    voice = VoiceReference(torch.zeros(60, 4), "A reference transcript that is long enough.", "test", {})
    text = "Target words here."
    for scale in (1.0, 0.8):
        single = tts.make_batch(voice.latents, voice.transcript, text, duration_scale=scale)["prompt"].size(1) - 60
        assert single == round(3.0 * len(text.encode()) * scale)
        assert single != tts.target_frames(60, voice.transcript, text, duration_scale=scale)[0]
        results, _ = tts.synthesize_many([text], voice, candidates=2, steps=1, guidance=1.0, duration_scale=scale,
                                         duration_factors=[1.0, 1.2])
        assert results[0][0]["frames"] == single and results[0][0]["duration"]["duration_rule"] == "duration_head"
        assert results[0][1]["frames"] == round(3.0 * len(text.encode()) * scale * 1.2)
    # A fixed length still overrides the head.
    results, _ = tts.synthesize_many([text], voice, seconds=0.5, steps=1, guidance=1.0)
    assert results[0][0]["frames"] == round(0.5 * 24000 / 512)


def test_auto_duration_picks_the_rule_per_prompt_rate():
    from dacvae_tts.duration import auto_mode

    text = "Normal hızda konuşulmuş bir cümle."  # 34 characters
    assert auto_mode(25 * 34 / 20, text) == "clamp"  # 20 chars/s
    assert auto_mode(25 * 34 / 15, text) == "rule"
    assert auto_mode(25 * 34 / 10, text) == "predictor"
