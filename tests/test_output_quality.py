"""Latent moment monitoring/matching and composite best-of-N selection (issue 13)."""

import warnings

import numpy as np
import pytest
import torch

from dacvae_tts.quality import (
    CandidateScorer,
    SelectionRule,
    channel_moments,
    judge_overlap,
    latent_moments,
    match_moments,
    parse_select_by,
    rank_candidates,
    warn_judge_overlap,
)


def test_channel_moments_and_summary():
    x = torch.tensor([[-1.0, 0.0], [1.0, 2.0]]).repeat(50, 1)
    mean, std, kurtosis = channel_moments(x)
    torch.testing.assert_close(mean, torch.tensor([0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(std, torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(kurtosis, torch.full((2,), -2.0, dtype=torch.float64))  # two-point distribution
    reference = torch.randn(4000, 8, generator=torch.Generator().manual_seed(1))
    summary = latent_moments(2 * reference, reference)
    assert summary["std_ratio_mean"] == pytest.approx(2) and summary["std_ratio_max"] == pytest.approx(2)
    assert summary["rms_ratio"] == pytest.approx(2)
    assert summary["kurtosis_generated"] == pytest.approx(summary["kurtosis_reference"])
    assert abs(summary["kurtosis_reference"]) < 0.2  # Gaussian
    with pytest.raises(ValueError):
        channel_moments(torch.zeros(1, 4))


def test_moment_matching():
    g = torch.Generator().manual_seed(2)
    reference = torch.randn(300, 6, generator=g) * torch.linspace(0.5, 1.5, 6) + torch.linspace(-1, 1, 6)
    generated = torch.randn(200, 6, generator=g) * 1.6 * torch.linspace(0.5, 1.5, 6) + 0.3
    _, r_std, _ = channel_moments(reference)
    r_mean = reference.double().mean(0)
    matched = match_moments(generated, reference, "std")
    m_mean, m_std, _ = channel_moments(matched)
    torch.testing.assert_close(m_std, r_std, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(m_mean, generated.double().mean(0), rtol=1e-5, atol=1e-5)
    assert matched.dtype == generated.dtype
    full = match_moments(generated, reference, "meanstd")
    torch.testing.assert_close(full.double().mean(0), r_mean, rtol=1e-5, atol=1e-5)
    loud = match_moments(generated * 10, reference, "std")
    torch.testing.assert_close(channel_moments(loud)[1], channel_moments(generated * 10)[1] * 0.5)  # clipped factor
    with pytest.raises(ValueError):
        match_moments(generated, reference, "rms")


def test_select_by_parsing_and_ranking():
    assert parse_select_by("cer,wer") == SelectionRule(("cer", "wer"))
    weighted = parse_select_by("cer:10, DNSMOS:1")
    assert weighted.metrics == ("cer", "dnsmos") and weighted.weights == {"cer": 10.0, "dnsmos": 1.0}
    for bad in ("", "cer,cer", "cer:1,wer", "mos", "cer:inf"):
        with pytest.raises(ValueError):
            parse_select_by(bad)
    scores = [dict(cer=0.1, wer=0.2, dnsmos=3.0), dict(cer=0.0, wer=0.5, dnsmos=2.5),
              dict(cer=0.0, wer=0.1, dnsmos=2.9), dict(cer=0.0, wer=0.1, dnsmos=3.2)]
    assert rank_candidates(scores, parse_select_by("cer,wer")) == [2, 3, 1, 0]  # exact ties keep the first
    assert rank_candidates(scores, parse_select_by("cer,dnsmos"))[0] == 3
    assert rank_candidates(scores, parse_select_by("dnsmos"))[0] == 3
    assert rank_candidates(scores, parse_select_by("cer:10,dnsmos:1"))[0] == 3  # 3.2 - 0 beats 3.0 - 1
    assert rank_candidates(scores, parse_select_by("cer:1,dnsmos:0.01"))[0] == 3
    with pytest.raises(ValueError, match="lack"):
        rank_candidates(scores, parse_select_by("utmos"))


def test_default_rule_matches_the_previous_whisper_selector():
    rng = np.random.default_rng(0)
    rule = parse_select_by("cer,wer")
    for _ in range(200):
        scores = [dict(cer=float(rng.integers(0, 3)) / 10, wer=float(rng.integers(0, 3)) / 10) for _ in range(4)]
        previous = min(range(4), key=lambda k: (scores[k]["cer"], scores[k]["wer"], k))
        assert rank_candidates(scores, rule)[0] == previous


class Transcriber:
    def __init__(self, texts):
        self.texts = texts

    def transcribe(self, audios, sample_rate):
        assert sample_rate == 24000 and len(audios) == len(self.texts)
        return self.texts


def test_candidate_scorer_with_fake_models():
    text = "bir iki üç"
    audios = [np.full(24000, value, dtype=np.float32) for value in (0.1, 0.2, 1.0)]
    transcriber = Transcriber(["bir iki", "bir iki üç", "bir iki üç"])
    dnsmos = lambda audio: dict(dnsmos_ovrl=float(np.median(audio)) * 10, dnsmos_sig=1.0, dnsmos_bak=2.0)  # noqa: E731
    best, scores = CandidateScorer("cer,wer", transcriber=transcriber).select(text, audios, 24000)
    assert best == 1 and scores[0]["cer"] > 0 and scores[2]["hypothesis"] == "bir iki üç"
    best, scores = CandidateScorer("cer,dnsmos", transcriber=transcriber, dnsmos=dnsmos).select(text, audios, 24000)
    assert best == 2 and scores[2]["dnsmos"] == pytest.approx(10, rel=1e-3)
    best, scores = CandidateScorer("cer,clip", transcriber=transcriber).select(text, audios, 24000)
    assert best == 1 and scores[2]["clip"] == 1.0
    speaker = lambda audio: torch.tensor([1.0, 0.0]) if np.median(audio) > 0.5 else torch.tensor([0.0, 1.0])  # noqa: E731
    scorer = CandidateScorer("sim", speaker=speaker)
    assert scorer.select(text, audios, 24000, reference=np.full(24000, 0.9, np.float32))[0] == 2
    with pytest.raises(ValueError, match="reference"):
        scorer.select(text, audios, 24000)
    with pytest.raises(ValueError, match="needs models"):
        CandidateScorer("cer,dnsmos", transcriber=transcriber)


def test_judge_overlap_flags_same_family_selection():
    rule = parse_select_by("cer,sim,dnsmos")
    selection = {"asr": "openai/whisper-large-v3-turbo", "speaker": "microsoft/unispeech-sat-base-plus-sv",
                 "dnsmos": "models/sig_bak_ovr.onnx"}
    judge = {"asr": "whisper-large-v3", "speaker": "microsoft/wavlm-base-plus-sv", "dnsmos": "models/sig_bak_ovr.onnx"}
    assert set(judge_overlap(rule, selection, judge)) == {"cer", "dnsmos"}
    same_speaker = {**selection, "speaker": "microsoft/wavlm-base-sv"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        overlap = warn_judge_overlap(rule, same_speaker, judge)
    assert "sim" in overlap and any("oracle" in str(w.message) for w in caught)
