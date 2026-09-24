"""Output-quality tools around the sampler: latent moment monitoring/matching and composite best-of-N selection.

Both are opt-in except the moment summary, which is cheap metadata. Neither changes the default outputs.
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch

MOMENT_MATCH = ("std", "meanstd")


def channel_moments(latents):
    """Per-channel mean, standard deviation and excess kurtosis of [frames, C] latents (float64)."""
    if latents.ndim != 2 or latents.size(0) < 2:
        raise ValueError("Latent moments need [frames>=2, C]")
    x = latents.detach().double()
    mean = x.mean(0)
    centered = x - mean
    variance = centered.square().mean(0)
    kurtosis = centered.pow(4).mean(0) / variance.square().clamp_min(1e-24) - 3
    return mean, variance.sqrt(), kurtosis


def latent_moments(generated, reference):
    """Moments of a generated target against its voice prompt, both normalized latents [frames, C].

    High guidance mostly inflates amplitude (the component of the guidance difference parallel to the conditional
    estimate; see `model.guided_update`), which ends as tanh saturation in the decoder. The per-channel spread of
    the target relative to the prompt of the same voice is a codec-free view of that: std ratios well above 1 or
    heavy tails (excess kurtosis above the prompt's) flag over-guided rows before any audio metric is computed.
    """
    g_mean, g_std, g_kurtosis = channel_moments(generated)
    r_mean, r_std, r_kurtosis = channel_moments(reference)
    ratio = g_std / r_std.clamp_min(1e-6)
    rms = [value.detach().double().square().mean().sqrt() for value in (generated, reference)]
    return {
        "std_ratio_mean": float(ratio.mean()),
        "std_ratio_min": float(ratio.min()),
        "std_ratio_max": float(ratio.max()),
        "rms_ratio": float(rms[0] / rms[1].clamp_min(1e-12)),
        "mean_shift": float(((g_mean - r_mean).abs() / r_std.clamp_min(1e-6)).mean()),
        "kurtosis_generated": float(g_kurtosis.mean()),
        "kurtosis_reference": float(r_kurtosis.mean()),
    }


def match_moments(generated, reference, mode="std", limits=(0.5, 2.0)):
    """Rescale each channel of the generated target [frames, C] to the voice prompt's per-channel std.

    "std" scales around the target's own channel means (content-dependent offsets stay); "meanstd" also moves the
    means to the prompt's. The factors are clipped to `limits` so a near-constant channel cannot be blown up. The
    normalization is a per-channel affine map, so matching normalized or raw latents gives the same waveform.
    Hypothesis under test (issue 13): pulling an over-guided target's spread back to the prompt's reduces tanh
    saturation without the alignment cost of weaker guidance.
    """
    if mode not in MOMENT_MATCH:
        raise ValueError(f"moment_match must be one of {MOMENT_MATCH}")
    g_mean, g_std, _ = channel_moments(generated)
    r_mean, r_std, _ = channel_moments(reference)
    factor = (r_std / g_std.clamp_min(1e-6)).clamp(*limits)
    center = r_mean if mode == "meanstd" else g_mean
    return ((generated.double() - g_mean) * factor + center).to(generated.dtype)


# Best-of-N selection. Direction: +1 higher is better, -1 lower is better.
SELECTION_METRICS = {
    "cer": -1, "wer": -1, "dnsmos": 1, "dnsmos_sig": 1, "dnsmos_bak": 1, "utmos": 1, "sim": 1, "clip": -1,
}
METRIC_FAMILY = {"cer": "asr", "wer": "asr", "dnsmos": "dnsmos", "dnsmos_sig": "dnsmos", "dnsmos_bak": "dnsmos",
                 "utmos": "utmos", "sim": "speaker", "clip": None}
MODEL_FAMILIES = ("whisper", "wavlm", "unispeech", "wav2vec2", "hubert", "ecapa", "titanet", "dnsmos", "utmos")


@dataclass(frozen=True)
class SelectionRule:
    """Metrics in priority order; `weights` None ranks lexicographically, else by the weighted sum."""

    metrics: tuple
    weights: dict = None


def parse_select_by(spec):
    """"cer,wer" ranks lexicographically (lowest CER, ties by WER: the existing Whisper selector); "cer:10,dnsmos:1"
    maximizes the weighted sum of direction * value on raw scales (CER is a fraction, DNSMOS/UTMOS are 1-5 MOS, SIM a
    cosine), so the weights also convert units: here 0.1 CER is worth one DNSMOS point."""
    names, weights = [], {}
    for part in (p.strip() for p in str(spec).split(",")):
        if not part:
            continue
        name, _, weight = part.partition(":")
        name = name.strip().lower()
        if name not in SELECTION_METRICS:
            raise ValueError(f"Unknown selection metric {name!r}; choose from {sorted(SELECTION_METRICS)}")
        if name in names:
            raise ValueError(f"Selection metric {name!r} given twice")
        names.append(name)
        if weight:
            weights[name] = float(weight)
            if not math.isfinite(weights[name]):
                raise ValueError("Selection weights must be finite")
    if not names:
        raise ValueError("Empty selection rule")
    if weights and len(weights) != len(names):
        raise ValueError("Give a weight to every selection metric or to none (lexicographic)")
    return SelectionRule(tuple(names), weights or None)


def rank_candidates(scores, rule):
    """Candidate indices, best first; exact ties keep the earlier candidate (lower seed offset)."""
    missing = {m for s in scores for m in rule.metrics if m not in s}
    if missing:
        raise ValueError(f"Candidates lack the selection metrics {sorted(missing)}")

    def key(k):
        values = [-SELECTION_METRICS[m] * float(scores[k][m]) for m in rule.metrics]
        if rule.weights is None:
            return (*values, k)
        return (sum(rule.weights[m] * v for m, v in zip(rule.metrics, values)), k)

    return sorted(range(len(scores)), key=key)


def model_family(name):
    lowered = str(name).lower()
    return next((family for family in MODEL_FAMILIES if family in lowered), lowered)


def judge_overlap(rule, selection_models, judge_models):
    """Selection metrics scored by the evaluation model or its family; returns {metric: (selector, judge)}.

    Selecting and judging with the same model family inflates best-of-N gains 2-3x (arXiv 2607.08256), e.g. the
    Whisper-turbo selector with the Whisper large-v3 judge. Such runs measure an oracle upper bound, not a gain.
    """
    overlap = {}
    for metric in rule.metrics:
        family = METRIC_FAMILY[metric]
        selector, judge = selection_models.get(family), judge_models.get(family)
        if family and selector and judge and model_family(selector) == model_family(judge):
            overlap[metric] = (str(selector), str(judge))
    return overlap


def to_16k(audio, sample_rate):
    from scipy.signal import resample_poly

    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate == 16000:
        return audio
    factor = math.gcd(int(sample_rate), 16000)
    return resample_poly(audio, 16000 // factor, int(sample_rate) // factor).astype(np.float32)


class CandidateScorer:
    """Scores best-of-N candidates on the metrics of a SelectionRule and picks the best.

    Every model is supplied by the caller (and so loaded lazily, only when a rule needs it):
      transcriber: .transcribe(list of waveforms, sample_rate) -> list of str (ASR for cer/wer)
      dnsmos: callable(16 kHz waveform) -> {"dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"} (metrics.DNSMOS)
      utmos: callable(16 kHz waveform) -> MOS (see `load_utmos`)
      speaker: callable(16 kHz waveform) -> L2-normalized embedding; its model must differ from the SIM judge
    `clip` (fraction of samples at |x| >= 0.999, the decoder's tanh ceiling) needs no model.
    """

    def __init__(self, rule, transcriber=None, dnsmos=None, utmos=None, speaker=None, normalization="turkish-v1"):
        self.rule = parse_select_by(rule) if isinstance(rule, str) else rule
        needs = {METRIC_FAMILY[m] for m in self.rule.metrics}
        available = {"asr": transcriber, "dnsmos": dnsmos, "utmos": utmos, "speaker": speaker, None: True}
        absent = sorted(f for f in needs if available[f] is None)
        if absent:
            raise ValueError(f"The selection rule needs models for {absent}")
        self.transcriber, self.dnsmos, self.utmos, self.speaker = transcriber, dnsmos, utmos, speaker
        self.normalization = normalization

    def score(self, text, audios, sample_rate, reference=None):
        from .metrics import error_counts

        needs = {METRIC_FAMILY[m] for m in self.rule.metrics}
        scores = [{} for _ in audios]
        if "asr" in needs:
            for score, hypothesis in zip(scores, self.transcriber.transcribe(list(audios), sample_rate)):
                counts = error_counts(text, hypothesis, self.normalization)
                score.update(cer=counts["cer"], wer=counts["wer"], hypothesis=hypothesis)
        clips = [to_16k(a, sample_rate) for a in audios] if needs & {"dnsmos", "utmos", "speaker"} else None
        if "speaker" in needs:
            if reference is None:
                raise ValueError("The sim selection metric needs the reference waveform")
            anchor = torch.as_tensor(self.speaker(to_16k(reference, sample_rate))).flatten().float()
        for index, (score, audio) in enumerate(zip(scores, audios)):
            if "dnsmos" in needs:
                quality = self.dnsmos(clips[index])
                score.update(dnsmos=quality["dnsmos_ovrl"], dnsmos_sig=quality["dnsmos_sig"],
                             dnsmos_bak=quality["dnsmos_bak"])
            if "utmos" in needs:
                score["utmos"] = float(self.utmos(clips[index]))
            if "speaker" in needs:
                score["sim"] = float(torch.as_tensor(self.speaker(clips[index])).flatten().float() @ anchor)
            if "clip" in self.rule.metrics:
                score["clip"] = float(np.mean(np.abs(np.asarray(audio)) >= 0.999))
        return scores

    def select(self, text, audios, sample_rate, reference=None):
        """(index of the best candidate, per-candidate score dicts)."""
        scores = self.score(text, audios, sample_rate, reference)
        return rank_candidates(scores, self.rule)[0], scores


def load_utmos(kind="utmos22", device="cpu"):
    """Lazily loaded UTMOS predictor: callable(16 kHz waveform) -> MOS.

    utmos22: UTMOS22 strong through torch.hub (tarepan/SpeechMOS:v1.2.0; no package, downloads on first use).
    utmosv2: the UTMOSv2 package (pip install git+https://github.com/sarulab-speech/UTMOSv2.git), slower.
    """
    if kind == "utmos22":
        predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()

        @torch.inference_mode()
        def score(audio):
            return float(predictor(torch.as_tensor(audio, dtype=torch.float32, device=device)[None], 16000)[0])

        return score
    if kind == "utmosv2":
        try:
            import utmosv2
        except ImportError as error:
            raise ImportError("utmosv2 needs: pip install git+https://github.com/sarulab-speech/UTMOSv2.git") from error
        model = utmosv2.create_model(pretrained=True, device=device)
        return lambda audio: float(np.asarray(model.predict(data=np.asarray(audio, np.float32), sr=16000)).reshape(-1)[0])
    raise ValueError("UTMOS kind must be utmos22 or utmosv2")


def warn_judge_overlap(rule, selection_models, judge_models):
    overlap = judge_overlap(rule, selection_models, judge_models)
    for metric, (selector, judge) in overlap.items():
        warnings.warn(
            f"Best-of-N selects on {metric} with {selector}, the family of the evaluation model {judge}: the gain is "
            "optimistically biased (arXiv 2607.08256); report it as an oracle upper bound",
            stacklevel=2,
        )
    return overlap
