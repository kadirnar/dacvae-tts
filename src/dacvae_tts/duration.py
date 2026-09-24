"""Target-length rules and a small learned duration predictor for the rule-duration (joined-text) models.

The nano/Turkish models have no duration head: the number of target frames comes from the voice prompt. Ways to
turn (prompt frames, prompt transcript, target text) into target frames:

- `rule`: frames per UTF-8 byte of the prompt times target bytes (F5-TTS; what the models were evaluated with).
- `clamp`: the rule, but prompts faster than `threshold` normalized characters per second are slowed towards
  `target` (at most by `max_scale`). Fast podcast prompts otherwise produce rushed, less intelligible speech; on
  Freya-TR-Eval the fast prompts' WER fell from 7.5% to 3.1% at x1.15 while slowing normal/slow prompts hurt.
- `predictor`: log-linear regression fitted on same-speaker (prompt, target) pairs of the corpus
  (`scripts/train_duration.py`), using syllables (Turkish vowels), words and punctuation of the target and the
  prompt's speaking rate. It learns how strongly a prompt's rate carries over to a new sentence.
- `auto`: the rule where it works and a correction where it does not. On Freya-TR-Eval the rule is best for prompts
  of 13-17 characters/s; slow prompts (long pauses) get the predictor and fast prompts the clamp.
- `articulation`: the prompt's articulation rate, syllables per second of net speaking time measured on its
  waveform (`audio.speech_timing`: edge silence cut, pauses >= 200 ms excluded), times the target's syllables, plus
  a pause budget for the target's internal punctuation and a floor for very short texts. Frames per byte count the
  prompt's pauses and edge silence as speech, so a pausy prompt gives every target, short ones most visibly, extra
  frames that the model fills with lengthening or filler words; `auto`'s 13/17 chars/s switch only patches this.
  Syllables are the unit because syllable-level rates transfer best across texts and languages (Cross-Lingual
  F5-TTS 2, arXiv 2609.15184), and Turkish orthography is ~95% transparent, so vowels count syllables. F5-TTS does
  the same clean-ups heuristically: it trims the prompt's edge silence (-42 dBFS), adds 50 ms, ends the prompt text
  with ". " and slows texts of <= 10 bytes.

A duration model is judged by the speech it produces, not by its duration error: in DMOSpeech 2 (arXiv 2507.14988)
ground-truth durations give 1.821 WER, the F5 rate rule 2.028, a supervised predictor 3.750 (worse than the rule) and
the same predictor optimized for WER/SIM 1.752. scripts/fit_duration_best_factor.py refits the predictor on the
duration factor that decodes best instead of on corpus durations.
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

VOWELS = set("aeıioöuüâîûAEIİOÖUÜÂÎÛ")
FRAME_RATE = 25.0  # DACVAE latent frames per second
AUTO_SLOW, AUTO_FAST = 13.0, 17.0  # prompt speaking-rate bounds of `auto` (normalized characters per second)


def units(text):
    """Counts of a normalized transcript: bytes, characters, syllables (vowels), words, pauses."""
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    return {
        "bytes": len(text.encode("utf-8")),
        "chars": len(text),
        "syllables": sum(c in VOWELS for c in text),
        "words": len(words),
        "commas": sum(text.count(c) for c in ",;:-"),
        "stops": len(re.findall(r"[.!?]+", text)),
    }


def rule_frames(reference_frames, reference_text, text, unit="bytes"):
    ref, tgt = units(reference_text), units(text)
    return reference_frames / max(ref[unit], 1) * max(tgt[unit], 1)


def speaking_rate(reference_frames, reference_text):
    """Normalized characters per second of the voice prompt."""
    return len(reference_text) / max(reference_frames / FRAME_RATE, 1e-3)


def clamp_scale(reference_frames, reference_text, threshold=17.0, target=16.0, max_scale=1.2):
    """Duration factor >= 1 that slows prompts faster than `threshold` chars/s towards `target` chars/s."""
    rate = speaking_rate(reference_frames, reference_text)
    if rate <= threshold:
        return 1.0
    return min(rate / target, max_scale)


def auto_mode(reference_frames, reference_text):
    """Which rule `auto` applies to this prompt: predictor (slow), rule (normal) or clamp (fast)."""
    rate = speaking_rate(reference_frames, reference_text)
    return "predictor" if rate < AUTO_SLOW else ("clamp" if rate > AUTO_FAST else "rule")


def pause_budget(text, comma_pause=0.15, stop_pause=0.3):
    """Seconds of pause the target's internal punctuation asks for: `comma_pause` per , ; : or spaced dash and
    `stop_pause` per sentence-final . ! ? (or ellipsis) inside the text. Punctuation at the end adds nothing: outputs
    are trimmed, and chunked long texts get their pauses when joined. Returns (seconds, commas, stops)."""
    body = re.sub(r"[\s.!?,;:…\"'»”’)\]-]+$", "", text)
    commas = len(re.findall(r"[,;:]|\s-\s", body))
    stops = len(re.findall(r"[.!?…]+", body))
    return commas * comma_pause + stops * stop_pause, commas, stops


def articulation_seconds(timing, reference_text, text, min_rate=3.0, max_rate=8.5, comma_pause=0.15, stop_pause=0.3,
                         min_seconds=0.5):
    """Target seconds of the `articulation` rule and its profile: syllables / rate + pause budget, >= `min_seconds`.

    `timing` is `audio.speech_timing` of the prompt waveform. The rate (prompt syllables per second of speaking time)
    is clamped to [min_rate, max_rate] syllables/s, roughly 9-26 chars/s of Turkish, against a failed pause
    detection; the cap also keeps >= 0.12 s per syllable. Pause defaults are conservative (short comma breaths, no
    trailing pause), and the floor is for one- or two-word texts whose syllable time would be ~0.3 s (F5-TTS slows
    texts of <= 10 bytes, FreyaTTS has a floor for short inputs). Returns (None, profile) when the prompt gives no
    rate (no waveform, no speech detected, no vowel in its transcript); the caller then falls back to the byte rule.
    """
    syllables = units(reference_text)["syllables"]
    speech = float(timing.get("speech_seconds", 0.0)) if timing else 0.0
    profile = {"duration_prompt_timing": dict(timing) if timing else None, "duration_prompt_syllables": syllables}
    if speech <= 0 or syllables < 1:
        reason = "no prompt waveform" if not timing else ("no speech detected" if speech <= 0 else "no prompt vowels")
        return None, {**profile, "duration_articulation_fallback": f"rule ({reason})"}
    measured = syllables / speech
    rate = min(max(measured, min_rate), max_rate)
    pauses, commas, stops = pause_budget(text, comma_pause, stop_pause)
    target = max(units(text)["syllables"], 1)
    seconds = target / rate + pauses
    profile.update(
        duration_articulation_rate=rate,
        duration_articulation_measured_rate=measured,
        duration_target_syllables=target,
        duration_pause_budget_seconds=pauses,
        duration_pause_commas=commas,
        duration_pause_stops=stops,
        duration_floor_applied=seconds < min_seconds,
    )
    return max(seconds, min_seconds), profile


def features(reference_frames, reference_text, text):
    ref, tgt = units(reference_text), units(text)
    log_rate_syllable = math.log(reference_frames / max(ref["syllables"], 1))
    log_rate_byte = math.log(reference_frames / max(ref["bytes"], 1))
    return [
        1.0,
        math.log(max(tgt["syllables"], 1)),
        math.log(max(tgt["words"], 1)),
        math.log(max(tgt["bytes"], 1)),
        tgt["commas"] / max(tgt["words"], 1),
        tgt["stops"] / max(tgt["words"], 1),
        log_rate_syllable,
        log_rate_byte,
        math.log(max(reference_frames, 1)),
    ]


FEATURES = ["bias", "log_syllables", "log_words", "log_bytes", "commas_per_word", "stops_per_word",
            "log_prompt_frames_per_syllable", "log_prompt_frames_per_byte", "log_prompt_frames"]


@dataclass
class DurationPredictor:
    """log(target frames) = w . features; fitted by ridge regression on corpus pairs."""

    weights: list
    metadata: dict

    @classmethod
    def load(cls, path=None):
        path = Path(path) if path else Path(__file__).with_name("duration_tr.json")
        obj = json.loads(path.read_text())
        if obj.get("features") != FEATURES:
            raise ValueError("Duration predictor was fitted with a different feature set")
        return cls(obj["weights"], obj.get("metadata", {}))

    def predict(self, reference_frames, reference_text, text):
        values = features(reference_frames, reference_text, text)
        return math.exp(sum(w * v for w, v in zip(self.weights, values)))

    def save(self, path):
        Path(path).write_text(json.dumps({"features": FEATURES, "weights": self.weights, "metadata": self.metadata},
                                         indent=2, ensure_ascii=False))


def fit(rows, ridge=1e-3):
    """rows: iterable of (reference_frames, reference_text, text, target_frames). Returns DurationPredictor."""
    import numpy as np

    x, y = [], []
    for reference_frames, reference_text, text, target_frames in rows:
        x.append(features(reference_frames, reference_text, text))
        y.append(math.log(target_frames))
    x, y = np.asarray(x), np.asarray(y)
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0  # no shrinkage of the intercept
    weights = np.linalg.solve(x.T @ x + penalty * len(x), x.T @ y)
    return DurationPredictor([float(w) for w in weights], {"pairs": len(x), "ridge": ridge})
