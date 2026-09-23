"""Target-length rules and a small learned duration predictor for the rule-duration (joined-text) models.

The nano/Turkish models have no duration head: the number of target frames comes from the voice prompt. Three
ways to turn (prompt frames, prompt transcript, target text) into target frames are provided:

- `rule`: frames per UTF-8 byte of the prompt times target bytes (F5-TTS; what the models were evaluated with).
- `clamp`: the rule, but prompts faster than `threshold` normalized characters per second are slowed towards
  `target` (at most by `max_scale`). Fast podcast prompts otherwise produce rushed, less intelligible speech; on
  Freya-TR-Eval the fast prompts' WER fell from 7.5% to 3.1% at x1.15 while slowing normal/slow prompts hurt.
- `predictor`: log-linear regression fitted on same-speaker (prompt, target) pairs of the corpus
  (`scripts/train_duration.py`), using syllables (Turkish vowels), words and punctuation of the target and the
  prompt's speaking rate. It learns how strongly a prompt's rate carries over to a new sentence.
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

VOWELS = set("aeıioöuüâîûAEIİOÖUÜÂÎÛ")
FRAME_RATE = 25.0  # DACVAE latent frames per second


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
