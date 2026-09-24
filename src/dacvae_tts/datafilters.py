"""Pure, testable filters for growing the Turkish corpus from open sources (YODAS, Common Voice, ISSAI TSC).

The per-source preparers in `scripts/data/` measure every clip with these functions, drop the failures and record
the measurements as manifest columns, so a threshold can be tightened later without decoding the audio again.

Evidence behind the choices (see issue #15):
- How much to drop: Raon-OpenTTS (arXiv 2605.20830) dropping the worst 15% by DNSMOS/VAD/WER improves Seed-TTS
  WER 2.19 -> 2.00 and SIM 0.661 -> 0.672, while dropping 50% hurts (WER 2.32). Filters here therefore default
  to outlier cuts, and the preparers warn when the manifest stage alone removes more than ~30%.
- Bandwidth: a latent TTS reproduces the bandwidth of its training audio (Spectral Codecs, arXiv 2406.05298);
  HiFiTTS-2 filters on the -50 dB spectral rolloff. Our podcast MP3s are 12-16 kHz, YODAS is 16 kHz sampled
  (<= 8 kHz) and YODAS2/Sidon 24 kHz (<= 12 kHz), so the bandwidth is measured per clip, not assumed per source.
- Speech ratio: Silero VAD (MIT) speech fraction below 0.8 marks music beds, long pauses or mis-segmentation.
- Evaluation hygiene: the 495 Freya-TR-Eval sentences (its short-native part comes from Common Voice 17 /
  CoVoST2 texts) must never enter training, so transcripts are matched after the Turkish metric normalization.
"""

import csv
import importlib
import json
import math
import threading
import unicodedata
from pathlib import Path

import numpy as np

from .turkish import metric_text_turkish

# Sources that are evaluation-only for this project or lack a usable license. Matching is by substring of the
# repository id / local path, lower-cased, so "google/fleurs" and "/data/fleurs_tr" are both refused.
EVAL_ONLY_SOURCES = {
    "fleurs": "FLEURS-tr is an evaluation set (Sidon's Turkish DNSMOS/CER numbers are measured on it)",
    "mediaspeech": "MediaSpeech-tr is an evaluation set",
    "antalia": "Antalia is an evaluation set",
    "freya": "Freya-TR-Eval is the main benchmark",
}
UNLICENSED_SOURCES = {"kiraat": "KIRAAT is published without a license"}


def check_training_source(*names):
    """Raise if any repository id / path names an evaluation-only or unlicensed corpus."""
    for name in names:
        if not name:
            continue
        lowered = str(name).lower()
        for table in (EVAL_ONLY_SOURCES, UNLICENSED_SOURCES):
            for key, reason in table.items():
                if key in lowered:
                    raise ValueError(f"{name}: {reason}; it must not enter a training manifest")


def load_sentences(path):
    """Evaluation sentences from .txt (one per line), .jsonl/.json (`text` or `sentence`), or .csv/.tsv."""
    path = Path(path)
    suffix = path.suffix.lower()

    def pick(row):
        if isinstance(row, str):
            return row
        for key in ("text", "sentence", "transcript", "transcription"):
            if row.get(key):
                return row[key]
        raise ValueError(f"{path}: row without a text/sentence field: {row}")

    if suffix == ".jsonl":
        rows = [pick(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = [pick(row) for row in (data if isinstance(data, list) else data.get("sentences", []))]
    elif suffix in {".csv", ".tsv"}:
        with open(path, encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t" if suffix == ".tsv" else ",")
            rows = [pick(row) for row in reader]
    else:
        rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [row.strip() for row in rows if row and row.strip()]
    if not rows:
        raise ValueError(f"{path}: no sentences")
    return rows


# Spelling variants that must not hide a duplicate: ASCII "I" typed for "İ" lower-cases to "ı", and the circumflex
# is optional in modern spelling (kâğıt / kağıt). Folding them can only merge sentences that differ in these letters.
MATCH_FOLD = str.maketrans({"ı": "i", "â": "a", "î": "i", "û": "u"})


def match_key(text):
    """Turkish metric text (numbers spelled out, Turkish casing, no punctuation/apostrophes) with ı/i and â/î/û
    folded: the form evaluation sentences are compared in.

    Combining marks that NFKC cannot attach are dropped first: Python's str.lower() turns "İstanbul" into
    "i̇stanbul" (i + U+0307 combining dot above), which the frozen turkish-v1 metric text splits into "i stanbul".
    """
    text = "".join(c for c in unicodedata.normalize("NFKC", text) if unicodedata.category(c) != "Mn")
    return metric_text_turkish(text).translate(MATCH_FOLD)


def _ngrams(words, n):
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


class FreyaExclusion:
    """Match transcripts against held-out evaluation sentences after Turkish metric normalization.

    `match_key` spells numbers out, applies Turkish casing (I -> ı, İ -> i), drops apostrophes and punctuation and
    folds ı/i and the circumflex, so "İSTANBUL'DA 3 GÜN.", "Istanbul'da 3 gün" and "istanbulda üç gün" are the same
    sentence. Three tests, cheapest first:
    - `freya_exact`: identical normalized text (the Common Voice case: the same prompt read by another speaker);
    - `freya_contains`: a long transcript contains a whole evaluation sentence of >= `min_contained_words` words
      (long YouTube caption segments);
    - `freya_near` (optional, `near_threshold`): word n-gram overlap coefficient |A∩B| / min(|A|, |B|) at or above
      the threshold, which also catches a fragment of an evaluation sentence or a one-word edit.
    """

    def __init__(self, sentences, near_threshold=None, ngram=3, min_contained_words=5):
        if near_threshold is not None and not 0 < near_threshold <= 1:
            raise ValueError("near_threshold must be in (0, 1]")
        self.exact = {match_key(s) for s in sentences} - {""}
        self.contained = sorted(s for s in self.exact if len(s.split()) >= min_contained_words)
        self.near_threshold, self.ngram = near_threshold, ngram
        self.grams, self.index = [], {}
        if near_threshold is not None:
            for i, sentence in enumerate(sorted(self.exact)):
                grams = _ngrams(sentence.split(), ngram)
                self.grams.append(grams)
                for gram in grams:
                    self.index.setdefault(gram, set()).add(i)

    def __len__(self):
        return len(self.exact)

    def match(self, text):
        """The reason a transcript must be excluded, or None."""
        normalized = match_key(text)
        if not normalized:
            return None
        if normalized in self.exact:
            return "freya_exact"
        padded = f" {normalized} "
        if any(f" {sentence} " in padded for sentence in self.contained):
            return "freya_contains"
        if self.near_threshold is not None:
            grams = _ngrams(normalized.split(), self.ngram)
            candidates = set().union(*(self.index.get(gram, ()) for gram in grams)) if grams else set()
            for i in candidates:
                smaller = min(len(grams), len(self.grams[i]))
                # One shared n-gram between two tiny texts is not evidence of a duplicate.
                if smaller >= 2 and len(grams & self.grams[i]) / smaller >= self.near_threshold:
                    return "freya_near"
        return None


def text_characters(text):
    """Spoken characters: letters/digits of the Turkish metric text, numbers already spelled out, no spaces."""
    return len(metric_text_turkish(text).replace(" ", ""))


def chars_per_second(text, duration):
    if duration <= 0:
        raise ValueError("Duration must be positive")
    return text_characters(text) / duration


def iqr_bounds(values, k=1.5):
    """Tukey fences (Q1 - k*IQR, Q3 + k*IQR). Speaking-rate outliers are mostly misaligned segments: a transcript
    for a longer stretch than the audio (too fast) or leading/trailing untranscribed speech (too slow)."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return -math.inf, math.inf
    q1, q3 = np.percentile(values, [25, 75])
    return float(q1 - k * (q3 - q1)), float(q3 + k * (q3 - q1))


def iqr_outliers(values, groups=None, k=1.5):
    """Boolean mask of values outside the per-group Tukey fences (one group when `groups` is None)."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.zeros(len(values), dtype=np.int64) if groups is None else np.asarray(groups)
    mask = np.zeros(len(values), dtype=bool)
    for group in np.unique(groups):
        selected = groups == group
        low, high = iqr_bounds(values[selected], k)
        mask[selected] = (values[selected] < low) | (values[selected] > high)
    return mask


def duration_ok(duration, min_seconds=1.0, max_seconds=20.0):
    return duration is not None and min_seconds <= duration <= max_seconds


def bandwidth_hz(audio, sample_rate, threshold_db=50.0, segment_seconds=0.04, smooth_bins=5, floor_hz=50.0):
    """Effective bandwidth: the highest frequency whose average power is within `threshold_db` of the peak.

    Welch-averaged power spectrum (Hann, 50% overlap, ~40 ms segments, i.e. ~23 Hz bins at 48 kHz), a short
    moving average over bins to steady the estimate, peak searched above `floor_hz` so DC/rumble cannot set the
    reference. This is the -50 dB rolloff HiFiTTS-2 filters on: an MP3 encoded with a 16 kHz lowpass reads
    ~16 kHz, a 16 kHz-sampled YODAS clip at most 8 kHz. Clean full-band speech with very little high-frequency
    energy can read lower than its sampling rate allows; that is the bandwidth a model trained on it learns.
    """
    from scipy.signal import welch

    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if audio.size < 16 or not np.any(audio):
        return 0.0
    nperseg = min(audio.size, max(256, 2 ** math.ceil(math.log2(segment_seconds * sample_rate))))
    freqs, power = welch(audio, fs=sample_rate, window="hann", nperseg=nperseg, noverlap=nperseg // 2)
    levels = 10 * np.log10(power + 1e-30)
    if smooth_bins > 1:
        padded = np.pad(levels, smooth_bins // 2, mode="edge")
        levels = np.convolve(padded, np.ones(smooth_bins) / smooth_bins, mode="valid")[: len(freqs)]
    usable = freqs >= floor_hz
    if not usable.any():
        return 0.0
    peak = levels[usable].max()
    above = np.nonzero(usable & (levels >= peak - threshold_db))[0]
    return float(freqs[above[-1]])


def speech_ratio_from_segments(segments, num_samples):
    """Fraction of samples covered by the union of speech segments ({start, end} dicts or pairs, in samples)."""
    if num_samples <= 0:
        return 0.0
    spans = sorted(
        (max(0, int(s["start"] if isinstance(s, dict) else s[0])),
         min(num_samples, int(s["end"] if isinstance(s, dict) else s[1])))
        for s in segments
    )
    covered, current_start, current_end = 0, None, None
    for start, end in spans:
        if end <= start:
            continue
        if current_end is None or start > current_end:
            if current_end is not None:
                covered += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        covered += current_end - current_start
    return covered / num_samples


def resample(audio, sample_rate, target_rate):
    if sample_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    from scipy.signal import resample_poly

    factor = math.gcd(sample_rate, target_rate)
    return resample_poly(audio, target_rate // factor, sample_rate // factor).astype(np.float32)


class SileroVAD:
    """Speech ratio from Silero VAD (MIT, github.com/snakers4/silero-vad), loaded lazily on first use.

    Uses the `silero-vad` pip package when installed, otherwise `torch.hub` (downloads the model once). The
    recurrent model state is per instance, so each thread gets its own copy.
    """

    def __init__(self, threshold=0.5, min_silence_ms=100, speech_pad_ms=30):
        self.options = dict(threshold=threshold, min_silence_duration_ms=min_silence_ms, speech_pad_ms=speech_pad_ms)
        self._local = threading.local()

    def _load(self):
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad

            model = load_silero_vad()
        except ImportError:
            import torch

            model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
            get_speech_timestamps = utils[0]
        self._local.model, self._local.timestamps = model, get_speech_timestamps

    def __call__(self, audio, sample_rate):
        import torch

        if getattr(self._local, "model", None) is None:
            self._load()
        audio = resample(audio, sample_rate, 16000)
        segments = self._local.timestamps(
            torch.from_numpy(np.ascontiguousarray(audio)), self._local.model, sampling_rate=16000, **self.options
        )
        return speech_ratio_from_segments(segments, len(audio))


def load_callable(spec):
    """`package.module:function` -> the function (restoration hooks are named on the command line)."""
    module, _, name = str(spec).partition(":")
    if not module or not name:
        raise ValueError(f"Expected MODULE:FUNCTION, got {spec!r}")
    function = getattr(importlib.import_module(module), name)
    if not callable(function):
        raise ValueError(f"{spec} is not callable")
    return function
