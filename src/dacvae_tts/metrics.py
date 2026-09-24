"""Optional, frozen evaluators. Missing models produce errors, never proxy scores."""

import importlib.metadata
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .codec import file_digest, read_audio
from .data import jsonl
from .eval_protocol import WHISPER_V1

METRIC_NORMALIZATIONS = ("english-unicode-v2", "legacy-ascii-v1", "turkish-v1", "turkish-v2")


def default_metric_normalization(language):
    """Turkish needs its own case folding (İ/ı) and number spelling; other languages keep the old default."""
    return "turkish-v1" if language == "tr" else "english-unicode-v2"


def metric_text(text, version="english-unicode-v2"):
    if version == "turkish-v1":
        from .turkish import metric_text_turkish

        return metric_text_turkish(text)
    if version == "turkish-v2":  # opt-in; turkish-v1 stays the default for Turkish so scores remain comparable
        from .turkish import metric_text_turkish_v2

        return metric_text_turkish_v2(text)
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    if version == "legacy-ascii-v1":
        text = re.sub(r"[^a-z0-9'\s]", " ", text)
    elif version == "english-unicode-v2":
        text = "".join(
            c if c.isalnum() or c.isspace() or c == "'" or unicodedata.category(c) == "Mn" else " "
            for c in text
        )
    else:
        raise ValueError("Unsupported metric normalization version")
    return " ".join(text.split())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def word_edit_counts(reference, hypothesis):
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            d, s, de, ins = previous[j - 1]
            diagonal = (d + (ref != hyp), s + (ref != hyp), de, ins)
            d, s, de, ins = previous[j]
            deletion = (d + 1, s, de + 1, ins)
            d, s, de, ins = current[-1]
            insertion = (d + 1, s, de, ins + 1)
            current.append(min((diagonal, deletion, insertion), key=lambda item: item[0]))
        previous = current
    return previous[-1]


def error_counts(reference, hypothesis, normalization="english-unicode-v2"):
    ref, hyp = metric_text(reference, normalization), metric_text(hypothesis, normalization)
    if not ref:
        raise ValueError("Reference transcript is empty after metric normalization")
    rw, hw = ref.split(), hyp.split()
    rc, hc = ref.replace(" ", ""), hyp.replace(" ", "")
    we, substitutions, deletions, insertions = word_edit_counts(rw, hw)
    ce = edit_distance(rc, hc)
    return {
        "word_edits": we,
        "words": len(rw),
        "char_edits": ce,
        "chars": len(rc),
        "wer": we / len(rw),
        "cer": ce / len(rc),
        "word_substitutions": substitutions,
        "word_deletions": deletions,
        "word_insertions": insertions,
    }


class DNSMOS:
    """Official non-personalized SIG/BAK/OVRL ONNX model, 9.01s windows / 1s hop.

    Calibration follows Microsoft's DNS-Challenge/DNSMOS/dnsmos_local.py.
    P.808 is a different model/metric and is deliberately not reported as OVRL.
    """

    def __init__(self, model_path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        # One thread per session: DNSMOS is run from many worker processes at once; per-process thread pools
        # sized to every core oversubscribe the CPU and starve concurrent training data loaders.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if not audio.size or not np.isfinite(audio).all():
            raise ValueError("Cannot score empty/nonfinite audio")
        size, hop = 144160, 16000
        while len(audio) < size:
            audio = np.tile(audio, 2)
        outputs = []
        calibration = [
            (-0.08397278, 1.22083953, 0.0052439),
            (-0.13166888, 1.60915514, -0.39604546),
            (-0.06766283, 1.11546468, 0.04602535),
        ]
        # Preserve the official implementation's integer-second hop-count convention.
        num_hops = int(np.floor(len(audio) / hop) - 9.01) + 1
        for index in range(num_hops):
            start = index * hop
            raw = np.asarray(
                self.session.run(None, {self.input_name: audio[None, start : start + size]})[0]
            ).reshape(-1)
            if raw.size != 3:
                raise ValueError("Expected the standard sig_bak_ovr.onnx DNSMOS model")
            outputs.append([np.polyval(coeff, value) for coeff, value in zip(calibration, raw)])
        scores = np.mean(outputs, axis=0)
        return dict(zip(("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"), map(float, scores)))


# What changes a row's WER/CER/SIM for the same audio: kept in every result row (`row_identity`) so that
# comparison.compare_evaluations can refuse to pair runs scored differently; the full identity (library versions,
# model revisions, hashes) goes to the summary.
ROW_IDENTITY_KEYS = (
    "asr_backend", "asr_model", "language", "metric_normalization", "decoding", "compute_type", "device",
    "speaker_model",
)


def row_identity(identity):
    """Compact scorer identity of an Evaluator identity (or of a compact one: the projection is idempotent).

    Protocol v2 contributes its options (`protocol_options`, e.g. band_limit_8k, sim_o); None/missing values stay
    None, so rows written before these fields existed compare as "unknown", not as equal.
    """
    if not isinstance(identity, dict):
        return None
    protocol = identity.get("protocol")
    options = protocol.get("options") if isinstance(protocol, dict) else identity.get("protocol_options")
    return {**{key: identity.get(key) for key in ROW_IDENTITY_KEYS}, "protocol_options": options}


class Evaluator:
    def __init__(
        self,
        asr_model="large-v3",
        dnsmos_model=None,
        speaker_model="microsoft/wavlm-base-plus-sv",
        device="cpu",
        metric_normalization=None,
        language="en",
        protocol=None,
    ):
        """`protocol` (eval_protocol.ProtocolOptions, default None = v1 scoring, unchanged) enables the opt-in
        protocol-v2 extras; see dacvae_tts.eval_protocol."""
        self.language = language
        if metric_normalization is None:
            metric_normalization = default_metric_normalization(language)
        from faster_whisper import WhisperModel

        compute_type = "float16" if device == "cuda" else "int8"
        self.asr = WhisperModel(asr_model, device=device, compute_type=compute_type)
        self.dnsmos = DNSMOS(dnsmos_model) if dnsmos_model else None
        self.device = torch.device(device)
        self.speaker_name = speaker_model
        self.asr_name = asr_model
        self.metric_normalization = metric_normalization
        self.dnsmos_path = str(dnsmos_model) if dnsmos_model else None
        self.extractor = self.speaker = None
        if speaker_model:
            from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

            self.extractor = AutoFeatureExtractor.from_pretrained(speaker_model)
            self.speaker = AutoModelForAudioXVector.from_pretrained(speaker_model).to(device).eval()
        self.identity = {
            "asr_model": asr_model,
            "faster_whisper_version": importlib.metadata.version("faster-whisper"),
            "speaker_model": speaker_model,
            "speaker_revision": getattr(getattr(self.speaker, "config", None), "_commit_hash", None),
            "dnsmos_sha256": file_digest(dnsmos_model) if dnsmos_model else None,
            "metric_normalization": metric_normalization,
            "language": language,
            # int8 on the CPU and float16 on CUDA transcribe (slightly) differently.
            "asr_backend": "faster-whisper",
            "device": str(device),
            "compute_type": compute_type,
        }
        self.protocol = None
        if protocol is not None and protocol.enabled:
            from .eval_protocol import ProtocolScorer, whisper_snapshot

            self.protocol = ProtocolScorer(protocol, device, self.dnsmos)
            self.identity["protocol"] = {**self.protocol.identity, "asr_snapshot": whisper_snapshot(asr_model)}
        self.identity["decoding"] = self.protocol.whisper_kwargs() if self.protocol else dict(WHISPER_V1)
        self.row_identity = row_identity(self.identity)

    @torch.inference_mode()
    def embedding(self, audio):
        inputs = self.extractor(audio.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        return F.normalize(self.speaker(**inputs).embeddings.float(), dim=-1)

    def score(self, audio_path, text, reference_path=None, original_prompt=None, codec_prompt=None):
        """`reference_path` feeds the v1 `speaker_similarity`; `original_prompt`/`codec_prompt` (the prompt's
        original recording / its codec resynthesis) are used only by an enabled protocol (sim_o / sim_r)."""
        audio = read_audio(audio_path, 16000)
        protocol = getattr(self, "protocol", None)  # instances built with Evaluator.__new__ have none
        if protocol is None:
            asr_audio, asr_info = audio, {}
            decoding = dict(WHISPER_V1)
        else:
            asr_audio, asr_info = protocol.asr_audio(audio)
            decoding = protocol.whisper_kwargs()
        segments, _ = self.asr.transcribe(asr_audio.numpy(), language=self.language, **decoding)
        hypothesis = " ".join(segment.text for segment in segments)
        result = {
            **error_counts(text, hypothesis, self.metric_normalization),
            "hypothesis": hypothesis,
            "audio_seconds": len(audio) / 16000,
            "clipped_fraction": float((audio.abs() >= 0.999).float().mean()),
            "evaluator": self.identity,
        }
        if self.dnsmos:
            result.update(self.dnsmos(audio.numpy()))
        if self.speaker is not None and reference_path:
            ref = read_audio(reference_path, 16000)
            result["speaker_similarity"] = float((self.embedding(audio) * self.embedding(ref)).sum())
        if protocol is not None:
            result.update(asr_info)
            result.update(protocol.score(audio_path, audio, text, hypothesis, self.metric_normalization,
                                         original_prompt, codec_prompt))
        return result


def summarize(rows):
    if not rows:
        raise ValueError("No successful evaluation rows")
    result = {"count": len(rows)}
    for prefix, denominator in (("word", "words"), ("char", "chars")):
        result["wer" if prefix == "word" else "cer"] = sum(r[f"{prefix}_edits"] for r in rows) / sum(
            r[denominator] for r in rows
        )
    for key in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "speaker_similarity", "rtf"):
        if all(key in row for row in rows):
            result[key] = float(np.mean([row[key] for row in rows]))
    from .eval_protocol import summary_extras  # additive keys only: per-utterance means, S/D/I, v2 metrics

    result.update(summary_extras(rows))
    return result


def evaluate(args):
    from .eval_protocol import manifest_prompts, protocol_from_args

    evaluator = Evaluator(
        args.asr_model,
        args.dnsmos_model,
        None if args.no_speaker else args.speaker_model,
        args.device,
        getattr(args, "metric_normalization", None),
        getattr(args, "language", "en"),
        protocol=protocol_from_args(args),
    )
    reference_kind = getattr(args, "reference_kind", None)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError("Evaluation output exists; use a new path")
    rows, failed = [], 0
    with open(out, "w") as stream:
        for row in jsonl(args.manifest):
            try:
                prompts = manifest_prompts(row, reference_kind)
                result = {**row, **evaluator.score(row["audio"], row["text"], row.get("reference_audio"), **prompts)}
                rows.append(result)
            except (ValueError, OSError) as exc:
                result = {**row, "error": str(exc)}
                failed += 1
            stream.write(json.dumps(result) + "\n")
    summary = summarize(rows)
    summary.update(
        {
            "failed": failed,
            "failure_rate": failed / (len(rows) + failed),
            "asr_model": args.asr_model,
            "speaker_model": evaluator.speaker_name,
            "dnsmos_model": evaluator.dnsmos_path,
            "normalization": evaluator.metric_normalization,
            "evaluator": evaluator.identity,
        }
    )
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
