"""Models, ZeroGPU jobs and post-processing of the DACVAE-TTS Turkish demo (the UI lives in app.py).

ZeroGPU runs every @spaces.GPU function in a forked worker process; CUDA is only real inside it, while the main
process emulates it. Everything that should be ready without reloading (shared DACVAE codec, the published
model, Whisper, the speaker-verification model) is therefore created at import time with `.to("cuda")`: ZeroGPU packs
those tensors once and moves them onto the GPU when a worker starts. Other checkpoints are downloaded in the main
process and loaded inside the worker, where they stay cached while the worker is reused. Text, audio I/O and
DNSMOS (CPU ONNX) run in the main process so that GPU time only covers encoding, sampling, decoding and ASR.
"""

import json
import math
import os
import re
import tempfile
import threading
import time
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path

try:
    import spaces  # must be imported before torch initializes CUDA
except ImportError:  # local runs without the ZeroGPU package
    spaces = None

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from scipy.signal import resample_poly

from dacvae_tts.audio import finalize, join, loudness, to_pcm16, trim_silence
from dacvae_tts.codec import Codec, read_audio
from dacvae_tts.duration import speaking_rate
from dacvae_tts.frontend import speakable, split_sentences
from dacvae_tts.inference import Synthesizer, VoiceReference
from dacvae_tts.metrics import DNSMOS, error_counts

ROOT = Path(__file__).resolve().parent
ZERO = spaces is not None and os.environ.get("SPACES_ZERO_GPU", "").lower() in {"1", "true"}
DEVICE = "cuda" if (ZERO or torch.cuda.is_available()) else "cpu"
SAMPLE_RATE = 48000
ASR_NAME = os.environ.get("ASR_MODEL", "openai/whisper-large-v3-turbo" if DEVICE == "cuda" else "openai/whisper-small")
SV_NAME = "microsoft/wavlm-base-plus-sv"
DNSMOS_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
MAX_CUSTOM_BYTES = 1_600_000_000
MAX_BATCH_SENTENCES = 30

# Label -> (repo id, file, repo type). All public: downloads never use a token.
CHECKPOINTS = OrderedDict(
    [
        ("w512-clean 60k · published (Freya WER 4.3 %)", ("VoiceHub/dacvae-tts-tr-w512", "model.pt", "model")),
        ("w512-stage2-hq 10k (Freya 4.6 %)", ("VoiceHub/dacvae-tts-tr-w512-stage2-hq", "checkpoints/step-0010000.pt", "dataset")),
        ("stage3-hq 10k · 51M (Freya 6.9 %)", ("VoiceHub/dacvae-tts-tr-stage3-hq", "checkpoints/step-0010000.pt", "dataset")),
        ("stage2-b 20k · 51M (Freya 7.1 %)", ("VoiceHub/dacvae-tts-tr-stage2-b", "checkpoints/step-0020000.pt", "dataset")),
        ("nano-b-ke4 40k · 51M (Freya 9.1 %)", ("VoiceHub/dacvae-tts-tr-nano-b-ke4", "checkpoints/step-0040000.pt", "dataset")),
        ("nano-a 40k · 51M (Freya 10.5 %)", ("VoiceHub/dacvae-tts-tr-nano-a", "checkpoints/step-0040000.pt", "dataset")),
    ]
)
DEFAULT_MODEL = next(iter(CHECKPOINTS))

# Freya-TR-Eval validated defaults (see the About tab): guidance 5, 32 Euler steps, sway -1, duration mode `auto`
# (prompt rate for 13-17 chars/s prompts, predictor for slower, clamp for faster ones).
DEFAULTS = dict(guidance=5.0, steps=32, sway=-1.0, duration_mode="auto", candidates=3)
PAUSES = {"sentence": 0.30, "clause": 0.15}


def gpu(duration):
    """spaces.GPU with a dynamic duration on ZeroGPU; a no-op elsewhere. Errors reach the UI with their message
    (ZeroGPU only forwards gr.Error texts from the worker; any other exception would become "GPU task aborted")."""

    def wrap(function):
        def run(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except gr.Error:
                raise
            except ValueError as error:
                raise gr.Error(str(error)) from error
            except torch.cuda.OutOfMemoryError as error:
                raise gr.Error("Out of GPU memory; shorten the text or lower the number of candidates") from error

        run.__name__, run.__doc__ = function.__name__, function.__doc__
        return spaces.GPU(duration=duration)(run) if ZERO else run

    return wrap


def slim(checkpoint):
    """Drop the weight dictionaries once the model is built (they would stay in RAM otherwise)."""
    return {k: v for k, v in checkpoint.items() if k not in {"model", "ema", "optimizer", "rng"}}


# ---------------------------------------------------------------- startup: codec, published model, ASR, speaker model
def _download(repo, filename, repo_type, token=False):
    return hf_hub_download(repo, filename, repo_type=repo_type, token=token)


PATHS = {DEFAULT_MODEL: _download(*CHECKPOINTS[DEFAULT_MODEL])}
_first = torch.load(PATHS[DEFAULT_MODEL], map_location="cpu", weights_only=True)
CODEC = Codec(_first["codec"]["checkpoint"], DEVICE, loudness=_first["codec"].get("loudness_lufs"))
del _first
MODELS = {DEFAULT_MODEL: Synthesizer(PATHS[DEFAULT_MODEL], device=DEVICE, codec=CODEC, asr_language="tr")}
MODELS[DEFAULT_MODEL].checkpoint = slim(MODELS[DEFAULT_MODEL].checkpoint)

from transformers import (  # noqa: E402
    AutoFeatureExtractor,
    AutoModelForAudioXVector,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

ASR_PROCESSOR = WhisperProcessor.from_pretrained(ASR_NAME)
ASR = WhisperForConditionalGeneration.from_pretrained(
    ASR_NAME, dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE).eval()
SV_EXTRACTOR = AutoFeatureExtractor.from_pretrained(SV_NAME)
SV = AutoModelForAudioXVector.from_pretrained(SV_NAME).to(DEVICE).eval()


def _fetch_other_checkpoints():
    for label, spec in CHECKPOINTS.items():
        if label not in PATHS:
            try:
                PATHS[label] = _download(*spec)
            except Exception as error:  # the dropdown entry then reports the problem when used
                print(f"download failed for {label}: {error}", flush=True)


threading.Thread(target=_fetch_other_checkpoints, daemon=True).start()


def _load_dnsmos():
    path = ROOT / "models" / "sig_bak_ovr.onnx"
    try:
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(DNSMOS_URL, path)
        return DNSMOS(path)
    except Exception as error:
        print(f"DNSMOS unavailable: {error}", flush=True)
        return None


DNSMOS_MODEL = _load_dnsmos()
FREYA = []  # Freya-TR-Eval sentences (CC-BY-4.0), fetched lazily for the batch test


def freya_sentences():
    if not FREYA:
        path = hf_hub_download("freyavoice/freya-tr-eval", "freya_tr_eval.jsonl", repo_type="dataset", token=False)
        FREYA.extend(json.loads(line)["text"] for line in Path(path).read_text().splitlines() if line.strip())
    return FREYA


# ---------------------------------------------------------------- checkpoints typed in by the user
_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def parse_hub_url(text):
    """Accept 'org/name' or a full https://huggingface.co/... URL (blob/resolve) -> (repo, file or None, type)."""
    text = (text or "").strip()
    match = re.match(r"https?://huggingface\.co/(datasets/|spaces/)?([^/]+/[^/]+)(?:/(?:blob|resolve)/[^/]+/(.+))?", text)
    if match:
        kind = {"datasets/": "dataset", "spaces/": "space"}.get(match.group(1) or "", "model")
        return match.group(2), match.group(3), kind
    return text, None, None


def hub_errors(function):
    """Hub failures (missing/private repo, missing file, network) become readable ValueErrors."""

    def run(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (RepositoryNotFoundError, GatedRepoError) as error:
            raise ValueError("Repository not found or no access (for private repositories use 'Log in with Hugging Face')") from error
        except EntryNotFoundError as error:
            raise ValueError("File not found in the repository") from error
        except HfHubHTTPError as error:
            raise ValueError(f"Hub request failed: {str(error)[:160]}") from error

    return run


@hub_errors
def list_checkpoint_files(repo, repo_type, token=None):
    repo, _, kind = parse_hub_url(repo)
    repo_type = kind or repo_type
    if not _REPO.match(repo or ""):
        raise ValueError("The repo ID must have the form 'org/name' (e.g. VoiceHub/dacvae-tts-tr-w512-clean)")
    files = HfApi().list_repo_files(repo, repo_type=repo_type, token=token or False)
    return repo_type, sorted(f for f in files if f.endswith((".pt", ".pth")))


@hub_errors
def resolve_custom(repo, filename, repo_type, token=None):
    """Download a user-named checkpoint (anonymously unless the visitor logged in). Returns the local path."""
    repo, url_file, kind = parse_hub_url(repo)
    filename, repo_type = (url_file or (filename or "").strip()), (kind or repo_type or "dataset")
    if not _REPO.match(repo or "") or not filename.endswith((".pt", ".pth")) or ".." in filename:
        raise ValueError("For a custom checkpoint enter a repo of the form 'org/name' and a .pt/.pth file path")
    info = HfApi().get_paths_info(repo, [filename], repo_type=repo_type, token=token or False)
    if not info:
        raise ValueError(f"File not found: {repo}/{filename}")
    size = getattr(info[0], "size", 0) or 0
    if size > MAX_CUSTOM_BYTES:
        raise ValueError(f"Checkpoint too large ({size / 1e9:.1f} GB; limit {MAX_CUSTOM_BYTES / 1e9:.1f} GB)")
    return _download(repo, filename, repo_type, token=token or False), f"{repo}/{filename}"


@hub_errors
def model_path(choice):
    """Main-process side: make sure a listed checkpoint is on disk. Returns (path, display name)."""
    if choice not in CHECKPOINTS:
        raise ValueError(f"Unknown checkpoint: {choice}")
    if choice not in PATHS:
        PATHS[choice] = _download(*CHECKPOINTS[choice])
    return PATHS[choice], choice


_WORKER_MODELS = OrderedDict()  # inside a ZeroGPU worker (or the only process locally): path -> Synthesizer


def model_for(path):
    """Worker side: the published model is preloaded; others are loaded once per worker and kept (LRU of 3)."""
    if path == PATHS[DEFAULT_MODEL]:
        return MODELS[DEFAULT_MODEL]
    if path not in _WORKER_MODELS:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        missing = {"config", "ema", "codec", "mean", "std"} - set(checkpoint)
        if missing:
            raise ValueError(f"This file is not a DACVAE-TTS checkpoint (missing: {sorted(missing)})")
        del checkpoint
        while len(_WORKER_MODELS) >= 3:
            _WORKER_MODELS.popitem(last=False)
            torch.cuda.empty_cache() if DEVICE == "cuda" else None
        synth = Synthesizer(path, device=DEVICE, codec=CODEC, asr_language="tr")
        synth.checkpoint = slim(synth.checkpoint)
        _WORKER_MODELS[path] = synth
    _WORKER_MODELS.move_to_end(path)
    return _WORKER_MODELS[path]


# ---------------------------------------------------------------- GPU helpers (called inside GPU jobs)
def to_16k(audio):
    return resample_poly(np.asarray(audio, dtype=np.float32), 1, SAMPLE_RATE // 16000).astype(np.float32)


@torch.inference_mode()
def transcribe_many(clips_16k, beams=5, batch=8):
    texts = []
    for start in range(0, len(clips_16k), batch):
        chunk = [np.asarray(c, dtype=np.float32)[: 30 * 16000] for c in clips_16k[start : start + batch]]
        features = ASR_PROCESSOR(chunk, sampling_rate=16000, return_tensors="pt").input_features
        ids = ASR.generate(features.to(DEVICE, ASR.dtype), language="tr", task="transcribe", num_beams=beams,
                           max_new_tokens=220)
        texts.extend(t.strip() for t in ASR_PROCESSOR.batch_decode(ids, skip_special_tokens=True))
    return texts


@torch.inference_mode()
def embed(clip_16k):
    inputs = SV_EXTRACTOR(np.asarray(clip_16k, dtype=np.float32), sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    return torch.nn.functional.normalize(SV(**inputs).embeddings.float(), dim=-1)[0]


def score(text, hypothesis):
    """WER/CER of a chunk. The hypothesis goes through the same frontend as the input text, so that Whisper's
    written forms ("ABD'den", "Dr.", "23 Eylül") are compared with their spoken forms ("a be deden", "Doktor")."""
    try:
        spoken = speakable(hypothesis)[0] if hypothesis and hypothesis.strip() else ""
    except ValueError:
        spoken = hypothesis or ""
    try:
        return error_counts(text, spoken, "turkish-v1")
    except ValueError:
        words = max(len(text.split()), 1)
        return {"wer": 1.0, "cer": 1.0, "word_edits": words, "words": words, "char_edits": len(text), "chars": len(text)}


def synthesize_job(tts, job, reference_embedding=None):
    """Generate all chunks x candidates, rank candidates by Whisper CER (then WER, then similarity)."""
    voice = tts.prepare_reference((job["reference"], SAMPLE_RATE), job["reference_text"])
    started = time.time()
    if job.get("seconds") is None:
        results, meta = tts.synthesize_many(
            job["chunks"], voice, candidates=job["candidates"], duration_scale=job["duration_scale"],
            duration_mode=job["duration_mode"], steps=job["steps"], guidance=job["guidance"], seed=job["seed"],
            sway=-1.0, **job["sampler"],
        )
    else:
        results, meta = _fixed_rate(tts, job, voice)
    generation = time.time() - started
    need_asr = job["verify"] or job["candidates"] > 1
    flat = [c for row in results for c in row]
    hypotheses = transcribe_many([to_16k(c["audio"]) for c in flat]) if need_asr else [None] * len(flat)
    if job["verify"] and reference_embedding is None:
        reference_embedding = embed(to_16k(job["reference"]))
    chunks, position = [], 0
    for text, row in zip(job["chunks"], results):
        ranked = []
        for index, candidate in enumerate(row):
            hypothesis = hypotheses[position]
            position += 1
            counts = score(text, hypothesis) if hypothesis is not None else None
            similarity = float(embed(to_16k(candidate["audio"])) @ reference_embedding) if job["verify"] else None
            ranked.append({"index": index, "hypothesis": hypothesis, "counts": counts, "similarity": similarity,
                           "seconds": candidate["audio_seconds"]})
        best = min(ranked, key=lambda r: (r["counts"]["cer"], r["counts"]["wer"], -(r["similarity"] or 0), r["index"])) \
            if need_asr else ranked[0]
        chunks.append({"text": text, "audio": row[best["index"]]["audio"], "selected": best["index"],
                       "candidates": ranked, "duration": row[best["index"]]["duration"]})
    return {"chunks": chunks, "generation_seconds": generation, "meta": meta, "step": tts.checkpoint.get("step"),
            "text_normalization": tts.text_version}


def _fixed_rate(tts, job, voice):
    """Fixed speaking rate: one call per chunk (each has its own length in seconds)."""
    results, meta = [], {}
    for text, seconds in zip(job["chunks"], job["seconds"]):
        row, meta = tts.synthesize_many([text], voice, candidates=job["candidates"], seconds=seconds,
                                        steps=job["steps"], guidance=job["guidance"], seed=job["seed"], sway=-1.0,
                                        **job["sampler"])
        results.extend(row)
    return results, meta


# ---------------------------------------------------------------- GPU jobs (dynamic durations)
def _synthesis_seconds(job):
    rows = len(job["chunks"]) * job["candidates"]
    asr = job["verify"] or job["candidates"] > 1
    return 10 + 1.0 * rows + (0.8 * rows if asr else 0) + (6 if job["model"] != PATHS[DEFAULT_MODEL] else 0)


def _duration_generate(job):
    return int(min(15 + _synthesis_seconds(job) + (6 if job["reference_text"] is None else 0), 180))


@gpu(_duration_generate)
def run_generate(job):
    """One synthesis request. The reference transcript is produced here when the visitor left it empty."""
    started = time.time()
    tts = model_for(job["model"])
    load = time.time() - started
    auto = None
    if job["reference_text"] is None:
        auto = transcribe_many([to_16k(job["reference"])])[0]
        try:
            job["reference_text"], _ = speakable(auto)
        except ValueError as error:
            raise ValueError("Could not extract text from the reference audio; type the transcript by hand") from error
    result = synthesize_job(tts, job)
    result.update(model_load_seconds=load, gpu_seconds=time.time() - started, reference_text=job["reference_text"],
                  reference_asr=auto)
    return result


def _duration_compare(jobs):
    return int(min(20 + sum(_synthesis_seconds(job) for job in jobs), 200))


@gpu(_duration_compare)
def run_compare(jobs):
    """The same request on several checkpoints (same seed and reference), one GPU session."""
    reference_embedding = None
    out = []
    for job in jobs:
        started = time.time()
        tts = model_for(job["model"])
        if job["verify"] and reference_embedding is None:
            reference_embedding = embed(to_16k(job["reference"]))
        result = synthesize_job(tts, job, reference_embedding)
        result.update(gpu_seconds=time.time() - started, label=job["label"])
        out.append(result)
    return out


@gpu(lambda audio: 20)
def run_transcribe(audio_48k):
    return transcribe_many([to_16k(audio_48k)])[0]


@gpu(lambda: 60)
def run_diagnostics():
    report = [f"torch {torch.__version__} · cuda={torch.cuda.is_available()} · zero={ZERO} · device={DEVICE} · asr={ASR_NAME}"]
    started = time.time()
    example = json.loads((ROOT / "examples" / "prompts.json").read_text(encoding="utf-8"))[0]
    audio = read_audio(str(ROOT / example["audio"]), SAMPLE_RATE).numpy()
    report.append(f"whisper: {transcribe_many([to_16k(audio)])[0][:80]} ({time.time() - started:.1f} s)")
    started = time.time()
    tts = MODELS[DEFAULT_MODEL]
    voice = tts.prepare_reference((audio, SAMPLE_RATE), speakable(example["text"])[0])
    results, _ = tts.synthesize_many(["Merhaba, bu bir deneme."], voice, steps=8, guidance=5.0)
    report.append(f"synthesis: {results[0][0]['audio_seconds']:.1f} s audio ({time.time() - started:.1f} s)")
    return "\n".join(report)


# ---------------------------------------------------------------- main process: requests, post-processing, reports
def load_reference(path, max_seconds=15.0):
    """Read, mix to mono, resample to 48 kHz and trim silence. Returns (audio, notes)."""
    if not path:
        raise ValueError("Upload or record a reference audio (3–15 s, single speaker)")
    audio = read_audio(path, SAMPLE_RATE).numpy()
    notes = []
    trimmed, start, end = trim_silence(audio, SAMPLE_RATE)
    if len(audio) - len(trimmed) > 0.3 * SAMPLE_RATE:
        notes.append(f"leading/trailing silence trimmed ({(len(audio) - len(trimmed)) / SAMPLE_RATE:.1f} s)")
    seconds = len(trimmed) / SAMPLE_RATE
    if seconds < 1.0:
        raise ValueError(f"Reference too short ({seconds:.1f} s); at least 3 s of speech is recommended")
    if seconds > 30:
        raise ValueError(f"Reference too long ({seconds:.0f} s); upload a 3–15 s excerpt")
    if seconds > max_seconds:
        notes.append(f"long reference ({seconds:.0f} s): the model was trained on 3–15 s prompts; shortening it may improve quality")
    elif seconds < 3:
        notes.append(f"short reference ({seconds:.1f} s): voice similarity may drop")
    return trimmed, notes


def cut_for_transcript(audio, max_seconds=12.0):
    """For automatic transcripts only: cut a long reference at its quietest 10 ms frame between 8 s and max."""
    if len(audio) <= max_seconds * SAMPLE_RATE:
        return audio
    size = SAMPLE_RATE // 100
    low, high = int(8 * 100), int(max_seconds * 100)
    frames = audio[: high * size].reshape(high, size)
    energy = (frames[low:] ** 2).mean(1)
    cut = (low + int(np.argmin(energy))) * size
    return audio[:cut]


def dnsmos(audio_48k):
    if DNSMOS_MODEL is None or len(audio_48k) < 16000:
        return None
    try:
        return DNSMOS_MODEL(to_16k(audio_48k))
    except Exception:
        return None


def reference_report(audio, transcript, notes):
    seconds = len(audio) / SAMPLE_RATE
    lines = [f"**Reference:** {seconds:.1f} s"]
    if transcript:
        rate = speaking_rate(seconds * 25, transcript)
        label = "fast" if rate > 17 else ("slow" if rate < 13 else "normal")
        lines.append(f"speaking rate {rate:.1f} chars/s ({label}; corpus median 15)")
        if rate > 17:
            notes = notes + ["fast reference: the 'Automatic' rate slows the output down to ~16 chars/s (fast prompts produced the most errors)"]
        elif rate < 13:
            notes = notes + ["slow reference (may contain long pauses): the 'Automatic' rate sets the duration with the duration predictor"]
    quality = dnsmos(audio)
    if quality:
        lines.append(f"DNSMOS {quality['dnsmos_ovrl']:.2f}")
        if quality["dnsmos_ovrl"] < 2.7:
            notes = notes + ["the reference sounds noisy/reverberant (DNSMOS < 2.7): output quality and intelligibility may drop"]
    text = " · ".join(lines)
    if notes:
        text += "\n\n" + "\n".join(f"- ⚠️ {n}" for n in notes)
    return text


def plan_text(text, reference_seconds, rate=15.0, fixed_rate=None):
    """Frontend + chunking. Chunks keep prompt + target within the lengths seen in training (~20-25 s)."""
    if not text or not text.strip():
        raise ValueError("Enter the text to speak")
    from dacvae_tts.frontend import prepare_text

    prepared = prepare_text(text)
    target_seconds = min(max(22.0 - reference_seconds, 7.0), 14.0)
    max_chars = int(target_seconds * (fixed_rate or rate))
    pieces = split_sentences(prepared.text, max_chars=max(max_chars, 80))
    chunks, pauses, changes = [], [], list(prepared.changes)
    for piece, kind in pieces:
        try:
            normalized, extra = speakable(piece)
        except ValueError:
            continue
        changes.extend(c for c in extra if c not in changes)
        chunks.append(normalized)
        pauses.append(PAUSES[kind])
    if not chunks:
        raise ValueError("Nothing readable is left in the text")
    return chunks, pauses, changes


def assemble(result, pauses):
    """Trim each chunk, join with pauses, finalize loudness. Returns (float audio, info)."""
    pieces = []
    for chunk in result["chunks"]:
        trimmed, _, _ = trim_silence(chunk["audio"], SAMPLE_RATE, margin_start_ms=40, margin_end_ms=80)
        pieces.append(trimmed)
    audio = join(pieces, SAMPLE_RATE, pauses) if len(pieces) > 1 else pieces[0]
    return finalize(audio, SAMPLE_RATE)


def write_wav(audio, prefix="dacvae-tts"):
    path = Path(tempfile.mkdtemp()) / f"{prefix}.wav"
    sf.write(path, to_pcm16(audio), SAMPLE_RATE, subtype="PCM_16")
    return str(path)


def summarize_chunks(chunks):
    """Corpus WER/CER over chunks (sum of edits / sum of words) and mean similarity of the selected candidates."""
    selected = [c["candidates"][c["selected"]] for c in chunks]
    if any(s["counts"] is None for s in selected):
        return None
    words = sum(s["counts"]["words"] for s in selected)
    chars = sum(s["counts"]["chars"] for s in selected)
    out = {
        "wer": sum(s["counts"]["word_edits"] for s in selected) / max(words, 1),
        "cer": sum(s["counts"]["char_edits"] for s in selected) / max(chars, 1),
        "words": words,
        "hypothesis": " ".join(s["hypothesis"] or "" for s in selected),
    }
    similarities = [s["similarity"] for s in selected if s["similarity"] is not None]
    if similarities:
        out["similarity"] = float(np.mean(similarities))
    return out


def make_zip(files, rows, summary):
    folder = Path(tempfile.mkdtemp())
    path = folder / "batch-test.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, wav in files:
            archive.write(wav, name)
        archive.writestr("results.jsonl", "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        archive.writestr("summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    return str(path)


def loudness_of(audio):
    value = loudness(audio, SAMPLE_RATE)
    return None if math.isnan(value) else value


__all__ = [
    "CHECKPOINTS", "DEFAULT_MODEL", "DEFAULTS", "MAX_BATCH_SENTENCES", "SAMPLE_RATE", "ZERO", "DEVICE", "ASR_NAME",
    "VoiceReference", "assemble", "cut_for_transcript", "dnsmos", "freya_sentences", "list_checkpoint_files",
    "load_reference", "loudness_of", "make_zip", "model_path", "plan_text", "reference_report", "resolve_custom",
    "run_compare", "run_diagnostics", "run_generate", "run_transcribe", "summarize_chunks", "write_wav",
]
