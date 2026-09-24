"""Evaluation protocol v2 (issue #3): SIM-o, UTMOS, deterministic Whisper, hallucination flags, signal statistics.

Why each part exists:
- SIM-o (`dacvae_tts.sim_o`): the literature's speaker similarity. `speaker_similarity` (wavlm-base-plus-sv) stays
  as the development metric under its old key; SIM-o is reported as `sim_o` against the ORIGINAL prompt recording
  and `sim_r` against a codec-resynthesized prompt. A second, independent speaker model (SpeechBrain ECAPA,
  VoxCeleb; `sim_o_speechbrain`/`sim_r_speechbrain`) guards against training/reranking and evaluating with the
  same embedding model.
- Quality: UTMOS22-strong (SpeechMOS) and optional UTMOSv2 next to DNSMOS SIG/BAK/OVRL; the prompt's own DNSMOS
  (`prompt_dnsmos_*`) separates "the prompt was noisy" from "the model added noise".
- Signal statistics on the full-band file: fraction of samples at |x| >= 0.999 (the decoder's tanh ceiling),
  integrated loudness (LUFS) and the -50 dB bandwidth (highest frequency of the average power spectrum within
  50 dB of its peak; a codec/vocoder that loses the top octave shows up here before listeners complain).
- Whisper: faster-whisper falls back from temperature 0 to sampling (0.2 ... 1.0) on clips whose greedy result
  looks degenerate, so the same audio can score differently. `asr_deterministic` pins temperature=0,
  condition_on_previous_text=False, without_timestamps=True, beam 5, and records library versions and the model
  snapshot in the evaluator identity.
- Whisper-large-v3 hallucinates fixed strings on (near-)silence (arXiv 2501.11378); in Turkish subtitles-style
  phrases such as "Altyazı M.K.". `flag_hallucinations` marks such phrases (only occurrences beyond those in the
  reference text) and reports `wer_filtered`/`cer_filtered` NEXT TO the raw `wer`/`cer`, which are never replaced.
  `asr_trim_silence` optionally cuts trailing silence before ASR (the raw audio is still what every other metric
  sees).
- `band_limit_8k` (FreyaTTS scoring protocol): the audio is resampled to 8 kHz and back to 16 kHz before ASR only,
  so WER is comparable with Freya-TR-Eval tables.

Everything is opt-in: `protocol_from_args` returns None unless a v2 flag is given, and an `Evaluator` without a
protocol keeps its decoding, row keys and evaluator identity exactly as before. Heavy models are imported lazily.
"""

import importlib.metadata
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

# torch/soundfile are imported inside functions: the CLI builds this module's argparse flags for every command.
SAMPLE_RATE = 16000
BAND_LIMIT_RATE = 8000
CLIP_THRESHOLD = 0.999
SIM_O_BACKBONES = ("transformers", "s3prl")  # == sim_o.BACKBONES (not imported: sim_o needs torch)

# Fixed Whisper outputs on silence/music, kept deliberately short: only multi-word phrases that are implausible as
# the tail of an evaluation sentence. Matching is on metric-normalized word sequences, so casing and punctuation
# ("Altyazı M.K." / "ALTYAZI M. K") do not matter; a phrase that also occurs in the reference is not flagged.
KNOWN_HALLUCINATIONS = (
    "Altyazı M.K.",
    "İzlediğiniz için teşekkürler",
    "İzlediğiniz için teşekkür ederim",
    "Abone olmayı unutmayın",
    "Thank you for watching",
    "Thanks for watching",
    "Subtitles by the Amara.org community",
)

# The decoding `Evaluator.score` has always used (kept bit-for-bit when no protocol is given) and the v2 decoding.
WHISPER_V1 = dict(beam_size=5, vad_filter=False, condition_on_previous_text=False)
WHISPER_DETERMINISTIC = dict(WHISPER_V1, temperature=0.0, without_timestamps=True)

# Per-row metrics whose mean `summary_extras` reports (over rows that have a finite value).
MEAN_KEYS = (
    "sim_o", "sim_r", "sim_o_speechbrain", "sim_r_speechbrain", "utmos", "utmosv2",
    "prompt_dnsmos_sig", "prompt_dnsmos_bak", "prompt_dnsmos_ovrl",
    "clipped_fraction_fullband", "loudness_lufs", "bandwidth_hz", "prompt_bandwidth_hz", "asr_trimmed_seconds",
)


@dataclass(frozen=True)
class ProtocolOptions:
    """Switches of evaluation protocol v2; every field defaults to the v1 behavior (off)."""

    asr_deterministic: bool = False
    asr_trim_silence: bool = False
    band_limit_8k: bool = False
    flag_hallucinations: bool = False
    signal_stats: bool = False
    prompt_dnsmos: bool = False
    sim_o: bool = False
    sim_o_backend: str = "transformers"
    sim_o_checkpoint: str = None
    sim_speechbrain: bool = False
    utmos: bool = False
    utmosv2: bool = False

    def __post_init__(self):
        if self.sim_o_backend not in SIM_O_BACKBONES:
            raise ValueError(f"Unknown SIM-o backend {self.sim_o_backend!r}; choose from {SIM_O_BACKBONES}")
        if self.sim_o_checkpoint is not None and not self.sim_o:
            raise ValueError("sim_o_checkpoint is set but SIM-o is disabled")

    @property
    def enabled(self):
        return any(getattr(self, f.name) is True for f in fields(self))


_SWITCHES = tuple(f.name for f in fields(ProtocolOptions) if f.type is bool)


def protocol_v2(**overrides):
    """The bundle behind `--protocol-v2`: everything except UTMOSv2, silence trimming and the 8 kHz band limit,
    which change what is measured or need an extra package and stay separate switches."""
    return ProtocolOptions(**{
        **dict(asr_deterministic=True, flag_hallucinations=True, signal_stats=True, prompt_dnsmos=True, sim_o=True,
               sim_speechbrain=True, utmos=True),
        **overrides,
    })


def add_protocol_args(parser):
    """Add the v2 switches (all off by default) to an argparse parser."""
    group = parser.add_argument_group(
        "evaluation protocol v2", "Opt-in (issue #3). Without these flags scoring is unchanged."
    )
    group.add_argument(
        "--protocol-v2", action="store_true",
        help="Shorthand for --asr-deterministic --flag-hallucinations --signal-stats --sim-o --sim-speechbrain "
             "--utmos, plus --prompt-dnsmos when a DNSMOS model is given",
    )
    group.add_argument("--asr-deterministic", action="store_true",
                       help="faster-whisper with temperature=0 (no sampling fallback), without_timestamps, beam 5")
    group.add_argument("--asr-trim-silence", action="store_true", help="Cut trailing silence before ASR only")
    group.add_argument("--band-limit-8k", action="store_true",
                       help="FreyaTTS protocol: resample to 8 kHz and back to 16 kHz before ASR only")
    group.add_argument("--flag-hallucinations", action="store_true",
                       help="Flag known Whisper hallucination phrases; adds wer_filtered/cer_filtered next to raw WER")
    group.add_argument("--signal-stats", action="store_true",
                       help="Full-band clipped-sample fraction, integrated LUFS and -50 dB bandwidth (Hz)")
    group.add_argument("--prompt-dnsmos", action="store_true",
                       help="DNSMOS of the original prompt recording (needs a DNSMOS model and original prompts)")
    group.add_argument("--sim-o", action="store_true",
                       help="seed-tts-eval SIM (WavLM-Large ECAPA-TDNN): sim_o vs the original prompt, sim_r vs "
                            "the codec-decoded prompt")
    group.add_argument("--sim-o-checkpoint",
                       help="Local wavlm_large_finetune.pth (sha256-checked); default: download the pinned HF mirror")
    group.add_argument("--sim-o-backend", choices=SIM_O_BACKBONES, default="transformers",
                       help="WavLM-Large implementation (s3prl = upstream torch.hub path, needs s3prl)")
    group.add_argument("--sim-speechbrain", action="store_true",
                       help="Second, independent speaker metric: speechbrain/spkrec-ecapa-voxceleb (needs speechbrain)")
    group.add_argument("--utmos", action="store_true", help="UTMOS22-strong (torch.hub tarepan/SpeechMOS:v1.2.0)")
    group.add_argument("--utmosv2", action="store_true", help="UTMOSv2 (needs the utmosv2 package)")
    return group


def protocol_from_args(args):
    """ProtocolOptions from parsed flags, or None when no v2 flag was given (v1 behavior, nothing changes)."""
    has_dnsmos = bool(getattr(args, "dnsmos", None) or getattr(args, "dnsmos_model", None))
    chosen = {name: True for name in _SWITCHES if getattr(args, name, False)}
    if chosen.get("prompt_dnsmos") and not has_dnsmos:
        raise ValueError("--prompt-dnsmos needs a DNSMOS model (--dnsmos / --dnsmos-model)")
    extra = dict(sim_o_backend=getattr(args, "sim_o_backend", None) or "transformers",
                 sim_o_checkpoint=getattr(args, "sim_o_checkpoint", None))
    if getattr(args, "protocol_v2", False):
        return protocol_v2(**{"prompt_dnsmos": has_dnsmos, **chosen}, **extra)
    if not chosen:
        if extra["sim_o_checkpoint"]:
            raise ValueError("--sim-o-checkpoint needs --sim-o or --protocol-v2")
        return None
    return ProtocolOptions(**chosen, **extra)


def manifest_prompts(row, reference_kind=None):
    """Prompt paths of a manifest row for `Evaluator.score`: explicit `original_prompt_audio` /
    `codec_prompt_audio` keys win; otherwise `reference_audio` is used only when `reference_kind` says what it is
    ("original" recording or "codec" resynthesis). It is never guessed: mixing the two would blend SIM-o and SIM-r."""
    if reference_kind not in (None, "original", "codec"):
        raise ValueError(f"Unknown reference kind {reference_kind!r}")
    prompts = {"original_prompt": row.get("original_prompt_audio"), "codec_prompt": row.get("codec_prompt_audio")}
    key = {"original": "original_prompt", "codec": "codec_prompt"}.get(reference_kind)
    if key and not prompts[key]:
        prompts[key] = row.get("reference_audio")
    return prompts


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


# ----------------------------------------------------------------------------------------------- pure functions


def clipped_fraction(audio, threshold=CLIP_THRESHOLD):
    """Fraction of samples (all channels) with |x| >= threshold; 0 for empty input."""
    audio = np.asarray(audio)
    return float(np.mean(np.abs(audio) >= threshold)) if audio.size else 0.0


def bandwidth_hz(audio, sample_rate, floor_db=-50.0, n_fft=2048):
    """Highest frequency whose average power (Hann-windowed frames, 50% overlap) is within `floor_db` of the peak.

    Returns 0.0 for silent input. Resolution is sample_rate / n_fft (23 Hz at 48 kHz).
    """
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[:: n_fft // 2]
    power = np.mean(np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2, axis=0)
    peak = power.max()
    if not np.isfinite(peak) or peak <= 0:
        return 0.0
    above = np.flatnonzero(10 * np.log10(np.maximum(power / peak, 1e-30)) >= floor_db)
    return float(np.fft.rfftfreq(n_fft, 1.0 / sample_rate)[above[-1]])


def band_limit(audio, sample_rate=SAMPLE_RATE, limit_rate=BAND_LIMIT_RATE):
    """Resample to `limit_rate` and back (polyphase), keeping the input length and sample rate."""
    import torch
    from scipy.signal import resample_poly

    tensor = torch.is_tensor(audio)
    wave = np.asarray(audio.cpu().numpy() if tensor else audio, dtype=np.float64).reshape(-1)
    factor = math.gcd(sample_rate, limit_rate)
    low = resample_poly(wave, limit_rate // factor, sample_rate // factor)
    back = resample_poly(low, sample_rate // factor, limit_rate // factor)[: len(wave)]
    back = np.pad(back, (0, len(wave) - len(back))).astype(np.float32)
    return torch.from_numpy(back) if tensor else back


def trim_trailing_silence(audio, sample_rate=SAMPLE_RATE):
    """Cut trailing silence with `audio.trim_silence`'s rule (the start is kept). Returns (audio, removed samples)."""
    import torch

    from .audio import trim_silence

    tensor = torch.is_tensor(audio)
    wave = np.asarray(audio.cpu().numpy() if tensor else audio, dtype=np.float32).reshape(-1)
    _, _, end = trim_silence(wave, sample_rate)
    trimmed = wave[:end]
    return (torch.from_numpy(trimmed.copy()) if tensor else trimmed), len(wave) - end


def _occurrences(words, phrase):
    starts, index = [], 0
    while index + len(phrase) <= len(words):
        if words[index : index + len(phrase)] == phrase:
            starts.append(index)
            index += len(phrase)
        else:
            index += 1
    return starts


def find_hallucinations(hypothesis, reference=None, normalization="turkish-v1", phrases=KNOWN_HALLUCINATIONS):
    """Remove known hallucination phrases from a hypothesis.

    Returns (filtered hypothesis as metric-normalized text, list of removed phrases). Occurrences also present in
    the reference are legitimate: only the last (hypothesis count - reference count) occurrences are removed.
    """
    from .metrics import metric_text

    words = metric_text(hypothesis, normalization).split()
    reference_words = metric_text(reference, normalization).split() if reference else []
    removed = []
    for phrase in phrases:
        target = metric_text(phrase, normalization).split()
        if not target:
            continue
        starts = _occurrences(words, target)
        excess = len(starts) - len(_occurrences(reference_words, target))
        if excess <= 0:
            continue
        drop = set()
        for start in starts[-excess:]:
            drop.update(range(start, start + len(target)))
        words = [w for i, w in enumerate(words) if i not in drop]
        removed.extend([phrase] * excess)
    return " ".join(words), removed


def signal_stats(path):
    """Full-band statistics of an audio file: clipping on every channel, loudness/bandwidth of the channel mix."""
    import soundfile as sf

    from .audio import loudness

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"Empty or nonfinite audio: {path}")
    mono = audio.mean(axis=1)
    lufs = loudness(mono, rate)
    return {
        "clipped_fraction_fullband": clipped_fraction(audio),
        "loudness_lufs": float(lufs) if np.isfinite(lufs) else None,  # undefined below 0.4 s or for silence
        "bandwidth_hz": bandwidth_hz(mono, rate),
    }


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summary_extras(rows):
    """Summary keys added by protocol v2 to `metrics.summarize` (whose existing keys are unchanged).

    Per-utterance means complement the corpus-level WER/CER (a few long failures dominate the corpus rate, many
    short ones the mean); S/D/I totals show whether errors are dropped or invented words.
    """
    result = {}
    if all("wer" in r for r in rows):
        result["wer_mean"] = float(np.mean([r["wer"] for r in rows]))
        result["wer_over_half_fraction"] = float(np.mean([r["wer"] > 0.5 for r in rows]))
    if all("cer" in r for r in rows):
        result["cer_mean"] = float(np.mean([r["cer"] for r in rows]))
    if all("word_edits" in r for r in rows):
        result["error_free_fraction"] = float(np.mean([r["word_edits"] == 0 for r in rows]))
    for key in ("word_substitutions", "word_deletions", "word_insertions"):
        if all(key in r for r in rows):
            result[key] = int(sum(r[key] for r in rows))
    if all("word_edits_filtered" in r and "char_edits_filtered" in r for r in rows):
        result["wer_filtered"] = sum(r["word_edits_filtered"] for r in rows) / sum(r["words"] for r in rows)
        result["cer_filtered"] = sum(r["char_edits_filtered"] for r in rows) / sum(r["chars"] for r in rows)
    if all("hallucination" in r for r in rows):
        result["hallucination_rate"] = float(np.mean([bool(r["hallucination"]) for r in rows]))
    for key in MEAN_KEYS:
        values = [r[key] for r in rows if _finite(r.get(key))]
        if values:
            result[key] = float(np.mean(values))
            if len(values) < len(rows):
                result[f"{key}_rows"] = len(values)
    return result


def whisper_snapshot(asr_model):
    """Hugging Face snapshot (commit) of a faster-whisper model already in the local cache, else None."""
    if Path(str(asr_model)).is_dir():
        return str(Path(asr_model).resolve())
    try:
        from faster_whisper.utils import download_model

        return Path(download_model(asr_model, local_files_only=True)).name
    except Exception:  # not cached / offline / unknown size name: identity stays best-effort
        return None


# ------------------------------------------------------------------------------------------- optional models


class _FileEmbeddings:
    """Embedding of an audio file with the seed-tts-eval input protocol, prompt embeddings cached."""

    def file_embedding(self, path, cache=False):
        from .sim_o import load_audio_16k

        key = str(Path(path).resolve())
        store = self.__dict__.setdefault("_cache", {})
        if cache and key in store:
            return store[key]
        embedding = self.embedding(load_audio_16k(path))
        if cache:
            store[key] = embedding
        return embedding


class SpeechBrainECAPA(_FileEmbeddings):
    """speechbrain/spkrec-ecapa-voxceleb (192-d); independent of the WavLM models used elsewhere."""

    SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(self, device="cpu", savedir=None):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            try:
                from speechbrain.pretrained import EncoderClassifier  # speechbrain < 1.0
            except ImportError as error:
                raise RuntimeError("--sim-speechbrain needs speechbrain (pip install speechbrain)") from error
        savedir = savedir or str(Path.home() / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb")
        self.model = EncoderClassifier.from_hparams(source=self.SOURCE, savedir=savedir,
                                                    run_opts={"device": str(device)})
        self.identity = {"model": self.SOURCE, "speechbrain_version": _version("speechbrain")}

    def embedding(self, audio):
        import torch
        from torch.nn import functional as F

        with torch.inference_mode():
            wave = torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1)
            return F.normalize(self.model.encode_batch(wave).reshape(1, -1).float(), dim=-1)[0].cpu()


class UTMOS22Strong:
    """UTMOS22 strong learner, SpeechMOS port (MOS on a 1-5 scale from 16 kHz audio)."""

    REPO = "tarepan/SpeechMOS:v1.2.0"

    def __init__(self, device="cpu"):
        import torch

        try:
            model = torch.hub.load(self.REPO, "utmos22_strong", trust_repo=True)
        except Exception as error:  # hub download / dependency failure
            raise RuntimeError(f"--utmos could not load torch.hub {self.REPO} utmos22_strong") from error
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.identity = {"model": "utmos22_strong", "repo": self.REPO}

    def __call__(self, audio):
        import torch

        with torch.inference_mode():
            wave = torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1).to(self.device)
            return float(self.model(wave, SAMPLE_RATE).reshape(-1)[0])


class UTMOSv2Scorer:
    """UTMOSv2 (fusion_stage3, fold 0); optional package."""

    def __init__(self, device="cpu"):
        try:
            import utmosv2
        except ImportError as error:
            raise RuntimeError(
                "--utmosv2 needs the utmosv2 package (pip install git+https://github.com/sarulab-speech/UTMOSv2.git)"
            ) from error
        self.device = str(device)
        self.model = utmosv2.create_model(pretrained=True, device=self.device)
        self.identity = {"model": "utmosv2 fusion_stage3 fold0", "utmosv2_version": _version("utmosv2")}

    def __call__(self, audio):
        wave = np.asarray(audio.cpu().numpy() if hasattr(audio, "cpu") else audio, dtype=np.float32).reshape(-1)
        return float(np.asarray(self.model.predict(data=wave, sr=SAMPLE_RATE, device=self.device, verbose=False))
                     .reshape(-1)[0])


class ProtocolScorer:
    """Protocol v2 extras for one evaluator. Models are created from `options` unless injected (tests)."""

    def __init__(self, options, device="cpu", dnsmos=None, *, sim_o=None, speechbrain=None, utmos=None,
                 utmosv2=None):
        import torch

        if not isinstance(options, ProtocolOptions):
            raise TypeError("options must be ProtocolOptions")
        if options.prompt_dnsmos and dnsmos is None:
            raise ValueError("prompt_dnsmos needs a DNSMOS model")
        self.options = options
        self.dnsmos = dnsmos
        if options.sim_o and sim_o is None:
            from .sim_o import SimO

            sim_o = SimO(options.sim_o_checkpoint, options.sim_o_backend, device)
        if options.sim_speechbrain and speechbrain is None:
            speechbrain = SpeechBrainECAPA(device)
        if options.utmos and utmos is None:
            utmos = UTMOS22Strong(device)
        if options.utmosv2 and utmosv2 is None:
            utmosv2 = UTMOSv2Scorer(device)
        self.sim_o = sim_o if options.sim_o else None
        self.speechbrain = speechbrain if options.sim_speechbrain else None
        self.utmos = utmos if options.utmos else None
        self.utmosv2 = utmosv2 if options.utmosv2 else None
        self._prompt_cache = {}
        self.identity = {
            "version": "v2",
            "options": {k: v for k, v in asdict(options).items()
                        if v not in (False, None) and (options.sim_o or not k.startswith("sim_o_"))},
            "whisper_decoding": self.whisper_kwargs(),
            "band_limit_rate": BAND_LIMIT_RATE if options.band_limit_8k else None,
            "hallucination_phrases": list(KNOWN_HALLUCINATIONS) if options.flag_hallucinations else None,
            "torch_version": torch.__version__,
            "ctranslate2_version": _version("ctranslate2"),
            "transformers_version": _version("transformers"),
        }
        for name in ("sim_o", "speechbrain", "utmos", "utmosv2"):
            model = getattr(self, name)
            if model is not None:
                self.identity[name] = getattr(model, "identity", type(model).__name__)

    def whisper_kwargs(self):
        return dict(WHISPER_DETERMINISTIC if self.options.asr_deterministic else WHISPER_V1)

    def asr_audio(self, audio, sample_rate=SAMPLE_RATE):
        """The waveform given to ASR (trim, then band limit) and row fields describing what was done."""
        info = {}
        if self.options.asr_trim_silence:
            audio, removed = trim_trailing_silence(audio, sample_rate)
            info["asr_trimmed_seconds"] = removed / sample_rate
        if self.options.band_limit_8k:
            audio = band_limit(audio, sample_rate)
        return audio, info

    def _per_prompt(self, kind, path, compute):
        key = (kind, str(Path(path).resolve()))
        if key not in self._prompt_cache:
            self._prompt_cache[key] = compute(path)
        return self._prompt_cache[key]

    def score(self, audio_path, audio, text, hypothesis, normalization, original_prompt=None, codec_prompt=None):
        """Extra row fields. `audio` is the 16 kHz waveform used by the v1 metrics; `original_prompt` is the
        recording the voice came from (SIM-o), `codec_prompt` its codec resynthesis (SIM-r)."""
        from .metrics import error_counts

        result = {}
        if self.options.flag_hallucinations:
            filtered, removed = find_hallucinations(hypothesis, text, normalization)
            counts = error_counts(text, filtered, normalization)
            result.update(
                hallucination=bool(removed), hallucination_phrases=removed, hypothesis_filtered=filtered,
                word_edits_filtered=counts["word_edits"], char_edits_filtered=counts["char_edits"],
                wer_filtered=counts["wer"], cer_filtered=counts["cer"],
            )
        if self.options.signal_stats:
            result.update(signal_stats(audio_path))
            if original_prompt:
                result["prompt_bandwidth_hz"] = self._per_prompt(
                    "bandwidth", original_prompt, lambda p: signal_stats(p)["bandwidth_hz"]
                )
        if self.utmos is not None:
            result["utmos"] = self.utmos(audio)
        if self.utmosv2 is not None:
            result["utmosv2"] = self.utmosv2(audio)
        for model, suffix in ((self.sim_o, ""), (self.speechbrain, "_speechbrain")):
            if model is None or not (original_prompt or codec_prompt):
                continue
            generated = model.file_embedding(audio_path)
            for field, prompt in (("sim_o", original_prompt), ("sim_r", codec_prompt)):
                if prompt:
                    result[field + suffix] = float(generated @ model.file_embedding(prompt, cache=True))
        if self.options.prompt_dnsmos and original_prompt:
            from .codec import read_audio

            scores = self._per_prompt("dnsmos", original_prompt,
                                      lambda p: self.dnsmos(read_audio(p, SAMPLE_RATE).numpy()))
            result.update({f"prompt_{k}": v for k, v in scores.items()})
        return result
