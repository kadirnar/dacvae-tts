import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch

from .codec import Codec, backend_options, check_compatibility, normalize_loudness, read_audio
from .contracts import normalization_stats, target_mask
from .duration import DurationPredictor, clamp_scale, rule_frames
from .model import sample, text_only_rows
from .text import BYTE_OFFSET, normalize, tokenize
from .training import autocast, load_model

DURATION_MODES = ("rule", "clamp", "syllable", "predictor")
SAMPLER_OPTIONS = ("guidance_until", "guidance_from", "noise_scale", "cfg_rescale", "apg_eta", "apg_norm",
                   "apg_momentum", "speaker_guidance")


@dataclass
class VoiceReference:
    latents: torch.Tensor
    transcript: str
    transcript_source: str
    timings: dict


@dataclass
class SynthesisResult:
    audio: torch.Tensor
    sample_rate: int
    metadata: dict


class Synthesizer:
    def __init__(
        self,
        checkpoint,
        device="cuda",
        precision="bf16",
        compile_model=False,
        asr_model="small.en",
        asr_device="cpu",
        profile=False,
        codec_options=None,
        asr_language="en",
        codec=None,
        duration_model=None,
    ):
        """`codec`: an already loaded Codec to share between checkpoints (same DACVAE weights and loudness)."""
        self.asr_language = asr_language
        started = time.perf_counter()
        self.device = torch.device(device)
        self.precision = precision
        self.model, self.checkpoint = load_model(checkpoint, device)
        loudness = self.checkpoint["codec"].get("loudness_lufs")
        if codec is None:
            codec_options = dict(codec_options or {})
            if loudness is not None:
                # References must receive the loudness normalization the training cache used.
                codec_options["loudness"] = loudness
            codec = Codec(self.checkpoint["codec"]["checkpoint"], device, **codec_options)
        elif getattr(codec, "loudness", None) != loudness:
            raise ValueError("The shared codec normalizes reference loudness differently from this checkpoint's cache")
        self.codec = codec
        check_compatibility(self.codec.metadata, self.checkpoint["codec"])
        self.duration_model = duration_model
        self.mean = self.checkpoint["mean"].to(device)
        self.std = self.checkpoint["std"].to(device)
        normalization_stats(self.mean, self.std, self.codec.latent_dim)
        self.asr_model, self.asr_device, self._asr = asr_model, asr_device, None
        self.profile = profile
        self.text_version = self.checkpoint["codec"].get("text_normalization", "unicode-v1")
        self.duration_profile = {}
        if compile_model:
            self.model.forward = torch.compile(self.model.forward, dynamic=True)
        self._sync()
        self.load_seconds = time.perf_counter() - started

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _measure(self, function, timings, name):
        if not self.profile:
            return function()
        self._sync()
        started = time.perf_counter()
        result = function()
        self._sync()
        timings[name] = time.perf_counter() - started
        return result

    def transcribe_reference(self, path):
        """Optional ASR convenience layer; the baseline TTS still consumes a transcript."""
        if self._asr is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ImportError(
                    "Audio-only requests need the optional ASR dependency: pip install -e '.[asr]'"
                ) from exc
            self._asr = WhisperModel(
                self.asr_model,
                device=self.asr_device,
                compute_type="float16" if self.asr_device == "cuda" else "int8",
            )
        audio = read_audio(path, 16000)
        segments, _ = self._asr.transcribe(
            audio.numpy(),
            language=self.asr_language,
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        if not transcript:
            raise ValueError("Reference ASR returned no text; use clear speech or supply reference_text")
        return transcript

    def prepare_reference(self, ref_audio, reference_text=None):
        """`ref_audio`: a file path, or a (waveform, sample_rate) pair of an already loaded mono recording."""
        timings = {}
        started = time.perf_counter()
        if isinstance(ref_audio, tuple):
            latents = self.encode_reference(*ref_audio)
        else:
            latents = self.reference(ref_audio)
        self._sync()
        timings["reference_encode_seconds"] = time.perf_counter() - started
        source = "provided"
        if reference_text is None:
            started = time.perf_counter()
            reference_text = self.transcribe_reference(ref_audio)
            timings["reference_asr_seconds"] = time.perf_counter() - started
            source = f"asr:{self.asr_model}"
        if not reference_text.strip():
            raise ValueError("A nonempty reference transcript is required internally")
        return VoiceReference(latents, reference_text, source, timings)

    def synthesize(
        self,
        text,
        ref_audio=None,
        *,
        reference=None,
        reference_text=None,
        output=None,
        seconds=None,
        duration_scale=1.0,
        steps=16,
        guidance=1.5,
        seed=42,
        sway=-1.0,
        guidance_until=1.0,
        noise_scale=1.0,
        duration_mode="rule",
        **sampler,
    ):
        """Use text + ref_audio; no speaker ID, enrollment table, or per-voice fine-tuning.

        `sampler` takes the further options of `model.sample` (guidance_from, cfg_rescale, apg_eta, apg_norm,
        apg_momentum, speaker_guidance); `duration_mode` is one of rule, clamp, syllable, predictor.
        """
        if (ref_audio is None) == (reference is None):
            raise ValueError("Provide exactly one of ref_audio or a prepared VoiceReference")
        unknown = set(sampler) - set(SAMPLER_OPTIONS)
        if unknown:
            raise TypeError(f"Unknown sampler options: {sorted(unknown)}")
        started = time.perf_counter()
        reused = reference is not None
        reference = reference or self.prepare_reference(ref_audio, reference_text)
        batch = self.make_batch(reference.latents, reference.transcript, text, seconds, duration_scale, duration_mode)
        audio, _, metadata = self.generate(
            batch, steps, guidance, seed, sway, guidance_until, noise_scale, **sampler
        )
        self._sync()
        metadata.update(
            request_seconds=time.perf_counter() - started,
            model_load_seconds=self.load_seconds,
            reference_transcript=reference.transcript,
            reference_transcript_source=reference.transcript_source,
            reference_reused=reused,
            reference_preparation=reference.timings,
            text_normalization=self.text_version,
            **self.duration_profile,
        )
        if output is not None:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, audio.numpy(), self.codec.sample_rate, subtype="FLOAT")
            path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        return SynthesisResult(audio, self.codec.sample_rate, metadata)

    def reference(self, path):
        audio = read_audio(path, self.codec.sample_rate, getattr(self.codec, "loudness", None))
        seconds = len(audio) / self.codec.sample_rate
        if not 0.5 <= seconds <= 30:
            raise ValueError("Reference must be a complete .5–30 second utterance with an exact transcript")
        return (self.codec.encode(audio) - self.mean) / self.std

    def encode_reference(self, audio, sample_rate):
        """Normalized latents of a mono waveform (numpy or tensor) at any sample rate; same preprocessing as files."""
        import numpy as np
        from scipy.signal import resample_poly

        audio = np.asarray(audio.cpu() if torch.is_tensor(audio) else audio, dtype=np.float32).reshape(-1)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("Empty or nonfinite reference audio")
        if sample_rate != self.codec.sample_rate:
            factor = math.gcd(int(sample_rate), self.codec.sample_rate)
            audio = resample_poly(audio, self.codec.sample_rate // factor, int(sample_rate) // factor).astype(np.float32)
        if getattr(self.codec, "loudness", None) is not None:
            audio = normalize_loudness(audio, self.codec.sample_rate, self.codec.loudness)
        seconds = len(audio) / self.codec.sample_rate
        if not 0.5 <= seconds <= 30:
            raise ValueError("Reference must be a complete .5–30 second utterance with an exact transcript")
        return (self.codec.encode(torch.from_numpy(audio.copy())) - self.mean) / self.std

    def target_frames(self, reference_frames, reference_text, text, seconds=None, duration_scale=1.0, duration_mode="rule"):
        """Number of target latent frames for `text` spoken in the voice of a prompt of `reference_frames` frames."""
        if not math.isfinite(duration_scale) or duration_scale <= 0:
            raise ValueError("duration_scale must be finite and positive")
        if seconds is not None:
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("seconds must be finite and positive")
            profile = {"duration_override_seconds": seconds}
            return round(seconds * self.codec.sample_rate / self.codec.hop_length * duration_scale), profile
        if duration_mode not in DURATION_MODES:
            raise ValueError(f"duration_mode must be one of {DURATION_MODES}")
        version = self.text_version
        reference_text, text = normalize(reference_text, version), normalize(text, version)
        profile = {"duration_rule": "reference_frames_per_byte", "duration_mode": duration_mode}
        if duration_mode == "syllable":
            frames = rule_frames(reference_frames, reference_text, text, "syllables")
        elif duration_mode == "predictor":
            if self.duration_model is None or isinstance(self.duration_model, (str, Path)):
                self.duration_model = DurationPredictor.load(self.duration_model)
            frames = self.duration_model.predict(reference_frames, reference_text, text)
        else:
            frames = rule_frames(reference_frames, reference_text, text, "bytes")
            if duration_mode == "clamp":
                factor = clamp_scale(reference_frames, reference_text)
                frames *= factor
                profile["duration_clamp_factor"] = factor
        return round(frames * duration_scale), profile

    @torch.inference_mode()
    def make_batch(self, reference, reference_text, text, seconds=None, duration_scale=1.0, duration_mode="rule"):
        if not math.isfinite(duration_scale) or duration_scale <= 0:
            raise ValueError("duration_scale must be finite and positive")
        reference = reference.to(self.device)
        layout = self.model.cfg.text_layout
        tokens, segments = tokenize(reference_text, text, version=self.text_version, layout=layout)
        tokens, segments = tokens[None].to(self.device), segments[None].to(self.device)
        if tokens.numel() > 2048:
            raise ValueError("Text too long; split into sentences before synthesis")
        if seconds is None and self.model.duration is None:
            # Speaking-rate rule (F5-TTS): the target keeps the prompt's frames per transcript byte.
            # It also keeps length-normalized text/audio positions consistent across the boundary.
            frames, self.duration_profile = self.target_frames(
                len(reference), reference_text, text, None, duration_scale, duration_mode
            )
        elif seconds is None:
            prompt = reference[None]
            mask = torch.ones(prompt.shape[:2], device=self.device, dtype=torch.bool)
            with autocast(self.device, self.precision):
                self.duration_profile = {}
                log_rate = self._measure(
                    lambda: self.model.predict_duration(prompt, mask, tokens, segments),
                    self.duration_profile,
                    "duration_prediction_seconds",
                )
            nbytes = ((segments == 1) & (tokens >= BYTE_OFFSET)).sum().item()
            frames = round(math.exp(float(log_rate.clamp(-4, 6))) * nbytes * duration_scale)
        else:
            self.duration_profile = {"duration_override_seconds": seconds}
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("seconds must be finite and positive")
            frames = round(seconds * self.codec.sample_rate / self.codec.hop_length * duration_scale)
        seconds = frames * self.codec.hop_length / self.codec.sample_rate
        if not 0.25 <= seconds <= 30:
            raise ValueError(
                f"Predicted/requested target duration {seconds:.2f}s is outside .25–30s; set --seconds"
            )
        prompt = torch.zeros(1, len(reference) + frames, self.codec.latent_dim, device=self.device)
        prompt[:, : len(reference)] = reference
        prompt_mask = torch.arange(prompt.size(1), device=self.device)[None] < len(reference)
        only_tokens, only_segments = tokenize("", text, version=self.text_version, layout=layout)
        return {
            "prompt": prompt,
            "prompt_mask": prompt_mask,
            "valid": torch.ones_like(prompt_mask),
            "tokens": tokens,
            "segments": segments,
            "text_only_tokens": only_tokens[None].to(self.device),
            "text_only_segments": only_segments[None].to(self.device),
        }

    def _sample(self, batch, steps, guidance, seed, sway, stats, condition_cache=None, **sampler):
        """model.sample on a prepared batch; builds the prompt-free branch when speaker guidance is requested."""
        core = {k: batch[k] for k in ("prompt", "prompt_mask", "valid", "tokens", "segments")}
        if sampler.get("speaker_guidance") is not None:
            sampler["text_only"] = text_only_rows(
                self.model, batch["valid"], batch["prompt_mask"], batch["text_only_tokens"], batch["text_only_segments"]
            )
        return sample(
            self.model, **core, steps=steps, guidance=guidance, seed=seed, sway=sway, stats=stats,
            condition_cache=condition_cache, **sampler,
        )

    @torch.inference_mode()
    def generate(self, batch, steps=16, guidance=1.5, seed=0, sway=-1, guidance_until=1.0, noise_scale=1.0, **sampler):
        if batch["prompt"].size(0) != 1:
            raise ValueError(
                "Waveform generation currently accepts a single request; sample() supports batches"
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        stages, sampler_stats = {}, {}
        with autocast(self.device, self.precision):
            text, text_valid = self._measure(
                lambda: self.model.text(batch["tokens"], batch["segments"]), stages, "text_encoding_seconds"
            )
            voice = self._measure(
                lambda: self.model.reference_summary(batch["prompt"], batch["prompt_mask"]),
                stages,
                "reference_summary_seconds",
            )
            result = self._measure(
                lambda: self._sample(
                    batch, steps, guidance, seed, sway, sampler_stats, (text, text_valid, voice),
                    guidance_until=guidance_until, noise_scale=noise_scale, **sampler,
                ),
                stages,
                "iterative_generation_seconds",
            )
        target = result[0, target_mask(batch["valid"], batch["prompt_mask"])[0]]
        # Decode target alone: no reference audio leaks into the output waveform.
        audio = self._measure(
            lambda: self.codec.decode(target * self.std + self.mean), stages, "waveform_decode_seconds"
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        duration = audio.numel() / self.codec.sample_rate
        return (
            audio,
            target.cpu(),
            {
                "generation_seconds": elapsed,
                "audio_seconds": duration,
                "rtf": elapsed / max(duration, 1e-6),
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda"
                else 0,
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
                "sway": sway,
                "codec_runtime": getattr(self.codec, "runtime", {"backend": "reference"}),
                "profiled_stages": stages,
                **sampler_stats,
            },
        )


    @torch.inference_mode()
    def synthesize_many(
        self,
        texts,
        reference,
        *,
        candidates=1,
        seconds=None,
        duration_scale=1.0,
        duration_mode="rule",
        steps=32,
        guidance=5.0,
        seed=42,
        sway=-1.0,
        max_rows=16,
        **sampler,
    ):
        """Generate every text (e.g. the sentence chunks of a long input) `candidates` times in padded batches.

        All rows share the prepared `reference` (a VoiceReference). Returns (results, metadata) where results[i][k]
        is a dict with the float32 waveform (`audio`, numpy) of candidate k of text i and its duration. Candidates
        of a text differ only in their initial noise, which is what best-of-N reranking needs.
        """
        import numpy as np

        unknown = set(sampler) - set(SAMPLER_OPTIONS)
        if unknown:
            raise TypeError(f"Unknown sampler options: {sorted(unknown)}")
        if not texts or candidates < 1 or max_rows < 1:
            raise ValueError("Need at least one text, one candidate and a positive batch size")
        started = time.perf_counter()
        latents = reference.latents.to(self.device)
        layout = self.model.cfg.text_layout
        requests = []
        for text in texts:
            frames, profile = self.target_frames(len(latents), reference.transcript, text, seconds, duration_scale,
                                                 duration_mode)
            if not 0.25 <= frames * self.codec.hop_length / self.codec.sample_rate <= 30:
                raise ValueError(f"Target duration outside .25–30 s for: {text[:60]!r}; split the text")
            tokens, segments = tokenize(reference.transcript, text, version=self.text_version, layout=layout)
            only_tokens, only_segments = tokenize("", text, version=self.text_version, layout=layout)
            if tokens.numel() > 2048:
                raise ValueError("Text too long; split into sentences before synthesis")
            requests.append((frames, tokens, segments, only_tokens, only_segments, profile))
        rows = [(i, k) for i in range(len(texts)) for k in range(candidates)]
        results = [[None] * candidates for _ in texts]
        stats, generation = {}, 0.0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        pad = torch.nn.utils.rnn.pad_sequence
        for start in range(0, len(rows), max_rows):
            chunk = rows[start : start + max_rows]
            reference_frames = len(latents)
            totals = [reference_frames + requests[i][0] for i, _ in chunk]
            width = max(totals)
            prompt = torch.zeros(len(chunk), width, self.codec.latent_dim, device=self.device)
            prompt[:, :reference_frames] = latents
            positions = torch.arange(width, device=self.device)[None]
            batch = {
                "prompt": prompt,
                "prompt_mask": (positions < reference_frames).expand(len(chunk), -1).clone(),
                "valid": positions < torch.tensor(totals, device=self.device)[:, None],
                "tokens": pad([requests[i][1] for i, _ in chunk], batch_first=True).to(self.device),
                "segments": pad([requests[i][2] for i, _ in chunk], batch_first=True).to(self.device),
                "text_only_tokens": pad([requests[i][3] for i, _ in chunk], batch_first=True).to(self.device),
                "text_only_segments": pad([requests[i][4] for i, _ in chunk], batch_first=True).to(self.device),
            }
            tick = time.perf_counter()
            with autocast(self.device, self.precision):
                result = self._sample(batch, steps, guidance, seed + start, sway, stats, **sampler)
            self._sync()
            generation += time.perf_counter() - tick
            for row, (i, k) in enumerate(chunk):
                target = result[row, reference_frames : totals[row]]
                audio = self.codec.decode(target.float() * self.std + self.mean)
                results[i][k] = {
                    "audio": audio.numpy().astype(np.float32),
                    "audio_seconds": audio.numel() / self.codec.sample_rate,
                    "frames": int(requests[i][0]),
                    "duration": requests[i][5],
                }
        self._sync()
        metadata = {
            "texts": len(texts),
            "candidates": candidates,
            "rows": len(rows),
            "generation_seconds": generation,
            "request_seconds": time.perf_counter() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "sway": sway,
            "duration_mode": duration_mode,
            "duration_scale": duration_scale,
            **{k: v for k, v in stats.items() if k != "time_grid"},
        }
        return results, metadata


def infer(args):
    tts = Synthesizer(
        args.checkpoint,
        args.device,
        args.precision,
        args.compile,
        args.asr_model,
        args.asr_device,
        args.profile,
        codec_options=backend_options(args),
        asr_language=getattr(args, "asr_language", "en"),
    )
    result = tts.synthesize(
        args.text,
        args.reference,
        reference_text=args.reference_text,
        output=args.output,
        seconds=args.seconds,
        duration_scale=args.duration_scale,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        sway=args.sway,
        guidance_until=getattr(args, "guidance_until", 1.0),
        noise_scale=getattr(args, "noise_scale", 1.0),
        duration_mode=getattr(args, "duration_mode", "rule"),
        guidance_from=getattr(args, "guidance_from", 0.0),
        cfg_rescale=getattr(args, "cfg_rescale", 0.0),
        apg_eta=getattr(args, "apg_eta", 1.0),
        apg_norm=getattr(args, "apg_norm", 0.0),
        apg_momentum=getattr(args, "apg_momentum", 0.0),
        speaker_guidance=getattr(args, "speaker_guidance", None),
    )
    print(json.dumps(result.metadata, indent=2))
