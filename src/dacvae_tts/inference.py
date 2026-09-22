import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch

from .codec import Codec, backend_options, check_compatibility, read_audio
from .contracts import normalization_stats, target_mask
from .model import sample
from .text import BYTE_OFFSET, normalize, tokenize
from .training import autocast, load_model


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
    ):
        self.asr_language = asr_language
        started = time.perf_counter()
        self.device = torch.device(device)
        self.precision = precision
        self.model, self.checkpoint = load_model(checkpoint, device)
        codec_options = dict(codec_options or {})
        if self.checkpoint["codec"].get("loudness_lufs") is not None:
            # References must receive the loudness normalization the training cache used.
            codec_options["loudness"] = self.checkpoint["codec"]["loudness_lufs"]
        self.codec = Codec(self.checkpoint["codec"]["checkpoint"], device, **codec_options)
        check_compatibility(self.codec.metadata, self.checkpoint["codec"])
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
        timings = {}
        started = time.perf_counter()
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
    ):
        """Use text + ref_audio; no speaker ID, enrollment table, or per-voice fine-tuning."""
        if (ref_audio is None) == (reference is None):
            raise ValueError("Provide exactly one of ref_audio or a prepared VoiceReference")
        started = time.perf_counter()
        reused = reference is not None
        reference = reference or self.prepare_reference(ref_audio, reference_text)
        batch = self.make_batch(reference.latents, reference.transcript, text, seconds, duration_scale)
        audio, _, metadata = self.generate(batch, steps, guidance, seed, sway)
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

    @torch.inference_mode()
    def make_batch(self, reference, reference_text, text, seconds=None, duration_scale=1.0):
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
            version = self.text_version
            reference_bytes = len(normalize(reference_text, version).encode("utf-8"))
            target_bytes = len(normalize(text, version).encode("utf-8"))
            frames = round(len(reference) / max(reference_bytes, 1) * target_bytes * duration_scale)
            self.duration_profile = {"duration_rule": "reference_frames_per_byte"}
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
        return {
            "prompt": prompt,
            "prompt_mask": prompt_mask,
            "valid": torch.ones_like(prompt_mask),
            "tokens": tokens,
            "segments": segments,
        }

    @torch.inference_mode()
    def generate(self, batch, steps=16, guidance=1.5, seed=0, sway=-1):
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
                lambda: sample(
                    self.model,
                    **batch,
                    steps=steps,
                    guidance=guidance,
                    seed=seed,
                    sway=sway,
                    stats=sampler_stats,
                    condition_cache=(text, text_valid, voice),
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
    )
    print(json.dumps(result.metadata, indent=2))
