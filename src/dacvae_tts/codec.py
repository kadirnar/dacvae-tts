import contextlib
import hashlib
import math
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from .contracts import normalization_stats

PREPROCESSING = "mono-mean_scipy-resample-poly_no-gain_no-trim_v1"


def backend_options(args):
    """Keep legacy callers and the reference backend's constructor compatible."""
    if (
        getattr(args, "codec_backend", "reference") == "reference"
        and not getattr(args, "codec_compile", False)
        and not getattr(args, "codec_graphs", False)
        and getattr(args, "codec_layout", "native") == "native"
    ):
        return {}
    return dict(
        backend=getattr(args, "codec_backend", "reference"),
        compile_model=getattr(args, "codec_compile", False),
        cuda_graphs=getattr(args, "codec_graphs", False),
        graph_max_shapes=getattr(args, "codec_graph_max_shapes", 4),
        layout=getattr(args, "codec_layout", "native"),
    )


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_compatibility(left, right):
    for key in ("checkpoint", "sample_rate", "hop_length", "latent_dim", "posterior"):
        if left[key] != right[key]:
            raise ValueError(f"Incompatible codec/cache: {key}")
    for key in ("weights_sha256", "preprocessing"):
        if key in left and key in right:
            if left[key] != right[key]:
                raise ValueError(f"Incompatible codec/cache: {key}")
        else:
            warnings.warn(
                f"Legacy metadata lacks {key}; exact codec/preprocessing identity cannot be verified",
                stacklevel=2,
            )


def preprocessing_tag(loudness=None):
    """Identity of the waveform preprocessing; caches and checkpoints must agree on it."""
    return PREPROCESSING if loudness is None else f"mono-mean_scipy-resample-poly_lufs{loudness:g}_no-trim_v2"


def normalize_loudness(audio, sample_rate, loudness):
    """BS.1770 integrated loudness, then peak protection: DACVAE's own `compress()` convention."""
    import pyloudnorm

    if len(audio) < int(0.4 * sample_rate):
        return audio  # shorter than one gating block: loudness is undefined
    measured = pyloudnorm.Meter(sample_rate).integrated_loudness(audio.astype(np.float64))
    if not np.isfinite(measured):
        return audio  # digital silence or fully gated; the RMS filter rejects it later
    audio = audio * np.float32(10.0 ** ((loudness - measured) / 20.0))
    peak = float(np.abs(audio).max())
    return audio / np.float32(peak) if peak > 1.0 else audio


def read_audio(path, sample_rate, loudness=None):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Empty or nonfinite audio: {path}")
    if sr != sample_rate:
        factor = math.gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // factor, sr // factor).astype(np.float32)
    if loudness is not None:
        audio = normalize_loudness(audio, sample_rate, loudness)
    return torch.from_numpy(audio.copy())


# "auto" pre-tanh gain: bring the 99.9th percentile of |tanh input| down to atanh(0.95) ~ 1.83, i.e. keep all but
# the loudest 0.1% of samples below 0.95 at the output instead of on the flat top of the tanh (0.999 = atanh 3.8).
PRE_TANH_CEILING = 0.95
PRE_TANH_QUANTILE = 0.999


def parse_pre_tanh_gain(value):
    """CLI/API form of the pre-tanh gain: a positive number, "auto" or "auto:<output ceiling in (0,1)>"."""
    if isinstance(value, str):
        text = value.strip().lower()
        try:
            value = text if text.startswith("auto") else float(text)
        except ValueError:
            raise ValueError(f"pre_tanh_gain must be a positive number, auto or auto:<ceiling>, not {value!r}") from None
    pre_tanh_mode(value)
    return value


def pre_tanh_mode(gain):
    """None | ("fixed", gain) | ("auto", output ceiling); raises on anything else."""
    if gain is None:
        return None
    if isinstance(gain, str):
        head, _, tail = gain.partition(":")
        if head == "auto":
            try:
                ceiling = float(tail) if tail else PRE_TANH_CEILING
            except ValueError:
                ceiling = float("nan")
            if not 0 < ceiling < 1:
                raise ValueError("The auto pre-tanh output ceiling must be a number in (0,1)")
            return "auto", ceiling
    elif isinstance(gain, (int, float)) and not isinstance(gain, bool) and math.isfinite(gain) and gain > 0:
        return "fixed", float(gain)
    raise ValueError(f"pre_tanh_gain must be None, a positive number, auto or auto:<ceiling>, not {gain!r}")


def output_tanh(decoder):
    """The decoder's output nn.Tanh, found by type (DACVAE: decoder.wm_model.encoder_block.pre[2]), never by path."""
    found = [module for module in decoder.modules() if isinstance(module, torch.nn.Tanh)]
    if not found:
        raise ValueError("The codec decoder has no nn.Tanh output nonlinearity; pre_tanh_gain is unsupported")
    if len(found) > 1:
        raise ValueError(f"The codec decoder has {len(found)} nn.Tanh modules; its output tanh is ambiguous")
    return found[0]


def robust_level(x, quantile=PRE_TANH_QUANTILE):
    """The `quantile` of |x| over all elements (kthvalue: no size limit, unlike torch.quantile)."""
    values = x.detach().abs().flatten().float()
    k = min(max(math.ceil(quantile * values.numel()), 1), values.numel())
    return float(values.kthvalue(k).values)


@contextlib.contextmanager
def pre_tanh_gain_hook(decoder, gain):
    """Scale the input of the decoder's output tanh by `gain` while the context is open; yields {"gain", "level"}.

    DACVAE's decoder ends in `Decoder.watermark`: output = tanh(conv(snake(x))) + alpha * h, where the tanh sits in
    `WatermarkEncoderBlock.pre` = [Snake, Conv(->1), Tanh, Conv(1->32)] (the output path drops the last conv via
    `forward_no_conv`) and the watermark residual h is computed from the same tanh output. The 0.999 "clipping" of
    high-guidance outputs (79/98/100/100% of files at g=3/4/5/6) is this tanh saturating: soft clipping baked into
    the waveform that post-decode loudness normalization cannot undo. Scaling the latents is no loudness control
    either (-18 dB at the codec input moved the latent norm by only 4%), so the gain acts on the tanh input.
    `audio.finalize` restores -16 LUFS afterwards. The tanh runs twice per decode (watermark branch first, then the
    output): the gain is fixed at the first call and reused, so both paths see the same scaled signal and the
    watermark stays consistent. "auto" sets g = min(1, atanh(ceiling) / robust level), the level being the 99.9th
    percentile of |tanh input| of the utterance, in the same single decoder pass; g = 1 leaves the output bit-exact.
    """
    mode, value = pre_tanh_mode(gain)
    module = output_tanh(decoder)
    applied = {}

    def scale(_module, args):
        if "gain" not in applied:
            if mode == "auto":
                level = robust_level(args[0])
                applied.update(level=level, gain=min(1.0, math.atanh(value) / level) if level > 0 else 1.0)
            else:
                applied["gain"] = value
        return (args[0] * applied["gain"], *args[1:])

    handle = module.register_forward_pre_hook(scale)
    try:
        yield applied
    finally:
        handle.remove()


class Codec:
    """Frozen DACVAE; model-specific details are discovered, never guessed."""

    def __init__(
        self,
        checkpoint="facebook/dacvae-watermarked",
        device="cpu",
        *,
        encoder_only=False,
        fold_weight_norm=False,
        backend="reference",
        compile_model=False,
        cuda_graphs=False,
        graph_max_shapes=4,
        graph_warmup=3,
        layout="native",
        loudness=None,
    ):
        from dacvae import DACVAE

        self.loudness = loudness

        self.device = torch.device(device)
        if backend not in {"reference", "fast"}:
            raise ValueError("Codec backend must be reference or fast")
        if (compile_model or cuda_graphs or layout != "native") and backend != "fast":
            raise ValueError("Codec layout/compile/graphs require --codec-backend fast")
        if cuda_graphs and self.device.type != "cuda":
            raise ValueError("Codec CUDA graphs require a CUDA device")
        self.backend, self._fast = backend, None
        self.runtime = dict(backend=backend, compile=compile_model, cuda_graphs=cuda_graphs, layout=layout)
        self.encoder_only = encoder_only
        path = Path(checkpoint)
        if not path.exists() and str(checkpoint).startswith("facebook/"):
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(repo_id=checkpoint, filename="weights.pth"))
        self.weights_sha256 = file_digest(path)
        self.model = DACVAE.load(str(path)).eval().requires_grad_(False)
        if encoder_only:
            # The decoder is not needed when building the training cache.
            self.model.encoder.to(self.device)
            self.model.quantizer.in_proj.to(self.device)
        else:
            self.model.to(self.device)
        if fold_weight_norm or backend == "fast":
            # Frozen weights: materialize the exact effective weight once, not every forward.
            components = (
                (self.model,)
                if backend == "fast" and not encoder_only
                else (self.model.encoder, self.model.quantizer.in_proj)
            )
            for component in components:
                for module in component.modules():
                    if hasattr(module, "weight_g") and hasattr(module, "weight_v"):
                        torch.nn.utils.remove_weight_norm(module)
            # remove_weight_norm registers fresh Parameters; keep those frozen too.
            self.model.requires_grad_(False)
        self.sample_rate = int(self.model.sample_rate)
        self.hop_length = int(self.model.hop_length)
        # Probe the posterior, not DAC's pre-bottleneck latent_dim attribute.
        with torch.inference_mode():
            z = self.encode(torch.zeros(self.sample_rate, device=self.device))
        self.latent_dim = z.size(-1)
        self.checkpoint = checkpoint
        if backend == "fast":
            from .fast_codec import SOURCE_REVISION, FastCodec

            self._fast = FastCodec(
                self.model, encoder_only, compile_model, cuda_graphs, graph_max_shapes, graph_warmup, layout
            )
            self.runtime["source_revision"] = SOURCE_REVISION

    @property
    def metadata(self):
        loudness = getattr(self, "loudness", None)
        result = {
            "checkpoint": self.checkpoint,
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "latent_dim": self.latent_dim,
            "posterior": "mean",
            "format_version": 1,
            "weights_sha256": self.weights_sha256,
            "preprocessing": preprocessing_tag(loudness),
            "codec_runtime": self.runtime,
        }
        if loudness is not None:
            result["loudness_lufs"] = loudness
        return result

    def _posterior_mean(self, audio, precision="fp32"):
        # Use the same explicit math for cache preparation and inference references.
        # TF32 kernels can drift substantially across batch sizes.
        with (
            torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled,
                benchmark=torch.backends.cudnn.benchmark,
                deterministic=torch.backends.cudnn.deterministic,
                allow_tf32=False,
            ),
            torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=precision == "bf16"),
        ):
            if getattr(self, "_fast", None) is not None:
                mean = self._fast.encode(audio)
            else:
                z = self.model.encoder(audio)
                mean, _ = self.model.quantizer.in_proj(z).chunk(2, dim=1)
        return mean

    @torch.inference_mode()
    def encode(self, audio):
        if audio.ndim != 1 or not audio.is_floating_point() or not torch.isfinite(audio).all():
            raise ValueError("Codec.encode requires one finite mono float waveform [samples]")
        audio = audio.to(self.device).reshape(1, 1, -1)
        if audio.size(-1) < self.hop_length * 2:
            raise ValueError("Audio too short for DACVAE")
        mean = self._posterior_mean(self.model._pad(audio))
        return mean[0].transpose(0, 1).contiguous().float()

    @torch.inference_mode()
    def encode_batch(self, audios, precision="fp32"):
        """Return CPU [B,L,C] posterior means, preserving single-utterance boundaries.

        Only equal hop-rounded lengths may share a batch. Padding arbitrary-length
        inputs to the longest waveform changes intermediate convolution boundaries,
        even if the output is subsequently trimmed. Reflect-pad each input first.
        """
        if not audios:
            raise ValueError("encode_batch requires at least one waveform")
        if precision not in {"fp32", "bf16"}:
            raise ValueError("Encoder precision must be fp32 or bf16")
        if precision == "bf16" and self.device.type != "cuda":
            raise ValueError("BF16 cache encoding requires CUDA")
        lengths = []
        padded = []
        for audio in audios:
            if audio.ndim != 1 or not audio.is_floating_point() or not torch.isfinite(audio).all():
                raise ValueError("encode_batch requires finite mono float waveforms [samples]")
            if len(audio) < 2 * self.hop_length:
                raise ValueError("Audio too short for DACVAE")
            lengths.append(math.ceil(len(audio) / self.hop_length))
            padded.append(self.model._pad(audio.reshape(1, 1, -1)))
        if len(set(lengths)) != 1:
            raise ValueError("encode_batch requires equal hop-rounded lengths; bucket inputs first")
        audio = torch.cat(padded, dim=0)
        if self.device.type == "cuda" and audio.device.type == "cpu":
            audio = audio.pin_memory()
        audio = audio.to(self.device, non_blocking=True)
        mean = self._posterior_mean(audio, precision)
        if mean.shape != (len(audios), self.latent_dim, lengths[0]):
            raise ValueError(f"Unexpected codec batch shape: {tuple(mean.shape)}")
        # One device-to-host transfer/synchronization for the whole batch.
        return mean.transpose(1, 2).contiguous().float().cpu()

    @torch.inference_mode()
    def decode(self, latents, pre_tanh_gain=None, stats=None):
        """Waveform [samples] of one utterance's latents [frames, C].

        `pre_tanh_gain` (None, a positive number, "auto" or "auto:<output ceiling>") scales the input of the decoder's
        output tanh against saturation; see `pre_tanh_gain_hook`. Both backends support it (FastCodec.decode runs the
        original watermark head). None leaves the decoder untouched. `stats`, if a dict, receives the gain used
        (`pre_tanh_gain`), the requested mode and, for "auto", the measured tanh-input level.
        """
        if getattr(self, "encoder_only", False):
            raise ValueError("This codec was loaded encoder-only; reload with encoder_only=False to decode")
        if latents.ndim != 2 or latents.size(1) != self.latent_dim or latents.size(0) < 1:
            raise ValueError(f"Codec.decode requires [frames,C={self.latent_dim}]")
        if not torch.isfinite(latents).all():
            raise ValueError("Cannot decode nonfinite latents")
        inputs = latents.to(self.device).T[None].contiguous()
        gain = (
            contextlib.nullcontext() if pre_tanh_gain is None else pre_tanh_gain_hook(self.model.decoder, pre_tanh_gain)
        )
        with (
            torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled,
                benchmark=torch.backends.cudnn.benchmark,
                deterministic=torch.backends.cudnn.deterministic,
                allow_tf32=False,
            ),
            torch.autocast(self.device.type, enabled=False),
            gain as applied,
        ):
            waveform = (
                self._fast.decode(inputs.float(), self.model)
                if getattr(self, "_fast", None)
                else self.model.decode(inputs.float())
            )
        if pre_tanh_gain is not None:
            if "gain" not in applied:
                raise RuntimeError("The decoder never reached its output tanh; pre_tanh_gain had no effect")
            if stats is not None:
                stats.update(pre_tanh_gain=applied["gain"], pre_tanh_mode=str(pre_tanh_gain),
                             pre_tanh_level=applied.get("level"))
        return waveform[0, 0].float().cpu()

    @torch.inference_mode()
    def reconstruct(self, audio, mean, std, cache_precision="float16"):
        normalization_stats(mean, std, self.latent_dim)
        z = self.encode(audio)
        if cache_precision == "float16":
            z = z.half().float()
        elif cache_precision != "float32":
            raise ValueError("cache_precision must be float16 or float32")
        mean, std = mean.to(z.device), std.to(z.device)
        restored = ((z - mean) / std) * std + mean
        output = self.decode(restored)
        if output.numel() < audio.numel():
            raise ValueError("Codec returned fewer samples than the original waveform")
        return output[: audio.numel()], {
            "frames": len(z),
            "input_samples": audio.numel(),
            "decoded_samples": output.numel(),
            "trimmed_samples": output.numel() - audio.numel(),
            "normalization_roundtrip_max_error": (restored - z).abs().max().item(),
        }
