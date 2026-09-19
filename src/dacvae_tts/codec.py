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


def read_audio(path, sample_rate):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Empty or nonfinite audio: {path}")
    if sr != sample_rate:
        factor = math.gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // factor, sr // factor).astype(np.float32)
    return torch.from_numpy(audio.copy())


class Codec:
    """Frozen DACVAE; model-specific details are discovered, never guessed."""

    def __init__(self, checkpoint="facebook/dacvae-watermarked", device="cpu"):
        from dacvae import DACVAE

        self.device = torch.device(device)
        path = Path(checkpoint)
        if not path.exists() and str(checkpoint).startswith("facebook/"):
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(repo_id=checkpoint, filename="weights.pth"))
        self.weights_sha256 = file_digest(path)
        self.model = DACVAE.load(str(path)).to(self.device).eval().requires_grad_(False)
        self.sample_rate = int(self.model.sample_rate)
        self.hop_length = int(self.model.hop_length)
        # Probe the posterior, not DAC's pre-bottleneck latent_dim attribute.
        with torch.inference_mode():
            z = self.encode(torch.zeros(self.sample_rate, device=self.device))
        self.latent_dim = z.size(-1)
        self.checkpoint = checkpoint

    @property
    def metadata(self):
        return {
            "checkpoint": self.checkpoint,
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "latent_dim": self.latent_dim,
            "posterior": "mean",
            "format_version": 1,
            "weights_sha256": self.weights_sha256,
            "preprocessing": PREPROCESSING,
        }

    @torch.inference_mode()
    def encode(self, audio):
        if audio.ndim != 1 or not audio.is_floating_point() or not torch.isfinite(audio).all():
            raise ValueError("Codec.encode requires one finite mono float waveform [samples]")
        audio = audio.to(self.device).reshape(1, 1, -1)
        if audio.size(-1) < self.hop_length * 2:
            raise ValueError("Audio too short for DACVAE")
        z = self.model.encoder(self.model._pad(audio))
        mean, _ = self.model.quantizer.in_proj(z).chunk(2, dim=1)
        return mean[0].transpose(0, 1).contiguous().float()

    @torch.inference_mode()
    def decode(self, latents):
        if latents.ndim != 2 or latents.size(1) != self.latent_dim or latents.size(0) < 1:
            raise ValueError(f"Codec.decode requires [frames,C={self.latent_dim}]")
        if not torch.isfinite(latents).all():
            raise ValueError("Cannot decode nonfinite latents")
        waveform = self.model.decode(latents.to(self.device).T[None].contiguous())
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
