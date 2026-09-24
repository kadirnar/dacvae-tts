"""SIM-o: the seed-tts-eval speaker similarity (WavLM-Large + ECAPA-TDNN fine-tuned for speaker verification).

Why: `microsoft/wavlm-base-plus-sv` (the repository's `speaker_similarity`, kept as the development metric) gives
0.89-0.96 to the same speaker and still 0.60-0.84 to different speakers, so 0.94-0.95 sits at its ceiling and does
not separate systems. Published zero-shot TTS numbers (Seed-TTS, F5-TTS, ZipVoice, CosyVoice) use UniSpeech's
`ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large")` with `wavlm_large_finetune.pth`: ~0.60-0.69 for the
same speaker and about 0 for different speakers (VoxCeleb1 clips shipped with UniSpeech).

Protocol, as seed-tts-eval `verification.py` / F5-TTS `run_sim`: first channel, resampled to 16 kHz with torchaudio
only when needed, no trimming or loudness normalization, the whole clip; the score is the cosine between the
256-d embeddings of the generated audio (without prompt frames: `Synthesizer` decodes the target alone) and of the
prompt. Against the ORIGINAL prompt recording this is SIM-o; against a codec-resynthesized prompt it is SIM-r.
Callers keep the two apart in their field names (`sim_o` / `sim_r`).

Checkpoint: the original Google Drive/Azure links are dead; the verbatim Hugging Face mirror
`bezzam/wavlm_large_finetune_seed_tts_eval` is pinned by revision and its sha256 is checked before loading. It
holds 711 tensors: `feature_weight`, the 488 WavLM-Large backbone tensors (`feature_extract.model.*`, original
unilm/s3prl names), the ECAPA head and the unused training classifier `loss_calculator.projection.weight`.
Loading requires >= 700 matched tensors, no missing model tensor, and only `loss_calculator*` left over.

Backbones (the fine-tuned checkpoint overwrites every backbone tensor, so no pre-trained weights are used):
- "transformers" (default): transformers' `WavLMModel` with WavLM-Large's architecture and the checkpoint's names
  remapped one-to-one (`wavlm_to_transformers`); called like s3prl's upstream (per-waveform layer norm since the
  s3prl `wavlm_large` config has `normalize=True`; the 25 hidden states are the 24 layer inputs plus the final
  layer-normed output). No extra dependency: transformers is already required.
- "s3prl": the upstream code path, `torch.hub.load("s3prl/s3prl", "wavlm_large")`; needs network access and
  s3prl's dependencies (`pip install s3prl`). Use it to cross-check the default backend on a few clips.

The ECAPA head is vendored in `dacvae_tts.ecapa_tdnn` (CC BY-SA 3.0, see third_party/unispeech/LICENSE).
"""

import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from .codec import file_digest

REPO = "bezzam/wavlm_large_finetune_seed_tts_eval"
REVISION = "f127f699a44abb1c6e05f03522bd1b8c48be3184"
FILENAME = "wavlm_large_finetune.pth"
SHA256 = "51f07e3b94d9e0262a6a675ef5a087be3dd09e8c62e9d886827f44f82fe7f94b"
SIZE = 1_301_926_579
MIN_MATCHED_KEYS = 700
SAMPLE_RATE = 16000
BACKBONES = ("transformers", "s3prl")
BACKBONE_PREFIX = "feature_extract.model."

# Architecture of microsoft/wavlm-large (== s3prl `wavlm_large` cfg). Written out so building the model needs no
# download; dropouts/layerdrop are zero because the module is only ever evaluated. mask_time_prob > 0 keeps the
# `masked_spec_embed` parameter that receives the checkpoint's `mask_emb` (unused at inference).
WAVLM_LARGE = dict(
    conv_dim=[512] * 7, conv_kernel=[10, 3, 3, 3, 3, 2, 2], conv_stride=[5, 2, 2, 2, 2, 2, 2], conv_bias=False,
    feat_extract_norm="layer", feat_extract_activation="gelu", do_stable_layer_norm=True, hidden_size=1024,
    intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16, hidden_act="gelu", layer_norm_eps=1e-5,
    num_conv_pos_embeddings=128, num_conv_pos_embedding_groups=16, num_buckets=320, max_bucket_distance=800,
    hidden_dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, feat_proj_dropout=0.0, final_dropout=0.0,
    layerdrop=0.0, mask_time_prob=0.075, apply_spec_augment=True,
)

_LAYER_RENAMES = (
    ("self_attn.relative_attention_bias.", "attention.rel_attn_embed."),
    ("self_attn.grep_linear.", "attention.gru_rel_pos_linear."),
    ("self_attn.grep_a", "attention.gru_rel_pos_const"),
    ("self_attn_layer_norm.", "layer_norm."),
    ("self_attn.", "attention."),
    ("fc1.", "feed_forward.intermediate_dense."),
    ("fc2.", "feed_forward.output_dense."),
    ("final_layer_norm.", "final_layer_norm."),
)


def resolve_checkpoint(path=None, verify=True, expected_sha256=SHA256):
    """Local checkpoint path (downloaded from the pinned mirror when `path` is None), sha256-verified."""
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(REPO, FILENAME, revision=REVISION)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SIM-o checkpoint not found: {path}")
    if verify:
        digest = file_digest(path)
        if digest != expected_sha256:
            raise ValueError(
                f"SIM-o checkpoint sha256 mismatch for {path}: {digest} != {expected_sha256} "
                f"(expected {REPO}/{FILENAME}, {SIZE} bytes)"
            )
    return path


def load_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{path}: expected a dict with a non-empty 'model' state dict")
    return state


def load_verified(model, state, min_matched=MIN_MATCHED_KEYS):
    """Load `state` into `model` non-strictly, but fail unless the layout is the expected one.

    Upstream loads with strict=False and no checks, which would silently score with random weights if names
    drifted. Returns the number of matched tensors.
    """
    matched = set(state) & set(model.state_dict())
    if len(matched) < min_matched:
        raise ValueError(f"SIM-o checkpoint matches only {len(matched)} model tensors (need >= {min_matched})")
    result = model.load_state_dict(state, strict=False)
    stray = sorted(k for k in result.unexpected_keys if not k.startswith("loss_calculator"))
    if stray:
        raise ValueError(f"Unexpected SIM-o checkpoint tensors: {stray[:5]} ({len(stray)} total)")
    if result.missing_keys:
        raise ValueError(f"SIM-o model tensors missing from the checkpoint: {sorted(result.missing_keys)[:5]} "
                         f"({len(result.missing_keys)} total)")
    return len(matched)


def wavlm_to_transformers(name, target_names=()):
    """Original (unilm/s3prl) WavLM parameter name -> transformers `WavLMModel` parameter name.

    `target_names` (the transformers model's state-dict keys) chooses between the legacy weight-norm names
    (`weight_g`/`weight_v`) and torch's parametrization names (`original0`/`original1`) for the positional conv.
    """
    if name == "mask_emb":
        return "masked_spec_embed"
    match = re.fullmatch(r"feature_extractor\.conv_layers\.(\d+)\.(0|2\.1)\.(weight|bias)", name)
    if match:
        part = "conv" if match[2] == "0" else "layer_norm"
        return f"feature_extractor.conv_layers.{match[1]}.{part}.{match[3]}"
    if name.startswith("layer_norm."):
        return "feature_projection." + name
    if name.startswith("post_extract_proj."):
        return "feature_projection.projection." + name.split(".", 1)[1]
    if name.startswith("encoder.pos_conv.0."):
        suffix = name.rsplit(".", 1)[1]
        base = "encoder.pos_conv_embed.conv."
        if suffix == "bias":
            return base + "bias"
        parametrized = base + "parametrizations.weight." + {"weight_g": "original0", "weight_v": "original1"}[suffix]
        return parametrized if parametrized in set(target_names) else base + suffix
    if name.startswith("encoder.layer_norm."):
        return name
    match = re.fullmatch(r"encoder\.layers\.(\d+)\.(.+)", name)
    if match:
        for old, new in _LAYER_RENAMES:
            if match[2].startswith(old):
                return f"encoder.layers.{match[1]}.{new}{match[2][len(old):]}"
    raise KeyError(f"Unmapped WavLM parameter: {name}")


def remap_state(state, model):
    """Rename the checkpoint's backbone tensors for the transformers backbone; the head keys are unchanged."""
    targets = [k[len(BACKBONE_PREFIX):] for k in model.state_dict() if k.startswith(BACKBONE_PREFIX)]
    remapped = {}
    for key, value in state.items():
        if key.startswith(BACKBONE_PREFIX):
            key = BACKBONE_PREFIX + wavlm_to_transformers(key[len(BACKBONE_PREFIX):], targets)
        if key in remapped:
            raise ValueError(f"Two checkpoint tensors map to {key}")
        remapped[key] = value
    return remapped


class TransformersWavLM(torch.nn.Module):
    """WavLM-Large as transformers' `WavLMModel`, called like s3prl's `wavlm_large` upstream."""

    def __init__(self, normalize=True):
        super().__init__()
        try:
            from transformers import WavLMConfig, WavLMModel
        except ImportError as error:
            raise RuntimeError("SIM-o backend 'transformers' needs the transformers package") from error
        self.model = WavLMModel(WavLMConfig(**WAVLM_LARGE))
        self.normalize = normalize

    def forward(self, wavs):
        if len({len(w) for w in wavs}) != 1:
            raise ValueError("TransformersWavLM scores equal-length waveforms only (no padding mask)")
        if self.normalize:  # s3prl: wavs = [F.layer_norm(wav, wav.shape) for wav in wavs] when cfg.normalize
            wavs = [F.layer_norm(w, w.shape) for w in wavs]
        output = self.model(torch.stack(wavs), output_hidden_states=True)
        return {"hidden_states": list(output.hidden_states)}


def s3prl_wavlm_large():
    """The upstream backbone: s3prl's WavLM-Large through torch.hub (network + s3prl dependencies)."""
    try:
        backbone = torch.hub.load("s3prl/s3prl", "wavlm_large")
    except Exception as error:  # hub clone, missing s3prl dependency, download failure
        raise RuntimeError(
            "SIM-o backend 's3prl' could not load torch.hub('s3prl/s3prl', 'wavlm_large'); install s3prl "
            "(pip install s3prl) and allow network access, or use the default 'transformers' backend"
        ) from error
    layers = backbone.model.encoder.layers
    for index in (23, 11):  # as UniSpeech's ECAPA_TDNN does for 24-layer extractors
        if len(layers) == 24 and hasattr(layers[index].self_attn, "fp32_attention"):
            layers[index].self_attn.fp32_attention = False
    return backbone


def load_audio_16k(path):
    """seed-tts-eval input: first channel, torchaudio resampling to 16 kHz only when needed, nothing else."""
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    wave = torch.from_numpy(np.ascontiguousarray(audio[:, 0]))
    if not len(wave) or not torch.isfinite(wave).all():
        raise ValueError(f"Empty or nonfinite audio: {path}")
    if rate != SAMPLE_RATE:
        import torchaudio.functional as AF

        wave = AF.resample(wave, rate, SAMPLE_RATE)
    return wave


class SimO:
    """WavLM-Large ECAPA-TDNN speaker embeddings; `similarity(a, b)` is the seed-tts-eval SIM of two files.

    `backbone` (a module with the s3prl upstream interface) overrides `backend`; tests use it with a tiny stub.
    """

    def __init__(self, checkpoint=None, backend="transformers", device="cpu", *, backbone=None,
                 verify_sha256=True, expected_sha256=SHA256, min_matched=MIN_MATCHED_KEYS):
        if backend not in BACKBONES:
            raise ValueError(f"Unknown SIM-o backend {backend!r}; choose from {BACKBONES}")
        from .ecapa_tdnn import ECAPA_TDNN_SMALL

        path = resolve_checkpoint(checkpoint, verify_sha256, expected_sha256)
        state = load_state(path)
        custom = backbone is not None
        if backbone is None:
            backbone = TransformersWavLM() if backend == "transformers" else s3prl_wavlm_large()
        model = ECAPA_TDNN_SMALL(backbone, feat_dim=1024, emb_dim=256)
        if isinstance(backbone, TransformersWavLM):
            state = remap_state(state, model)
        self.matched = load_verified(model, state, min_matched)
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.identity = {
            "model": "wavlm_large_finetune (seed-tts-eval SIM, ECAPA_TDNN_SMALL wavlm_large)",
            "repo": REPO if checkpoint is None else None,
            "revision": REVISION if checkpoint is None else None,
            "sha256": expected_sha256 if verify_sha256 else None,
            "backend": "custom" if custom else backend,
            "matched_tensors": self.matched,
        }
        self._cache = {}

    @torch.inference_mode()
    def embedding(self, audio):
        """L2-normalized 256-d embedding of a 1-D 16 kHz waveform."""
        wave = torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1).to(self.device)
        return F.normalize(self.model(wave).float(), dim=-1)[0].cpu()

    def file_embedding(self, path, cache=False):
        key = str(Path(path).resolve())
        if cache and key in self._cache:
            return self._cache[key]
        embedding = self.embedding(load_audio_16k(path))
        if cache:
            self._cache[key] = embedding
        return embedding

    def similarity(self, audio_path, prompt_path, cache_prompt=True):
        """Cosine of (generated audio, prompt); prompt embeddings are cached (prompts repeat across sentences)."""
        return float(self.file_embedding(audio_path) @ self.file_embedding(prompt_path, cache_prompt))
