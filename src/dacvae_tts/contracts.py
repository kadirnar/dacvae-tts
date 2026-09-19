"""Public frame-level contracts. Shape checks are cheap; value checks run at boundaries."""

import torch


def target_mask(valid, reference):
    if valid.shape != reference.shape or valid.ndim != 2:
        raise ValueError("valid/reference masks must both have shape [B,L]")
    if valid.dtype != torch.bool or reference.dtype != torch.bool:
        raise ValueError("valid/reference masks must be boolean")
    return valid & ~reference


def audio_shapes(x, prompt, reference, valid, channels):
    if x.ndim != 3 or x.shape[-1] != channels:
        raise ValueError(f"audio must be [B,L,C={channels}], got {tuple(x.shape)}")
    if min(x.shape) < 1 or prompt.shape != x.shape:
        raise ValueError("audio and prompt must have identical nonempty [B,L,C] shapes")
    if valid.shape != x.shape[:2]:
        raise ValueError("audio masks must match [B,L]")
    target_mask(valid, reference)
    if not x.is_floating_point() or not prompt.is_floating_point():
        raise ValueError("audio and prompt must be floating point")
    if any(v.device != x.device for v in (prompt, reference, valid)):
        raise ValueError("audio, prompt and masks must share a device")


def mask_values(valid, reference, require_target=True):
    mask = target_mask(valid, reference)
    if (reference & ~valid).any():
        raise ValueError("reference frames must be a subset of valid frames")
    if (~valid[:, :-1] & valid[:, 1:]).any():
        raise ValueError("valid frames must be a contiguous prefix followed by padding")
    if require_target and (mask.sum(1) == 0).any():
        raise ValueError("Each example must contain at least one valid target frame")
    return mask


def text_shapes(tokens, segments, batch, device):
    if tokens.ndim != 2 or segments.shape != tokens.shape or tokens.size(0) != batch or tokens.size(1) < 1:
        raise ValueError("tokens and segments must be matching nonempty [B,S] tensors")
    if tokens.dtype != torch.long or segments.dtype != torch.long:
        raise ValueError("tokens and segments must be int64")
    if tokens.device != device or segments.device != device:
        raise ValueError("text and audio must share a device")


def sanitize(x, mask):
    # Multiplication is insufficient: NaN * 0 is still NaN.
    return x.masked_fill(~mask[..., None], 0)


def normalization_stats(mean, std, channels=None):
    if mean.ndim != 1 or std.shape != mean.shape or (channels is not None and len(mean) != channels):
        raise ValueError("normalization mean/std must be [C] and match codec channels")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("normalization statistics must be finite with strictly positive standard deviations")
