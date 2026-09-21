import torch

from dacvae_tts.text import BYTE_OFFSET


def make_batch(b, ref_frames, target_frames, ref_bytes=90, target_bytes=110, channels=128, device="cuda"):
    length = ref_frames + target_frames
    latents = torch.randn(b, length, channels, device=device)
    prompt_mask = torch.arange(length, device=device)[None].expand(b, -1) < ref_frames
    valid = torch.ones(b, length, dtype=torch.bool, device=device)
    tokens = torch.randint(
        BYTE_OFFSET + 32, BYTE_OFFSET + 120, (b, ref_bytes + target_bytes + 3), device=device
    )
    tokens[:, 0], tokens[:, ref_bytes + 1], tokens[:, -1] = 1, 2, 3
    segments = torch.zeros_like(tokens)
    segments[:, ref_bytes + 2 :] = 1
    return {
        "latents": latents,
        "prompt": latents * prompt_mask[..., None],
        "prompt_mask": prompt_mask,
        "valid": valid,
        "tokens": tokens,
        "segments": segments,
    }
