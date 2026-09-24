"""Sampler guidance variants: interval CFG, CFG rescale, APG, independent text/speaker guidance, padded batches."""

import torch

from dacvae_tts.config import ModelConfig
from dacvae_tts.model import FlowTTS, guided_update, sample, text_only_rows
from dacvae_tts.text import tokenize

NANO = dict(
    latent_dim=4, width=32, heads=2, depth=2, text_depth=1, text_attention=1, patch_size=1, positions="rope",
    qk_norm=True, prediction="edm", text_layout="joined", duration="rule", ctc_layer=1,
)


def trained_looking_model():
    torch.manual_seed(3)
    model = FlowTTS(ModelConfig(**NANO))
    # Zero-initialized AdaLN/output would make every branch identical; perturb them so guidance matters.
    for block in model.blocks:
        torch.nn.init.normal_(block.ada[-1].weight, std=0.05)
    torch.nn.init.normal_(model.output[-1].weight, std=0.05)
    return model.eval()


def rows(texts, reference_frames=5, target_frames=(7, 4)):
    """A padded batch: one shared reference, one target per text (different lengths)."""
    reference = torch.randn(reference_frames, 4)
    total = [reference_frames + n for n in target_frames]
    width = max(total)
    prompt = torch.zeros(len(texts), width, 4)
    prompt[:, :reference_frames] = reference
    prompt_mask = torch.zeros(len(texts), width, dtype=torch.bool)
    prompt_mask[:, :reference_frames] = True
    valid = torch.arange(width)[None] < torch.tensor(total)[:, None]
    encoded = [tokenize("Referans cümlesi.", text, version="turkish-v1", layout="joined") for text in texts]
    tokens = torch.nn.utils.rnn.pad_sequence([t for t, _ in encoded], batch_first=True)
    segments = torch.nn.utils.rnn.pad_sequence([s for _, s in encoded], batch_first=True)
    only = [tokenize("", text, version="turkish-v1", layout="joined") for text in texts]
    only_tokens = torch.nn.utils.rnn.pad_sequence([t for t, _ in only], batch_first=True)
    only_segments = torch.nn.utils.rnn.pad_sequence([s for _, s in only], batch_first=True)
    return dict(prompt=prompt, prompt_mask=prompt_mask, valid=valid, tokens=tokens, segments=segments), (only_tokens, only_segments)


def test_guided_update_defaults_and_neutral_apg_equal_plain_cfg():
    torch.manual_seed(0)
    v, u, x = torch.randn(2, 6, 4), torch.randn(2, 6, 4), torch.randn(2, 6, 4)
    mask = torch.ones(2, 6, dtype=torch.bool)
    t = torch.tensor([0.2, 0.7])
    plain = u + 4.0 * (v - u)
    assert torch.equal(guided_update(v, u, x, t, mask, 4.0), plain)
    # Taking the long path (a norm cap that never binds) must still be plain CFG.
    assert torch.allclose(guided_update(v, u, x, t, mask, 4.0, norm=1e9), plain, atol=1e-5)
    # eta=0 removes the component parallel to the conditional estimate: the result differs, stays finite.
    apg = guided_update(v, u, x, t, mask, 4.0, eta=0.0)
    assert torch.isfinite(apg).all() and not torch.allclose(apg, plain)
    # Rescale pulls the guided estimate's spread towards the conditional one.
    one_minus_t = (1 - t)[:, None, None]
    spread = lambda velocity: (x + one_minus_t * velocity).std(dim=(1, 2))  # noqa: E731
    rescaled = guided_update(v, u, x, t, mask, 4.0, rescale=1.0)
    assert torch.allclose(spread(rescaled), spread(v), rtol=1e-3)


def test_independent_guidance_with_equal_scales_is_plain_cfg_and_padding_is_harmless():
    model = trained_looking_model()
    batch, (only_tokens, only_segments) = rows(["Merhaba dünya.", "Kısa."])
    noise = torch.randn(batch["prompt"].shape)
    plain = sample(model, **batch, steps=4, guidance=3.0, initial_noise=noise)
    branch = text_only_rows(model, batch["valid"], batch["prompt_mask"], only_tokens, only_segments)
    same = sample(model, **batch, steps=4, guidance=3.0, initial_noise=noise, speaker_guidance=3.0, text_only=branch)
    target = batch["valid"] & ~batch["prompt_mask"]
    assert torch.allclose(plain[target], same[target], atol=1e-4)
    stats = {}
    different = sample(model, **batch, steps=4, guidance=3.0, initial_noise=noise, speaker_guidance=1.0,
                       text_only=branch, stats=stats)
    assert not torch.allclose(plain[target], different[target], atol=1e-4)
    assert stats["branch_evaluations"] == 12 and stats["forward_calls"] == 8
    # Row 1 of the padded batch equals the same request generated alone.
    single = {k: v[1:2, : int(batch["valid"][1].sum())] if k != "tokens" and k != "segments" else v[1:2] for k, v in batch.items()}
    single["tokens"] = batch["tokens"][1:2, : int(batch["tokens"][1].ne(0).sum())]
    single["segments"] = batch["segments"][1:2, : single["tokens"].size(1)]
    alone = sample(model, **single, steps=4, guidance=3.0, initial_noise=noise[1:2, : single["valid"].size(1)])
    length = int(batch["valid"][1].sum())
    assert torch.allclose(plain[1, :length], alone[0], atol=1e-4)


def test_interval_guidance_rescale_and_apg_run_and_count_evaluations():
    model = trained_looking_model()
    batch, _ = rows(["Bir iki üç.", "Dört beş."])
    stats = {}
    out = sample(model, **batch, steps=8, guidance=4.0, guidance_from=0.25, guidance_until=0.75, sway=0.0, stats=stats)
    assert torch.isfinite(out).all()
    assert stats["branch_evaluations"] == 8 + 4  # 4 of the 8 uniform steps start in [0.25, 0.75)
    for kwargs in (dict(cfg_rescale=0.7), dict(apg_eta=0.5, apg_momentum=-0.3), dict(apg_eta=0.0, apg_norm=2.0)):
        result = sample(model, **batch, steps=4, guidance=5.0, **kwargs)
        assert torch.isfinite(result).all()
        assert torch.equal(result[batch["prompt_mask"]], batch["prompt"][batch["prompt_mask"]])


@torch.inference_mode()
def speaker_guided_by_hand(model, batch, branch, noise, steps, speaker_guidance):
    """Euler loop of v_text + g_s (v_full - v_text) (text scale 1) from separate, unbatched forwards."""
    from dacvae_tts.contracts import mask_values, sanitize
    from dacvae_tts.model import time_grid, to_velocity

    prompt, prompt_mask, valid = batch["prompt"], batch["prompt_mask"], batch["valid"]
    mask = mask_values(valid, prompt_mask)
    x = sanitize(torch.where(prompt_mask[..., None], prompt, noise), valid)
    index = branch["index"][..., None].expand(-1, -1, x.size(-1))
    times = time_grid(steps, -1.0, x.device)
    for t0, t1 in zip(times[:-1], times[1:]):
        t = t0.expand(x.size(0))
        full = to_velocity(model, model(x, t, prompt, prompt_mask, valid, batch["tokens"], batch["segments"]), x, t)
        rows = sanitize(torch.gather(x, 1, index), branch["valid"])
        text = model(rows, t, branch["prompt"], branch["prompt_mask"], branch["valid"], branch["tokens"],
                     branch["segments"])
        text = torch.zeros_like(x).scatter_add(1, index, sanitize(to_velocity(model, text, rows, t), branch["valid"]))
        x = x + (t1 - t0) * sanitize(text + speaker_guidance * (full - text), mask)
        x = sanitize(torch.where(prompt_mask[..., None], prompt, x), valid)
    return x


def test_speaker_guidance_acts_when_the_text_scale_is_one():
    # It used to be skipped at guidance 1 (the demo's slider minimum): the output was the unguided one although the
    # metadata reported the speaker scale.
    model = trained_looking_model()
    batch, (only_tokens, only_segments) = rows(["Merhaba dünya.", "Kısa."])
    noise = torch.randn(batch["prompt"].shape)
    branch = text_only_rows(model, batch["valid"], batch["prompt_mask"], only_tokens, only_segments)
    target = batch["valid"] & ~batch["prompt_mask"]
    unguided = sample(model, **batch, steps=3, guidance=1.0, initial_noise=noise)
    stats = {}
    guided = sample(model, **batch, steps=3, guidance=1.0, initial_noise=noise, speaker_guidance=3.0,
                    text_only=branch, stats=stats)
    assert not torch.allclose(guided[target], unguided[target], atol=1e-3)
    expected = speaker_guided_by_hand(model, batch, branch, noise, 3, 3.0)
    assert torch.allclose(guided[target], expected[target], atol=1e-5)
    # Full and prompt-free branch per step; the null branch cancels at scale 1 and is not evaluated.
    assert stats["branch_evaluations"] == 6 and stats["forward_calls"] == 6
    # A late window with text scale 1 keeps the speaker guidance: the early window alone would differ.
    late = sample(model, **batch, steps=4, sway=0.0, guidance=3.0, guidance_split=0.5, guidance_late=1.0,
                  initial_noise=noise, speaker_guidance=3.0, text_only=branch, stats=stats)
    early_only = sample(model, **batch, steps=4, sway=0.0, guidance=3.0, guidance_until=0.5, initial_noise=noise,
                        speaker_guidance=3.0, text_only=branch)
    assert not torch.allclose(late[target], early_only[target], atol=1e-3)
    assert stats["late_window"]["guided_steps"] == 2 and stats["branch_evaluations"] == 2 * 3 + 2 * 2
