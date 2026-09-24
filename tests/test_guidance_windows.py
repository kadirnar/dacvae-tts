"""Two guidance windows of `model.sample` (issue 13); the defaults stay bit-identical to the single window."""

import pytest
import torch
from test_guidance import rows, trained_looking_model

import dacvae_tts.model as model_module
from dacvae_tts.contracts import mask_values, sanitize
from dacvae_tts.model import guided_update, sample, time_grid, to_velocity


@torch.inference_mode()
def legacy_sample(model, prompt, prompt_mask, valid, tokens, segments, steps=16, guidance=1.5, seed=0, sway=-1.0,
                  guidance_until=1.0, guidance_from=0.0, cfg_rescale=0.0, apg_eta=1.0, apg_norm=0.0, apg_momentum=0.0):
    """The single-window Euler/CFG loop of `sample` before issue 13, kept to prove the defaults are bit-identical."""
    gen = torch.Generator(device=prompt.device).manual_seed(seed)
    mask = mask_values(valid, prompt_mask)
    prompt = sanitize(prompt, prompt_mask)
    x = torch.randn(prompt.shape, device=prompt.device, dtype=prompt.dtype, generator=gen) * 1.0
    x = sanitize(torch.where(prompt_mask[..., None], prompt, x), valid)
    times = time_grid(steps, sway, x.device)
    cond = model.conditions(prompt, prompt_mask, tokens, segments)
    zeros = torch.zeros_like
    pair = dict(
        prompt=torch.cat([prompt, zeros(prompt)]), prompt_mask=torch.cat([prompt_mask, zeros(prompt_mask)]),
        valid=valid.repeat(2, 1), tokens=tokens.repeat(2, 1), segments=segments.repeat(2, 1),
        cached=(torch.cat([cond[0], zeros(cond[0])]), cond[1].repeat(2, 1), torch.cat([cond[2], zeros(cond[2])])),
    )
    state = {}
    for t0, t1 in zip(times[:-1], times[1:]):
        t = t0.expand(x.size(0))
        if guidance == 1 or not guidance_from <= float(t0) < guidance_until:
            v = to_velocity(model, model(x, t, prompt, prompt_mask, valid, tokens, segments, cached=cond), x, t)
        else:
            both = torch.cat([x, x.masked_fill(prompt_mask[..., None], 0)])
            v, u = to_velocity(model, model(both, t.repeat(2), **pair), both, t.repeat(2)).chunk(2)
            v = guided_update(v, u, x, t, mask, guidance, cfg_rescale, apg_eta, apg_norm, apg_momentum, state)
        x = x + (t1 - t0) * sanitize(v, mask)
        x = sanitize(torch.where(prompt_mask[..., None], prompt, x), valid)
    return x


SETTINGS = [
    {},
    dict(guidance=3.0, steps=8),
    dict(guidance=4.0, steps=8, guidance_from=0.25, guidance_until=0.75, sway=0.0),
    dict(guidance=5.0, steps=6, cfg_rescale=0.7),
    dict(guidance=5.0, steps=6, apg_eta=0.5, apg_momentum=-0.3),
    dict(guidance=5.0, steps=6, apg_eta=0.0, apg_norm=2.0, seed=3),
]


@pytest.mark.parametrize("settings", SETTINGS)
def test_default_sampler_is_bit_identical_to_the_single_window_loop(settings):
    model = trained_looking_model()
    batch, _ = rows(["Merhaba dünya.", "Kısa."])
    assert torch.equal(sample(model, **batch, **settings), legacy_sample(model, **batch, **settings))


def test_guided_update_fixed_seed_regression():
    torch.manual_seed(0)
    v, u, x = torch.randn(2, 6, 4), torch.randn(2, 6, 4), torch.randn(2, 6, 4)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1, 4:] = False
    t = torch.tensor([0.2, 0.7])
    # Checksums of the implementation this issue builds on; guided_update itself is unchanged.
    expected = {"plain": 197.32058715820312, "rescale": 104.15213775634766, "apg": 160.05514526367188}
    state = {}
    got = {
        "plain": guided_update(v, u, x, t, mask, 4.0),
        "rescale": guided_update(v, u, x, t, mask, 4.0, rescale=0.7),
        "apg": guided_update(v, u, x, t, mask, 4.0, eta=0.5, momentum=-0.3, state=state),
    }
    for name, value in got.items():
        assert float(value.abs().sum()) == pytest.approx(expected[name], rel=1e-5), name
    assert "running" in state


def test_split_without_overrides_and_late_guidance_one_reproduce_single_windows():
    model = trained_looking_model()
    batch, _ = rows(["Bir iki üç.", "Dört."])
    for settings in (dict(guidance=4.0), dict(guidance=5.0, apg_eta=0.5, apg_momentum=-0.3, cfg_rescale=0.2)):
        single = sample(model, **batch, steps=8, **settings)
        for split in (0.0, 0.3, 0.5, 1.0):
            assert torch.equal(sample(model, **batch, steps=8, guidance_split=split, **settings), single)
    # Guiding only the early window is exactly the guidance interval [0, split).
    stats = {}
    early_only = sample(model, **batch, steps=8, sway=0.0, guidance=4.0, guidance_split=0.5, guidance_late=1.0,
                        stats=stats)
    assert torch.equal(early_only, sample(model, **batch, steps=8, sway=0.0, guidance=4.0, guidance_until=0.5))
    assert stats["branch_evaluations"] == 8 + 4 and stats["late_window"]["guided_steps"] == 0


def test_windows_apply_their_own_settings_per_step(monkeypatch):
    calls = []

    def recording(v, u, x, t, mask, guidance, rescale=0.0, eta=1.0, norm=0.0, momentum=0.0, state=None):
        calls.append((round(float(t[0]), 4), guidance, rescale, eta, norm, momentum, "running" in (state or {})))
        return guided_update(v, u, x, t, mask, guidance, rescale, eta, norm, momentum, state)

    monkeypatch.setattr(model_module, "guided_update", recording)
    model = trained_looking_model()
    batch, _ = rows(["Merhaba dünya.", "Kısa."])
    stats = {}
    out = sample(model, **batch, steps=8, sway=0.0, guidance=5.0, guidance_split=0.5, guidance_until=0.875,
                 guidance_late=3.0, apg_eta_late=0.5, cfg_rescale_late=0.3, apg_momentum_late=-0.3, stats=stats)
    assert torch.isfinite(out).all()
    early = [c for c in calls if c[0] < 0.5]
    late = [c for c in calls if c[0] >= 0.5]
    assert [c[0] for c in early] == [0.0, 0.125, 0.25, 0.375] and [c[0] for c in late] == [0.5, 0.625, 0.75]
    assert all(c[1:6] == (5.0, 0.0, 1.0, 0.0, 0.0) for c in early)
    assert all(c[1:6] == (3.0, 0.3, 0.5, 0.0, -0.3) for c in late)
    # Momentum used only late starts fresh at the split, then accumulates.
    assert [c[6] for c in late] == [False, True, True]
    assert stats["guidance_split"] == 0.5
    assert stats["late_window"] == dict(guidance=3.0, rescale=0.3, eta=0.5, norm=0.0, momentum=-0.3, guided_steps=3)
    assert stats["branch_evaluations"] == 8 + 7


def test_momentum_state_continues_across_windows_when_both_use_it(monkeypatch):
    seen = []

    def recording(v, u, x, t, mask, guidance, rescale=0.0, eta=1.0, norm=0.0, momentum=0.0, state=None):
        seen.append((float(t[0]), "running" in state))
        return guided_update(v, u, x, t, mask, guidance, rescale, eta, norm, momentum, state)

    monkeypatch.setattr(model_module, "guided_update", recording)
    model = trained_looking_model()
    batch, _ = rows(["Merhaba dünya.", "Kısa."])
    sample(model, **batch, steps=4, sway=0.0, guidance=4.0, apg_momentum=-0.3, guidance_split=0.5, apg_eta_late=0.0)
    assert seen == [(0.0, False), (0.25, True), (0.5, True), (0.75, True)]


def test_cfg_only_in_the_late_window_counts_evaluations():
    model = trained_looking_model()
    batch, _ = rows(["Merhaba dünya.", "Kısa."])
    stats = {}
    sample(model, **batch, steps=8, sway=0.0, guidance=1.0, guidance_split=0.5, guidance_late=4.0, stats=stats)
    assert stats["branch_evaluations"] == 8 + 4 and stats["late_window"]["guided_steps"] == 4


def test_window_settings_are_validated():
    model = trained_looking_model()
    batch, _ = rows(["Merhaba dünya.", "Kısa."])
    with pytest.raises(ValueError, match="need guidance_split"):
        sample(model, **batch, steps=2, guidance=3.0, apg_eta_late=0.5)
    with pytest.raises(ValueError, match="guidance_split"):
        sample(model, **batch, steps=2, guidance=3.0, guidance_until=0.5, guidance_split=0.7)
    for bad in (dict(cfg_rescale_late=1.5), dict(apg_momentum_late=1.0), dict(guidance_late=-1.0),
                dict(apg_norm_late=-1.0), dict(apg_eta_late=float("nan"))):
        with pytest.raises(ValueError, match="Late window"):
            sample(model, **batch, steps=2, guidance=3.0, guidance_split=0.5, **bad)
