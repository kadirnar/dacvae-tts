"""Contrastive modes: the transcript hinge stays bit-identical, latent negatives need no extra pass."""

import json
import types
from pathlib import Path

import pytest
import torch
import yaml
from torch.nn import functional as F

from dacvae_tts.config import Config, ModelConfig, TrainConfig
from dacvae_tts.data import collate
from dacvae_tts.model import FlowTTS, flow_loss, flow_target, per_example_mse
from dacvae_tts.negatives import (
    MAX_EDITS,
    augmented_negatives,
    load_silence,
    negative_distance,
    random_negatives,
)
from dacvae_tts.text import BYTE_OFFSET, corrupt_transcript
from dacvae_tts.training import Objective

NANO = dict(
    latent_dim=4,
    width=32,
    heads=2,
    depth=2,
    text_depth=1,
    text_attention=1,
    patch_size=1,
    positions="rope",
    qk_norm=True,
    prediction="edm",
    text_layout="joined",
    duration="rule",
    ctc_layer=1,
    adaln_rank=8,
)
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SEGMENTS = dict(latent_dim=4, width=32, heads=2, depth=2, text_depth=1)  # velocity + duration head


def randomized(model):
    """Nonzero output layers, so predictions (and every loss term) depend on the inputs."""
    torch.manual_seed(3)
    for block in model.blocks:
        last = block.ada_up if hasattr(block, "ada_up") else block.ada[-1]
        torch.nn.init.normal_(last.weight, std=0.05)
    if getattr(model, "ada_shared", None) is not None:
        torch.nn.init.normal_(model.ada_shared[-1].weight, std=0.05)
    torch.nn.init.normal_(model.output[-1].weight, std=0.05)
    return model


def batch_of(layout="joined", prompts=(3, 0, 5), frames=(40, 52, 33)):
    torch.manual_seed(0)
    items = []
    for index, (prompt, length) in enumerate(zip(prompts, frames)):
        latents = torch.randn(length, 4)
        reference = latents[:prompt] if layout == "joined" else latents[: max(prompt, 1)]
        items.append(
            dict(
                reference=reference,
                target=latents[len(reference) :],
                reference_text="" if layout == "joined" else "Some reference words.",
                text=f"Spoken words number {index} here.",
                layout=layout,
            )
        )
    return collate(items)


def masks(lengths, prompts):
    width = max(lengths)
    valid = torch.arange(width)[None] < torch.tensor(lengths)[:, None]
    prompt_mask = torch.arange(width)[None] < torch.tensor(prompts)[:, None]
    return valid, prompt_mask


def legacy_forward(objective, batch):
    """Objective.forward as of main 8f01f08 (text-negative hinge), kept verbatim as the reference."""
    model = objective.model
    cached = model.conditions(batch["prompt"], batch["prompt_mask"], batch["tokens"], batch["segments"])
    expanded, shared = batch, cached
    if objective.expansion > 1 and objective.training:
        expanded = {key: value.repeat_interleave(objective.expansion, 0) for key, value in batch.items()}
        shared = tuple(value.repeat_interleave(objective.expansion, 0) for value in cached)
    details = flow_loss(
        model,
        expanded,
        model.cfg.cond_dropout if objective.training else 0,
        return_details=True,
        cached=shared,
        time_sampling=objective.time_sampling,
    )
    details.pop("prediction", None)  # new key; the legacy dict did not carry it
    flow = details["flow"]
    if model.duration is None:
        duration_loss = flow.new_zeros(batch["latents"].size(0))
    else:
        frames = (batch["valid"] & ~batch["prompt_mask"]).sum(1)
        characters = ((batch["segments"] == 1) & (batch["tokens"] >= BYTE_OFFSET)).sum(1)
        log_rate = (frames / characters.clamp_min(1)).log()
        duration = model.predict_duration(
            batch["prompt"], batch["prompt_mask"], batch["tokens"], batch["segments"], cached=cached
        )
        duration_loss = F.smooth_l1_loss(duration.float(), log_rate, reduction="none")
    copies = flow.numel() // duration_loss.numel()
    if objective.contrastive_weight > 0 and objective.training:
        rows, usable = [], []
        for row_tokens, row_segments in zip(batch["tokens"].cpu(), batch["segments"].cpu()):
            corrupted = corrupt_transcript(row_tokens, row_segments, objective.rng)
            usable.append(corrupted is not None)
            rows.append(corrupted if corrupted is not None else (row_tokens, row_segments))
        width = max(len(t) for t, _ in rows)
        tokens = torch.zeros(len(rows), width, dtype=torch.int64)
        segments = torch.zeros(len(rows), width, dtype=torch.int64)
        for i, (t, s) in enumerate(rows):
            tokens[i, : len(t)], segments[i, : len(s)] = t, s
        usable = torch.tensor(usable)
        wrong = {**batch, "tokens": tokens, "segments": segments}
        negative = flow_loss(
            model,
            wrong,
            0,
            details["times"][::copies],
            details["noise"][::copies],
            cached=model.conditions(batch["prompt"], batch["prompt_mask"], tokens, segments),
        )
        positive = flow[::copies]
        hinge = F.relu(positive + objective.contrastive_margin * positive.detach() - negative)
        hinge = hinge.masked_fill(details["drop"][::copies] | ~usable, 0)
        details["contrastive"] = hinge.repeat_interleave(copies) / copies
    total = flow + objective.duration_weight * duration_loss.repeat_interleave(copies)
    return {"loss": total, "duration": duration_loss, **details}


@pytest.mark.parametrize("config,layout", [(NANO, "joined"), (SEGMENTS, "segments")])
@pytest.mark.parametrize("expansion", [1, 3])
def test_text_hinge_matches_legacy_objective(config, layout, expansion):
    model = randomized(FlowTTS(ModelConfig(**config))).train()
    batch = batch_of(layout)
    options = dict(expansion=expansion, ctc_weight=0.1, contrastive_weight=0.2, contrastive_margin=0.1)
    results = []
    for build in (
        lambda: legacy_forward(Objective(model, **options).train(), batch),
        lambda: Objective(model, **options).train()(batch),
        lambda: Objective(model, **options, contrastive_mode="text_hinge").train()(batch),
    ):
        torch.manual_seed(11)
        results.append(build())
    reference = results[0]
    assert "contrastive" in reference and reference["contrastive"].abs().sum() > 0
    for result in results[1:]:
        assert result.keys() == reference.keys()
        for key in reference:
            assert torch.equal(result[key], reference[key]), key
    # Same gradients through the optimized sum as well.
    grads = []
    for build in (
        lambda: legacy_forward(Objective(model, **options).train(), batch),
        lambda: Objective(model, **options).train()(batch),
    ):
        model.zero_grad()
        torch.manual_seed(11)
        losses = build()
        ctc = losses["ctc"].sum() if "ctc" in losses else 0
        (losses["loss"].sum() + 0.2 * losses["contrastive"].sum() + 0.1 * ctc).backward()
        grads.append([p.grad.clone() for p in model.parameters() if p.grad is not None])
    assert all(torch.equal(a, b) for a, b in zip(*grads))


def count_passes(model, monkeypatch):
    calls = {"forward": 0, "conditions": 0}
    forward, conditions = model.forward, model.conditions

    def counted_forward(*args, **kwargs):
        calls["forward"] += 1
        return forward(*args, **kwargs)

    def counted_conditions(*args, **kwargs):
        calls["conditions"] += 1
        return conditions(*args, **kwargs)

    monkeypatch.setattr(model, "forward", counted_forward)
    monkeypatch.setattr(model, "conditions", counted_conditions)
    return calls


def test_latent_delta_runs_one_generator_pass(monkeypatch):
    model = randomized(FlowTTS(ModelConfig(**NANO))).train()
    batch = batch_of()
    calls = count_passes(model, monkeypatch)
    losses = Objective(model, expansion=2, ctc_weight=0.1, contrastive_mode="latent_delta").train()(batch)
    assert calls == {"forward": 1, "conditions": 1}
    assert {"latent_delta", "negative_random", "negative_aug"} <= losses.keys()
    assert "contrastive" not in losses and "prediction" not in losses
    assert losses["latent_delta"].shape == losses["flow"].shape == (6,)
    applied = losses["negative_random"] > 0
    assert applied.any() and torch.allclose(
        losses["latent_delta"], -0.2 * losses["negative_random"] - 0.2 * losses["negative_aug"]
    )
    (losses["loss"].sum() + losses["latent_delta"].sum()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # The transcript hinge, for comparison, pays one more text encoding and generator pass.
    calls.update(forward=0, conditions=0)
    Objective(model, expansion=2, contrastive_weight=0.2).train()(batch)
    assert calls == {"forward": 2, "conditions": 2}
    # No negatives outside training or in `none` mode, whatever the weights say.
    calls.update(forward=0, conditions=0)
    for objective in (
        Objective(model, contrastive_mode="latent_delta").eval(),
        Objective(model, contrastive_weight=0.2, contrastive_mode="none").train(),
    ):
        losses = objective(batch)
        assert not {"latent_delta", "contrastive"} & losses.keys()
    assert calls == {"forward": 2, "conditions": 2}
    with pytest.raises(ValueError, match="contrastive_mode"):
        Objective(model, contrastive_mode="latent")


def reference_augment(sequence, draws, span, repeat_coverage, skip_coverage, edits):
    """Sequential, list-based statement of `augmented_negatives` for one target (None = fill)."""
    n, (low, high) = len(sequence), span
    skip = draws[0] < 0.5
    lower, upper = skip_coverage if skip else repeat_coverage
    lower, upper = torch.tensor(lower), torch.tensor(upper)  # float32 like the vectorized code
    remaining = int(((lower + draws[1] * (upper - lower)) * n).long())
    usable = remaining >= low
    played, content = list(range(n)), n
    for edit in range(edits):
        u = draws[2 + 3 * edit : 5 + 3 * edit]
        size = min(low + int((u[0] * (high - low + 1)).long()), remaining)
        if size < low:
            break
        if skip:
            start = min(int((u[1] * (content - size + 1)).long()), content - size)
            played = played[:start] + played[start + size :] + [-1] * size
            content -= size
        else:
            slots = n - size
            k = min(int((u[1] * (slots + 1)).long()), slots)
            s = (k + 1 + min(int((u[2] * slots).long()), slots - 1)) % (slots + 1)
            original = list(range(n))
            played = played[:k] + original[s : s + size] + played[k + size :]
        remaining -= size
    return [None if p < 0 else sequence[p] for p in played], usable


@pytest.mark.parametrize("seed", range(6))
def test_augmented_negatives_follow_the_paper_procedure_and_keep_masks(seed):
    generator = torch.Generator().manual_seed(100 + seed)
    rows = 7
    lengths = torch.randint(8, 90, (rows,), generator=generator)
    prompts = (torch.rand(rows, generator=generator) * 0.5 * lengths).long()
    valid, prompt_mask = masks(lengths.tolist(), prompts.tolist())
    latents = torch.randn(rows, valid.size(1), 3, generator=generator) * valid[..., None]
    fill = torch.tensor([7.0, 8.0, 9.0])
    options = dict(span=(2, 30), repeat_coverage=(0.2, 0.4), skip_coverage=(0.4, 0.8))
    draws = torch.rand(rows, 2 + 3 * MAX_EDITS, generator=torch.Generator().manual_seed(seed))
    negative, usable = augmented_negatives(
        latents, valid, prompt_mask, **options, fill=fill, generator=torch.Generator().manual_seed(seed)
    )
    assert negative.shape == latents.shape
    target = valid & ~prompt_mask
    # Prompt and padding frames never change, and only rows with a negative change at all.
    assert torch.equal(negative[~target], latents[~target])
    assert torch.equal(negative[~usable], latents[~usable])
    for b in range(rows):
        frames = latents[b, target[b]]
        expected, row_usable = reference_augment(list(frames), draws[b], **options, edits=MAX_EDITS)
        assert bool(usable[b]) == row_usable
        if row_usable:
            expected = torch.stack([fill if frame is None else frame for frame in expected])
            assert torch.equal(negative[b, target[b]], expected)
            assert not torch.equal(negative[b, target[b]], frames)  # never a no-op
    # Silence source: without an explicit fill, a row's own last target frame pads its tail.
    edge, _ = augmented_negatives(
        latents, valid, prompt_mask, **options, generator=torch.Generator().manual_seed(seed)
    )
    padded = (negative == fill).all(-1) & target
    last = latents[torch.arange(rows), lengths - 1]
    assert torch.equal(edge[padded], last[:, None].expand_as(latents)[padded])


def test_augmented_negative_coverage_matches_the_budget():
    torch.manual_seed(0)
    rows, n = 400, 200
    valid, prompt_mask = masks([n] * rows, [0] * rows)
    latents = torch.arange(n).float()[None, :, None].expand(rows, n, 1).contiguous()
    negative, usable = augmented_negatives(latents, valid, prompt_mask, fill=torch.tensor([-1.0]))
    assert usable.all()
    silent = (negative[..., 0] < 0).float().mean(1)
    changed = (negative[..., 0] != latents[..., 0]).float().mean(1)
    skip = silent > 0
    assert 0.3 < skip.float().mean() < 0.7
    # Skip: the tail holds exactly the skipped share, in [0.4, 0.8); what remains keeps its order.
    assert (silent[skip] >= 0.4 - 3 / n).all() and (silent[skip] < 0.8).all()
    kept = negative[skip][..., 0]
    assert all((row[row >= 0].diff() > 0).all() for row in kept)
    # Repeat: overwrites up to [0.2, 0.4) of the frames with copies of other frames of the target.
    assert (changed[~skip] < 0.4).all() and (changed[~skip] > 0).all()


def test_random_negatives_take_another_utterance_cropped_or_padded():
    lengths, prompts = [20, 14, 30, 30], [4, 0, 10, 6]
    valid, prompt_mask = masks(lengths, prompts)
    latents = torch.randn(4, 30, 2) * valid[..., None]
    fill = torch.tensor([5.0, 6.0])
    # Expansion 2: rows (0,1) and (2,3) are copies; the partner is always the other utterance.
    negative, usable = random_negatives(latents, valid, prompt_mask, copies=2, fill=fill)
    target = valid & ~prompt_mask
    assert usable.all() and torch.equal(negative[~target], latents[~target])
    for row, partner in ((0, 2), (1, 3), (2, 0), (3, 1)):
        own, other = int(target[row].sum()), latents[partner, target[partner]]
        expected = torch.cat([other[:own], fill.expand(max(own - len(other), 0), -1)])
        assert torch.equal(negative[row, target[row]], expected)
    # Default padding: the partner's own last target frame (row 3 is longer than its partner row 1).
    edge, _ = random_negatives(latents, valid, prompt_mask, copies=2)
    assert (edge[3, target[3]][14:] == latents[1, lengths[1] - 1]).all()
    # One utterance (all rows are copies of it): no negative anywhere.
    alone, usable = random_negatives(latents, valid, prompt_mask, copies=4)
    assert not usable.any() and torch.equal(alone, latents)


def test_negative_terms_skip_dropped_rows_prompt_frames_and_reach_only_the_prediction():
    model = FlowTTS(ModelConfig(**NANO))
    batch = batch_of()
    torch.manual_seed(1)
    latents = batch["latents"].clone().requires_grad_(True)
    noise = torch.randn_like(latents).requires_grad_(True)
    prediction = torch.randn_like(latents).requires_grad_(True)
    time = torch.rand(3)
    drop = torch.tensor([False, True, False])
    objective = Objective(model, contrastive_mode="latent_delta", span=(2, 20)).train()
    terms = objective.latent_delta(
        {**batch, "latents": latents}, prediction, {"noise": noise, "times": time, "drop": drop}, 1
    )
    assert (terms["negative_random"][drop] == 0).all() and (terms["negative_aug"][drop] == 0).all()
    assert (terms["negative_random"][~drop] > 0).all() and (terms["latent_delta"][~drop] < 0).all()
    terms["latent_delta"].sum().backward()
    # Targets are data: no gradient reaches the latents or the noise, only the prediction.
    assert latents.grad is None and noise.grad is None
    target = batch["valid"] & ~batch["prompt_mask"]
    assert (prediction.grad[~target] == 0).all() and (prediction.grad[drop] == 0).all()
    assert (prediction.grad[target & ~drop[:, None]] != 0).any()
    # Negative targets equal the positive target on every non-target frame (prompt and padding).
    x1 = batch["latents"]
    positive = flow_target(model, x1, noise.detach(), time)
    for negative, _ in (
        random_negatives(x1, batch["valid"], batch["prompt_mask"]),
        augmented_negatives(x1, batch["valid"], batch["prompt_mask"], span=(2, 20)),
    ):
        difference = flow_target(model, negative, noise.detach(), time) - positive
        assert (difference[~target] == 0).all() and (difference[target] != 0).any()


@pytest.mark.parametrize("prediction_kind", ["velocity", "edm"])
def test_delta_objective_is_bounded_with_its_minimum_past_the_true_target(prediction_kind):
    model = FlowTTS(ModelConfig(**{**NANO, "prediction": prediction_kind}))
    batch = batch_of()
    valid, prompt_mask = batch["valid"], batch["prompt_mask"]
    target = valid & ~prompt_mask
    x1, noise, time = batch["latents"], torch.randn_like(batch["latents"]), torch.rand(3)
    positive = flow_target(model, x1, noise, time)
    random_latents, _ = random_negatives(x1, valid, prompt_mask)
    aug_latents, _ = augmented_negatives(x1, valid, prompt_mask, span=(2, 20))
    weights = (0.2, 0.2)
    negatives = [flow_target(model, z, noise, time) for z in (random_latents, aug_latents)]

    def objective(prediction):
        total = per_example_mse(prediction, positive, target)
        for weight, latents in zip(weights, (random_latents, aug_latents)):
            total = total - weight * negative_distance(model, prediction, latents, noise, time, target)
        return total.sum()

    optimum = (positive - sum(w * n for w, n in zip(weights, negatives))) / (1 - sum(weights))
    optimum = optimum.clone().requires_grad_(True)
    objective(optimum).backward()
    assert optimum.grad.abs().max() < 1e-5  # stationary point: F+ + Σλ(F+ - F-)/(1 - Σλ)
    floor = objective(optimum.detach())
    for scale in (1.0, 10.0, 1000.0):  # and a minimum: moving away only increases the objective
        assert objective(optimum.detach() + scale * torch.randn_like(optimum)) > floor
    # The cap bounds each distance by cap x the target gap, however far the prediction goes.
    far = positive + 1000.0 * (positive - negatives[1])
    gap = per_example_mse(positive, negatives[1], target)
    capped = negative_distance(model, far, aug_latents, noise, time, target, positive, cap=2.0)
    assert torch.equal(capped, 2.0 * gap)


def test_config_modes_validation_and_example():
    assert TrainConfig().contrastive_mode == "text_hinge"
    for bad in (
        dict(contrastive_mode="latent"),
        dict(contrastive_random_weight=0.6, contrastive_aug_weight=0.4),
        dict(contrastive_aug_weight=-0.1),
        dict(contrastive_span_min=0),
        dict(contrastive_span_min=30, contrastive_span_max=20),
        dict(contrastive_skip_coverage=[0.4, 1.0]),
        dict(contrastive_repeat_coverage=[0.3, 0.2]),
        dict(contrastive_repeat_coverage=[0.1, 0.2, 0.3]),
        dict(contrastive_negative_cap=-1.0),
    ):
        with pytest.raises(ValueError):
            TrainConfig(**bad)
    TrainConfig(contrastive_random_weight=0.6, contrastive_aug_weight=0.6, contrastive_negative_cap=2.0)
    # YAML lists become tuples, so a resumed run compares equal to the one that wrote the checkpoint.
    listed = Config.from_dict({"model": {}, "train": {"contrastive_skip_coverage": [0.4, 0.8]}})
    assert listed.to_dict() == Config.from_dict(json.loads(json.dumps(listed.to_dict()))).to_dict()
    assert listed.to_dict() == Config().to_dict()
    example = Config.load(CONFIGS / "experiments" / "tr_w512_latent_negatives.yaml").to_dict()
    base = Config.load(CONFIGS / "nano_tr_w512.yaml").to_dict()
    assert example["train"].pop("contrastive_mode") == "latent_delta"
    base["train"].pop("contrastive_mode")
    assert example == base


def test_load_silence_formats(tmp_path):
    assert load_silence(tmp_path, 4) is None
    torch.save(torch.ones(4), tmp_path / "silence.pt")
    assert torch.equal(load_silence(tmp_path, 4), torch.ones(4))
    torch.save({"latent": torch.arange(8.0).reshape(2, 4)}, tmp_path / "silence.pt")
    assert torch.equal(load_silence(tmp_path, 4), torch.tensor([2.0, 3.0, 4.0, 5.0]))
    torch.save(torch.ones(3), tmp_path / "silence.pt")
    with pytest.raises(ValueError, match="silence latent"):
        load_silence(tmp_path, 4)


def test_latent_delta_training_logs_negative_terms(cache, tmp_path, capsys):
    from dacvae_tts import training

    torch.save(torch.zeros(4), cache / "silence.pt")
    config = tmp_path / "latent.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {"latent_dim": 4, "width": 16, "depth": 1, "heads": 2, "text_depth": 1},
                "train": {
                    "steps": 3,
                    "warmup": 1,
                    "batch_size": 3,
                    "accumulation": 2,
                    "workers": 0,
                    "precision": "fp32",
                    "log_every": 1,
                    "checkpoint_every": 3,
                    "validate_every": 3,
                    "batch_expansion": 2,
                    "flow_reduction": "frame",
                    "contrastive_mode": "latent_delta",
                    "contrastive_span_min": 1,
                    "contrastive_span_max": 4,
                },
            }
        )
    )
    args = types.SimpleNamespace(
        config=str(config),
        cache=str(cache),
        output=str(tmp_path / "latent"),
        resume=None,
        init_from=None,
        device="cpu",
        steps=None,
        batch_size=None,
        accumulation=None,
        workers=None,
        precision=None,
        learning_rate=None,
        optimizer=None,
        worker_threads=None,
        prefetch_factor=None,
        loader_start_method=None,
        cuda_prefetch=None,
        frame_budget=0,
        compile=None,
        no_validation=True,
        stop_after=None,
    )
    training.train(args)
    assert '{"latent_negative_fill": "silence.pt"}' in capsys.readouterr().out
    records = [json.loads(line) for line in (tmp_path / "latent" / "train.jsonl").read_text().splitlines()]
    assert len(records) == 3
    for record in records:
        assert record["contrastive"] == 0.0
        assert record["latent_delta"] < 0 and record["negative_random"] > 0 and record["negative_aug"] > 0
        assert 0 < record["negative_random_coverage"] <= 1 and 0 < record["negative_aug_coverage"] <= 1
    # The logged delta is the weighted distances over the same frames as the flow term.
    for record in records:
        expected = -0.2 * (
            record["negative_random"] * record["negative_random_coverage"]
            + record["negative_aug"] * record["negative_aug_coverage"]
        )
        assert record["latent_delta"] == pytest.approx(expected, rel=1e-5)
