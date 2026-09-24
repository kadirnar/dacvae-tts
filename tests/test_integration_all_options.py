"""Integration of the feature branches (#2-#16, tracking issue #17): the options work together.

Every feature is config-gated and off by default; each branch pins its own "off == main" regression
(test_speed, test_latent_negatives, test_block_options, test_pairs, test_teacher,
test_schedule_regularization). This file covers what no single branch can: options of different
branches combined in one model, objective and training run, plus the combinations that config
validation rejects on purpose.
"""

import torch

from dacvae_tts.config import ModelConfig, TrainConfig
from dacvae_tts.data import collate
from dacvae_tts.model import FlowTTS
from dacvae_tts.speed import training_loader
from dacvae_tts.training import Objective


def tiny_batch(frames=(40, 52, 33), prompts=(6, 9, 5)):
    torch.manual_seed(0)
    items = []
    for index, (length, prompt) in enumerate(zip(frames, prompts)):
        latents = torch.randn(length, 4)
        items.append(
            dict(
                reference=latents[:prompt],
                target=latents[prompt:],
                reference_text="Önceki cümle burada.",
                text=f"Söylenen sözler numara {index} burada.",
            )
        )
    return collate(items)


def test_loader_negatives_are_text_hinge_only():
    """#7 x #8: loader-side text negatives are drawn for the transcript hinge only; latent_delta
    corrupts target latents inside the step, so the loader stays the plain collate."""

    class Rows:
        costs = [3, 4, 5]
        epoch = 0

        def __len__(self):
            return 3

    train = TrainConfig(loader_negatives=True, contrastive_weight=0.2)
    assert training_loader(Rows(), train)[1].negatives
    train = TrainConfig(loader_negatives=True, contrastive_weight=0.2, contrastive_mode="latent_delta")
    items, collate_fn, _ = training_loader(Rows(), train)
    assert collate_fn is collate and isinstance(items, Rows)


def test_latent_delta_without_strict_checks_is_identical():
    """#7 x #8: strict_checks false only drops host-syncing value checks, latent negatives included."""
    cfg = ModelConfig(latent_dim=4, width=32, heads=2, depth=2, text_depth=1)
    results = []
    for strict in (True, False):
        torch.manual_seed(1)
        model = FlowTTS(cfg)
        torch.nn.init.normal_(model.output[-1].weight, std=0.05)
        model.strict_checks = strict
        objective = Objective(model, contrastive_mode="latent_delta").train()
        torch.manual_seed(2)
        results.append(objective(tiny_batch()))
    for key in ("flow", "latent_delta", "negative_random", "negative_aug"):
        assert torch.equal(results[0][key], results[1][key]), key


def test_utmos_flag_serves_protocol_and_eval_sentences():
    """#3 x #13: one `--utmos` switch. Bare it is the protocol's UTMOS22-strong (`utmos` field); with a
    model name (eval_sentences.py's former own flag) it picks that model, UTMOSv2 going to `utmosv2`."""
    import argparse

    from dacvae_tts.eval_protocol import add_protocol_args, protocol_from_args, utmos_models

    parser = argparse.ArgumentParser()
    add_protocol_args(parser)
    assert protocol_from_args(parser.parse_args([])) is None
    options = protocol_from_args(parser.parse_args(["--utmos"]))
    assert options.utmos and not options.utmosv2 and utmos_models(options) == ["utmos22"]
    options = protocol_from_args(parser.parse_args(["--utmos", "utmosv2"]))
    assert options.utmosv2 and not options.utmos and utmos_models(options) == ["utmosv2"]
    options = protocol_from_args(parser.parse_args(["--protocol-v2", "--utmos", "utmosv2"]))
    assert utmos_models(options) == ["utmos22", "utmosv2"]
