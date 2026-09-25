"""Fixes from the tr-combined pre-flight review: exact tail-silence batch costs, target-only text negatives on
cross-prompt rows, and the compiled-block recompilation limit."""

import random

import torch

from dacvae_tts.data import LatentDataset, collate
from dacvae_tts.speed import compile_blocks, corrupt_rows, training_epoch_costs
from dacvae_tts.text import BYTE_OFFSET, SPACE, corrupt_transcript


def silent(cache):
    torch.save(torch.zeros(4), cache / "silence_raw.pt")


def tail_dataset(cache, monkeypatch):
    import dacvae_tts.data as data_module

    monkeypatch.setattr(data_module, "load_silence", lambda directory, meta: torch.zeros(4))
    return LatentDataset(cache, "train", pairing="within", layout="joined", tail_silence_prob=0.5,
                         tail_silence_max_seconds=0.5)


def test_tail_silence_costs_are_the_drawn_frames(cache, monkeypatch):
    data = tail_dataset(cache, monkeypatch)
    for epoch in (0, 1, 2):
        data.epoch = epoch
        costs = data.epoch_costs(epoch)
        for index in range(len(data)):
            item = data[(epoch, index)]
            assert len(item["reference"]) + len(item["target"]) == costs[index]
            assert item["tail_silence"] == data.tail_plan(epoch, index)
        assert (costs <= data.costs).all() and (costs >= data.lengths).all()
    train = type("T", (), dict(cross_prompt_prob=0.0, tempo_prompt_prob=0.0, tail_silence_prob=0.5, pad_multiple=1))
    assert training_epoch_costs(data, train) is not None  # the sampler batches by the exact per-epoch costs


def test_cross_rows_mark_where_the_target_transcript_starts(cache):
    data = LatentDataset(cache, "train", pairing="within", layout="joined", cross_prompt_prob=1.0,
                         cross_prompt_max_utterances=2, cross_prompt_max_seconds=60.0)
    within = LatentDataset(cache, "train", pairing="within", layout="joined")
    items = [data[i] for i in range(len(data))]
    batch = collate(items)
    assert "target_start" in batch
    for row, item in enumerate(items):
        alone = collate([{**item, "reference_token_ids": None, "reference_text": "", "reference_text_bytes": b""}])
        body = alone["tokens"][0][1:][alone["tokens"][0][1:] != 0]  # target body + EOS
        start = int(batch["target_start"][row])
        tokens = batch["tokens"][row]
        assert torch.equal(tokens[start : start + len(body)], body)
        assert int(tokens[start - 1]) == SPACE  # the joined layout's space between prompt and target transcripts
    assert "target_start" not in collate([within[i] for i in range(4)])  # default batches are unchanged


def test_negatives_never_change_prompt_words():
    tokens = torch.tensor([1] + [BYTE_OFFSET + 65, SPACE, BYTE_OFFSET + 66] + [SPACE] +
                          [BYTE_OFFSET + 67, SPACE, BYTE_OFFSET + 68, SPACE, BYTE_OFFSET + 69] + [2])
    segments = torch.ones_like(tokens)
    start = 5  # "A B" is the prompt transcript, "C D E" the target
    for seed in range(40):
        result = corrupt_transcript(tokens, segments, random.Random(seed), start)
        assert result is not None and torch.equal(result[0][:start], tokens[:start])
    rows = corrupt_rows(tokens[None], segments[None], [random.Random(3)], torch.tensor([start]))
    assert torch.equal(rows[0][0][:start], tokens[:start])
    assert corrupt_transcript(tokens, segments, random.Random(0), len(tokens)) is None  # no target words left


def test_block_compilation_allows_many_length_pairs():
    import inspect

    assert inspect.signature(compile_blocks).parameters["recompile_limit"].default >= 256


def test_dropout_models_compile_blocks_with_eager_rng():
    """Dropout in checkpointed compiled blocks failed Inductor's partitioner and fell back to eager for the run."""
    import torch._inductor.config as inductor_config

    from dacvae_tts.config import ModelConfig
    from dacvae_tts.model import FlowTTS

    model = FlowTTS(ModelConfig(latent_dim=4, width=16, heads=2, depth=2, text_depth=1, dropout=0.1))
    previous = inductor_config.fallback_random
    try:
        inductor_config.fallback_random = False
        compile_blocks(model)
        assert inductor_config.fallback_random
    finally:
        inductor_config.fallback_random = previous
