import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dacvae_tts.data import BucketBatchSampler, LatentDataset, collate, speaker_split
from dacvae_tts.metrics import error_counts, summarize
from dacvae_tts.posttrain import preference_loss, rank_pairs


def test_reference_pairing_and_cache(cache):
    dataset = LatentDataset(cache)
    for epoch in range(3):
        for i in range(len(dataset)):
            row = dataset[(epoch, i)]
            assert row["uid"] != row["reference_uid"]
            assert row["uid"].rsplit("-", 1)[0] == row["reference_uid"].rsplit("-", 1)[0]
            assert torch.isfinite(row["target"]).all()
    batch = collate([dataset[0], dataset[1]])
    assert torch.equal(batch["latents"][batch["prompt_mask"]], batch["prompt"][batch["prompt_mask"]])


def test_eight_rank_bucket_plan_no_overlap_or_step_mismatch():
    costs = np.arange(100, 413)
    batches = [list(BucketBatchSampler(costs, 4, rank=r, world_size=8, frame_budget=1300)) for r in range(8)]
    assert len(set(map(len, batches))) == 1
    seen = set()
    for rank_batches in batches:
        for batch in rank_batches:
            ids = [i for epoch, i in batch]
            assert not seen.intersection(ids)
            seen.update(ids)
            assert max(costs[ids]) * len(ids) <= 1300


def test_split_is_speaker_deterministic():
    assert speaker_split("speaker") == speaker_split("speaker")
    assert {speaker_split(str(i)) for i in range(1000)} == {"train", "val", "test"}


def test_corpus_wer_is_not_mean_of_utterance_rates():
    a = error_counts("hello", "")
    b = error_counts("one two three four", "one two three four")
    assert summarize([a, b])["wer"] == 0.2
    assert error_counts("Hello, WORLD!", "hello world")["cer"] == 0
    assert error_counts("one", "one two three")["wer"] == 2
    with pytest.raises(ValueError):
        error_counts("!!!", "")


def test_preference_gradient_direction():
    winner = torch.tensor([1.0], requires_grad=True)
    loser = torch.tensor([1.0], requires_grad=True)
    loss = preference_loss(winner, loser, torch.ones(1), torch.ones(1)).mean()
    loss.backward()
    assert winner.grad > 0  # Gradient descent reduces winner error.
    assert loser.grad < 0


def test_rank_rejects_validation_data(tmp_path):
    source = tmp_path / "scores.jsonl"
    source.write_text(json.dumps({"split": "val"}) + "\n")
    with pytest.raises(ValueError, match="training-split"):
        rank_pairs(SimpleNamespace(scores=source))
