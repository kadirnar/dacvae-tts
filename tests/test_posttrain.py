import copy
import json

import torch

from dacvae_tts.config import Config, ModelConfig, TrainConfig
from dacvae_tts.data import LatentDataset, collate
from dacvae_tts.model import FlowTTS, sample
from dacvae_tts.posttrain import DistillObjective, PreferenceObjective, model_layout, representation_id


def test_preference_and_distill_backward(cache):
    data = LatentDataset(cache)
    batch = collate([data[0], data[1]])
    model = FlowTTS(ModelConfig(latent_dim=4, width=16, depth=1, heads=2, text_depth=1))
    ref = copy.deepcopy(model)
    objective = PreferenceObjective(model, ref)
    loser = {**batch, "latents": batch["latents"] + 0.1 * (~batch["prompt_mask"])[..., None]}
    loss = objective({"winner": batch, "loser": loser}, batch)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad is None for p in ref.parameters())
    model.zero_grad(set_to_none=True)
    condition = {k: v[:1] for k, v in batch.items() if k != "latents"}
    _, times, states = sample(model.eval(), **condition, steps=4, return_trajectory=True)
    # Materialize ordinary training tensors, as torch.save/load does for the offline cache.
    obj = {
        "times": times.clone(),
        "states": states[:, 0].clone(),
        "condition": {k: v.clone() for k, v in condition.items()},
    }
    loss = DistillObjective(model.train())([obj], batch)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_distill_regresses_the_velocity_of_an_edm_model(cache):
    # EDM models output the preconditioned F, not the velocity the trajectory target is built from: the
    # teacher's own unguided trajectory must score ~0 at every step (raw F scored 0.92/0.77/0.25/0.08).
    data = LatentDataset(cache)
    batch = collate([data[0], data[1]])
    model = FlowTTS(ModelConfig(latent_dim=4, width=16, depth=1, heads=2, text_depth=1, prediction="edm"))
    torch.nn.init.normal_(model.output[-1].weight, std=0.05)  # zero init would make F trivially 0
    model.eval()
    condition = {k: v[:1] for k, v in batch.items() if k != "latents"}
    _, times, states = sample(model, **condition, steps=4, guidance=1.0, return_trajectory=True)
    objective = DistillObjective(model, replay_weight=0.0)
    for index in range(len(times) - 1):
        # A two-state trajectory pins the scored step, which is otherwise drawn at random.
        obj = {
            "times": times[index : index + 2].clone(),
            "states": states[index : index + 2, 0].clone(),
            "condition": {k: v.clone() for k, v in condition.items()},
        }
        with torch.no_grad():
            assert objective([obj], batch).item() < 1e-10


def test_preference_and_distill_cli(cache, tmp_path):
    import os
    import subprocess
    import sys

    data = LatentDataset(cache)
    cfg = Config(ModelConfig(latent_dim=4, width=16, depth=1, heads=2, text_depth=1), TrainConfig())
    model = FlowTTS(cfg.model)
    saved = {
        "model": model.state_dict(),
        "ema": model.state_dict(),
        "config": cfg.to_dict(),
        "codec": data.meta,
        "mean": data.mean,
        "std": data.std,
    }
    checkpoint = tmp_path / "base.pt"
    torch.save(saved, checkpoint)
    identity = representation_id(saved)
    item = data[0]
    reference, winner, loser = [tmp_path / name for name in ("ref.pt", "win.pt", "lose.pt")]
    torch.save({**item, "representation": identity}, reference)
    torch.save({"target": item["target"], "representation": identity}, winner)
    torch.save({"target": item["target"] + 0.1, "representation": identity}, loser)
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        json.dumps(
            {
                "winner": str(winner),
                "loser": str(loser),
                "reference": str(reference),
                "representation": identity,
                "split": "train",
            }
        )
        + "\n"
    )
    batch = collate([item])
    condition = {k: v for k, v in batch.items() if k != "latents"}
    _, times, states = sample(model.eval(), **condition, steps=2, return_trajectory=True)
    trajectory = tmp_path / "trajectory.pt"
    torch.save(
        {"states": states[:, 0], "times": times, "condition": condition, "representation": identity},
        trajectory,
    )
    trajectories = tmp_path / "trajectories.jsonl"
    trajectories.write_text(
        json.dumps({"path": str(trajectory), "representation": identity, "split": "train"}) + "\n"
    )
    for mode, manifest in (("preference", pairs), ("distill", trajectories)):
        output = tmp_path / mode
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "dacvae_tts",
            "post-train",
            "--mode",
            mode,
            "--checkpoint",
            str(checkpoint),
            "--cache",
            str(cache),
            "--data",
            str(manifest),
            "--output",
            str(output),
            "--steps",
            "2",
            "--batch-size",
            "1",
            "--accumulation",
            "2",
            "--workers",
            "0",
            "--device",
            "cpu",
            "--precision",
            "fp32",
        ]
        result = subprocess.run(
            command, env={**os.environ, "OMP_NUM_THREADS": "1"}, capture_output=True, text=True, timeout=90
        )
        assert result.returncode == 0, result.stderr
        result = torch.load(output / f"{mode}-000002.pt", weights_only=True)
        assert result["stage"] == mode


def test_posttrain_rows_use_the_model_text_layout(cache):
    # candidates/distill_cache/replay rows are cross pairs; a joined-layout model must get [BOS ref SPACE target EOS]
    # in one segment, as in training, not the segments layout [BOS ref SEP target EOS] with segments 0/1.
    from dacvae_tts.text import SEP

    joined = ModelConfig(latent_dim=4, width=16, heads=2, text_layout="joined", duration="rule")
    batch = collate([LatentDataset(cache, "train", **model_layout(joined))[0]])
    assert (batch["segments"] == 1).all() and not (batch["tokens"] == SEP).any()
    segments = collate([LatentDataset(cache, "train", **model_layout(ModelConfig(latent_dim=4, width=16, heads=2)))[0]])
    assert (segments["tokens"] == SEP).any() and (segments["segments"] == 0).any()
