import json
import os
import subprocess
import sys

import pytest
import torch
import yaml

from dacvae_tts.training import load_model


def config_file(tmp_path):
    path = tmp_path / "test.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"latent_dim": 4, "width": 16, "depth": 1, "heads": 2, "text_depth": 1},
                "train": {
                    "steps": 4,
                    "warmup": 1,
                    "batch_size": 2,
                    "accumulation": 2,
                    "workers": 0,
                    "precision": "fp32",
                    "log_every": 1,
                    "checkpoint_every": 2,
                    "validate_every": 2,
                },
            }
        )
    )
    return path


def run(args):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    subprocess.run([sys.executable, *args], check=True, env=env, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("reduction", ["utterance", "frame"])
def test_pretraining_resume_is_exact(cache, tmp_path, reduction):
    config = config_file(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg["train"].update(
        flow_reduction=reduction, diagnostics_every=1, workers=2 if reduction == "frame" else 0
    )
    config.write_text(yaml.safe_dump(cfg))
    base = ["-m", "dacvae_tts", "train", "--config", str(config), "--cache", str(cache), "--device", "cpu"]
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run([*base, "--output", str(full)])
    run([*base, "--output", str(resumed), "--stop-after", "2"])
    run([*base, "--output", str(resumed), "--resume", str(resumed / "last.pt")])
    _, a = load_model(full / "last.pt")
    _, b = load_model(resumed / "last.pt")
    assert a["step"] == b["step"] == 4
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key
        assert torch.equal(a["ema"][key], b["ema"][key]), key


@pytest.mark.parametrize("reduction", ["utterance", "frame"])
def test_two_rank_ddp_smoke(cache, tmp_path, reduction):
    config = config_file(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg["train"].update(
        flow_reduction=reduction, diagnostics_every=1, workers=2 if reduction == "frame" else 0
    )
    config.write_text(yaml.safe_dump(cfg))
    output = tmp_path / "ddp"
    run(
        [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "dacvae_tts",
            "train",
            "--config",
            str(config),
            "--cache",
            str(cache),
            "--output",
            str(output),
            "--device",
            "cpu",
        ]
    )
    saved = torch.load(output / "last.pt", weights_only=True)
    assert saved["world_size"] == len(saved["rng"]) == 2
    assert saved["step"] == 4
    assert len([json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]) == 4


def test_warm_start_loads_weights_and_restarts_schedule(cache, tmp_path):
    config = config_file(tmp_path)
    base = ["-m", "dacvae_tts", "train", "--config", str(config), "--cache", str(cache), "--device", "cpu"]
    first, second = tmp_path / "first", tmp_path / "second"
    run([*base, "--output", str(first)])
    run([*base, "--output", str(second), "--init-from", str(first / "last.pt"), "--stop-after", "1"])
    _, a = load_model(first / "last.pt")
    _, b = load_model(second / "last.pt")
    assert b["step"] == 1 and b["init_from"] == str(first / "last.pt")
    # One update moved the warm-started weights away from the source checkpoint.
    assert any(not torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    assert all(torch.isfinite(b["model"][key]).all() for key in b["model"])
