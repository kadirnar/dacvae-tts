import json
import math
import os
import subprocess
import sys
import types

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
    records = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert len([r for r in records if "flow" in r]) == 4
    validations = [r for r in records if "validation_loss" in r]
    assert [r["step"] for r in validations] == [2, 4]
    assert all(math.isfinite(r["validation_text_gain"]) for r in validations)


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


def test_compiler_failure_falls_back_to_eager(cache, tmp_path, monkeypatch, capsys):
    import torch._inductor.exc as inductor

    from dacvae_tts import training
    from dacvae_tts.config import Config

    calls = {"count": 0}

    def fake_compile(function, dynamic=True):
        def wrapped(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:  # the second forward hits a "rare shape"
                raise inductor.InductorError(AssertionError("synthetic"), None)
            return function(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(torch, "compile", fake_compile)
    config = config_file(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg["train"]["compile"] = "model"
    config.write_text(yaml.safe_dump(cfg))
    args = types.SimpleNamespace(
        config=str(config),
        cache=str(cache),
        output=str(tmp_path / "fallback"),
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
    _, saved = load_model(tmp_path / "fallback" / "last.pt")
    assert saved["step"] == 4 and calls["count"] == 2  # compiled path abandoned after the failure
    assert Config.from_dict(saved["config"]).train.compile == "model"  # resume must still match the YAML
    assert any("activation checkpointing" in line for line in capsys.readouterr().out.splitlines())


def test_wandb_tracking_mirrors_logs(cache, tmp_path, monkeypatch):
    """With a project set, train/val records reach wandb.log with step numbers; nothing else changes."""
    import sys
    import types

    from dacvae_tts import training

    calls = {"init": [], "log": [], "finish": 0}

    class FakeRun:
        def log(self, payload, step=None):
            calls["log"].append((step, payload))

        def finish(self):
            calls["finish"] += 1

    fake = types.ModuleType("wandb")
    fake.init = lambda **kwargs: calls["init"].append(kwargs) or FakeRun()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    config = config_file(tmp_path)
    args = types.SimpleNamespace(
        config=str(config),
        cache=str(cache),
        output=str(tmp_path / "tracked"),
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
        no_validation=False,
        stop_after=None,
        wandb_project="unit-test",
        wandb_group="g",
        wandb_id=None,
    )
    training.train(args)
    assert calls["init"][0]["project"] == "unit-test" and calls["init"][0]["name"] == "tracked"
    assert calls["init"][0]["config"]["train"]["wandb_project"] == "unit-test"
    steps = sorted({step for step, _ in calls["log"]})
    assert steps == [1, 2, 3, 4]
    keys = {key for _, payload in calls["log"] for key in payload}
    assert {"train/flow", "train/lr", "val/validation_flow", "val/validation_text_gain"} <= keys
    assert not any(isinstance(v, (list, dict, str)) for _, p in calls["log"] for v in p.values())
    assert calls["finish"] == 1
    assert (tmp_path / "tracked" / "train.jsonl").exists()
