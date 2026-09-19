"""Bounded synthetic integration check, NOT a speech-quality or learnability experiment.

Run: python scripts/smoke_pipeline.py --output /tmp/dacvae-smoke-new
Requires codec extras and a CUDA GPU. Creates six tones and performs two toy updates.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for i in range(6):
        split = ("train", "val", "test")[i // 2]
        audio = root / f"tone-{i}.wav"
        t = np.arange(48000) / 48000
        sf.write(audio, 0.05 * np.sin(2 * np.pi * (200 + i * 73) * t), 48000)
        rows.append(
            dict(
                id=f"tone-{i}",
                audio=str(audio),
                text="Synthetic test words.",
                speaker_id=split,
                split=split,
                session_id=f"session-{i}",
            )
        )
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    commands = []

    def run(*arguments):
        command = [sys.executable, "-m", "dacvae_tts", *map(str, arguments)]
        commands.append(command)
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"}
        )
        (root / f"{len(commands):02d}-{arguments[0]}.log").write_text(result.stdout + result.stderr)

    run("prepare", "--manifest", manifest, "--output", root / "part")
    run("merge", "--inputs", root / "part", "--output", root / "cache")
    run("audit-cache", "--cache", root / "cache", "--scan-latents", "--output", root / "audit.json")
    run(
        "codec-reconstruct",
        "--manifest",
        manifest,
        "--cache",
        root / "cache",
        "--limit",
        2,
        "--output",
        root / "codec",
    )
    config = {
        "model": dict(latent_dim=128, width=32, depth=2, heads=2, text_depth=1),
        "train": dict(
            steps=2,
            warmup=1,
            batch_size=2,
            accumulation=1,
            workers=0,
            precision="bf16",
            checkpoint_every=1,
            validate_every=1,
            log_every=1,
            flow_reduction="frame",
            diagnostics_every=1,
        ),
    }
    (root / "toy.yaml").write_text(yaml.safe_dump(config))
    run("train", "--config", root / "toy.yaml", "--cache", root / "cache", "--output", root / "run")
    run(
        "make-cases",
        "--cache",
        root / "cache",
        "--split",
        "val",
        "--limit",
        1,
        "--output",
        root / "cases.jsonl",
        "--cross-session",
    )
    evaluation = dict(
        checkpoint=str(root / "run/last.pt"),
        cases=str(root / "cases.jsonl"),
        output=str(root / "evaluation"),
        max_cases=1,
        device="cuda",
        precision="bf16",
        sampling=dict(
            steps=[2],
            guidance=[1.5],
            sway=[-1],
            duration=["predicted", "ground_truth"],
            duration_scale=[1],
            seed=[42],
        ),
    )
    (root / "evaluation.yaml").write_text(yaml.safe_dump(evaluation))
    run("run-eval", "--config", root / "evaluation.yaml")
    report = dict(
        status="SYNTHETIC_INTEGRATION_ONLY",
        speech_quality_evaluated=False,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        commands=commands,
        codec=json.loads((root / "cache/metadata.json").read_text()),
        reconstruction=[
            json.loads(line)["diagnostics"]
            for line in (root / "codec/reconstructed.jsonl").read_text().splitlines()
        ],
        cases_sha256=json.loads((root / "cases.metadata.json").read_text())["cases_sha256"],
        summaries=[
            json.loads(path.read_text()) for path in sorted((root / "evaluation").glob("*/summary.json"))
        ],
    )
    (root / "smoke-report.json").write_text(json.dumps(report, indent=2))
    print(root / "smoke-report.json")


if __name__ == "__main__":
    main()
