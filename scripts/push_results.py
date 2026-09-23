"""Publish a training run's results to a Hugging Face dataset repository.

Uploads: config.json, train.jsonl, monitor.jsonl and per-checkpoint results, the monitor audio of the
selected checkpoints (plus the prompts), any extra evaluation folders, the selected checkpoint(s), and a
README.md with the metric table. Run after each training so results are reproducible and shareable.

  python scripts/push_results.py --run runs/tr-nano --repo VoiceHub/dacvae-tts-tr-nano \
      --checkpoints runs/tr-nano/step-0060000.pt --extra outputs/tr-nano-eval --notes "..."
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def monitor_table(run):
    log = run / "monitor.jsonl"
    if not log.exists():
        return "No monitor results.\n"
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: (r.get("step") or 0, r.get("guidance") or 0))
    head = "| step | cases | WER | CER | SIM (codec prompt) | duration ratio | guidance | steps | ASR |\n|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    body = "".join(
        f"| {r.get('step')} | {r.get('cases')} | {r.get('wer'):.3f} | {r.get('cer'):.3f} | "
        f"{(r.get('speaker_similarity') or 0):.3f} | {(r.get('duration_ratio_mean') or 0):.2f} | "
        f"{r.get('guidance')} | {r.get('sampler_steps')} | {r.get('asr_model')} |\n"
        for r in rows
        if r.get("wer") is not None
    )
    return head + body


def train_summary(run):
    log = run / "train.jsonl"
    if not log.exists():
        return ""
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    trains = [r for r in rows if "flow" in r]
    vals = [r for r in rows if "validation_flow" in r]
    lines = []
    if trains:
        last = trains[-1]
        lines.append(f"- Last training record: step {last['step']}, flow {last['flow']:.4f}, ctc {last.get('ctc', 0):.4f}, "
                     f"contrastive {last.get('contrastive', 0):.4f}, epoch {last.get('epoch')}")
    if vals:
        best = min(vals, key=lambda r: r["validation_flow"])
        lines.append(f"- Best validation flow {best['validation_flow']:.4f} at step {best['step']} "
                     f"(text gain {best.get('validation_text_gain', 0):.4f}); last {vals[-1]['validation_flow']:.4f} at step {vals[-1]['step']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--repo", required=True, help="ORG/NAME dataset repository")
    parser.add_argument("--checkpoints", nargs="*", default=[], help="Checkpoint files to upload")
    parser.add_argument("--audio-steps", nargs="*", default=[], help="Monitor folders (e.g. step-0060000) whose WAVs are uploaded; default: all")
    parser.add_argument("--extra", nargs="*", default=[], help="Extra folders copied under extra/")
    parser.add_argument("--notes", default="", help="Free-text notes for the README")
    parser.add_argument("--title", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--max-wavs", type=int, default=48, help="Per monitor folder")
    args = parser.parse_args()

    run = Path(args.run).resolve()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True, private=args.private)
    staging = Path(tempfile.mkdtemp(prefix="push-"))
    try:
        for name in ("config.json", "train.jsonl", "monitor.jsonl", "notes.md"):
            if (run / name).exists():
                shutil.copy(run / name, staging / name)
        monitor = run / "monitor"
        if monitor.exists():
            (staging / "monitor").mkdir()
            if (monitor / "cases.json").exists():
                shutil.copy(monitor / "cases.json", staging / "monitor" / "cases.json")
            for wav in sorted(monitor.glob("prompt-*.wav"))[: args.max_wavs]:
                shutil.copy(wav, staging / "monitor" / wav.name)
            folders = [monitor / s for s in args.audio_steps] if args.audio_steps else sorted(p for p in monitor.iterdir() if p.is_dir())
            for folder in folders:
                if not folder.exists():
                    continue
                target = staging / "monitor" / folder.name
                target.mkdir(parents=True)
                for f in sorted(folder.glob("results.jsonl")):
                    shutil.copy(f, target / f.name)
                for wav in sorted(folder.glob("*.wav"))[: args.max_wavs]:
                    shutil.copy(wav, target / wav.name)
                    meta = wav.with_suffix(".json")
                    if meta.exists():
                        shutil.copy(meta, target / meta.name)
        for extra in args.extra:
            source = Path(extra)
            if source.exists():
                shutil.copytree(source, staging / "extra" / source.name)
        (staging / "checkpoints").mkdir()
        for checkpoint in args.checkpoints:
            shutil.copy(checkpoint, staging / "checkpoints" / Path(checkpoint).name)
        config = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
        readme = f"""---
license: cc-by-nc-4.0
language:
- tr
tags:
- text-to-speech
- flow-matching
- dacvae
- turkish
pretty_name: {args.title or run.name}
---

# {args.title or run.name}

Turkish zero-shot voice-cloning TTS trained from scratch with [dacvae-tts](https://github.com/kadirnar/dacvae-tts)
on `Vyvo/tr-dataset-12` (Turkish podcast segments) in frozen Meta DACVAE latent space (48 kHz, 128 channels, 25 fps).

{args.notes}

## Training summary

{train_summary(run)}

## Held-out monitor results (unseen speakers, cross-utterance prompts)

WER/CER: faster-whisper large-v3 (Turkish), Turkish text normalization (numbers spelled out, İ/ı-aware
lower-casing, punctuation removed). SIM: `microsoft/wavlm-base-plus-sv` cosine against the codec-decoded prompt.

{monitor_table(run)}

## Files

- `config.json`, `train.jsonl`: full training configuration and loss curves (flow / CTC / contrastive / validation).
- `monitor.jsonl`, `monitor/step-*/results.jsonl`: per-checkpoint and per-case scores; `monitor/**/*.wav`: generated audio (48 kHz).
- `checkpoints/*.pt`: EMA + raw weights (`torch.load(..., weights_only=True)`), loadable with `dacvae_tts.inference.Synthesizer`.
- `extra/`: additional evaluation outputs, if any.

## Model configuration

```json
{json.dumps(config.get("model", {}), indent=2)}
```

```json
{json.dumps(config.get("train", {}), indent=2)}
```
"""
        (staging / "README.md").write_text(readme)
        api.upload_folder(folder_path=str(staging), repo_id=args.repo, repo_type="dataset", commit_message=f"Results of {run.name}")
        print(f"https://huggingface.co/datasets/{args.repo}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
