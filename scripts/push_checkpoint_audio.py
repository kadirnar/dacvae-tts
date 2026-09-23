"""Push every scored checkpoint's generated audio (and metrics) of a run to a Hugging Face dataset, incrementally.

For each record in RUN/monitor.jsonl whose monitor folder has not been uploaded yet, upload
`monitor/<step-folder>/` (48 generated WAVs + per-case results.jsonl + sampler metadata JSON), the prompt WAVs,
cases.json, monitor.jsonl, train.jsonl, config.json and a README with the results table and a listening guide.
Progress is tracked in RUN/pushed.json, so the script can be run repeatedly (e.g. in a loop while training).

  python scripts/push_checkpoint_audio.py --run runs/tr-nano-a --repo VoiceHub/dacvae-tts-tr-nano-a [--watch 300]
"""

import argparse
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi

SETTINGS = ("guidance", "guidance_until", "noise_scale", "sway", "duration_scale", "sampler_steps")
DEFAULTS = (2.0, 1.0, 1.0, -1.0, 1.0, 16)


def folder_for(record):
    values = tuple(record.get(k, d) for k, d in zip(SETTINGS, DEFAULTS))
    if values == DEFAULTS:
        return Path(record["checkpoint"]).stem
    g, u, n, s, d, k = values
    return f"{Path(record['checkpoint']).stem}-g{g:g}-u{u:g}-n{n:g}-s{s:g}-d{d:g}-k{k}"


def readme(run, records, title, notes):
    config = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    rows = sorted(records, key=lambda r: (r.get("step") or 0, r.get("guidance") or 0))
    table = "| step | folder | WER | CER | SIM | duration ratio | guidance | steps |\n|---:|---|---:|---:|---:|---:|---:|---:|\n"
    for r in rows:
        if r.get("wer") is None:
            continue
        table += (f"| {r['step']} | `monitor/{folder_for(r)}` | {r['wer']:.3f} | {r['cer']:.3f} | "
                  f"{(r.get('speaker_similarity') or 0):.3f} | {(r.get('duration_ratio_mean') or 0):.2f} | "
                  f"{r.get('guidance')} | {r.get('sampler_steps')} |\n")
    return f"""---
license: cc-by-nc-4.0
language:
- tr
tags:
- text-to-speech
- flow-matching
- dacvae
- turkish
- audio
pretty_name: {title}
---

# {title}

Generated audio of every evaluated checkpoint of the training run `{run.name}` (Turkish zero-shot voice-cloning TTS,
[dacvae-tts](https://github.com/kadirnar/dacvae-tts), frozen Meta DACVAE latents, 48 kHz). This repository holds
**model outputs and metrics, not training data**. Training data: `Vyvo/tr-dataset-12` (Turkish podcast segments).

{notes}

## How to listen

- `monitor/prompt-<uid>.wav`: the reference voice given to the model (a real validation recording of an unseen speaker,
  decoded through the DACVAE codec, so it also shows the codec's own quality ceiling).
- `monitor/<step-folder>/NNN.wav`: the model's synthesis of case `NNN` — the text of another recording of the same
  speaker, in the prompt's voice. `monitor/<step-folder>/results.jsonl` lists per case: `text` (target transcript),
  `prompt_uid` (which prompt WAV), `hypothesis` (Whisper-large-v3 transcript of the synthesis), `wer`, `cer`,
  `speaker_similarity`, `duration_ratio`. `NNN.json` holds the sampler settings and timings.
- Folder name suffix `-gG-uU-nN-sS-dD-kK`: guidance G, guidance applied while t < U, initial-noise scale N, sway S,
  duration scale D, Euler steps K (no suffix = guidance 2, 16 steps).
- `monitor/cases.json`: the 48 (prompt, target) cases; `monitor.jsonl`: one summary row per evaluated checkpoint/setting;
  `train.jsonl`: training/validation curves; `config.json`: the full training configuration.

WER/CER: faster-whisper large-v3 (Turkish), Turkish text normalization (numbers spelled out, İ/ı-aware lower-casing,
punctuation removed). SIM: `microsoft/wavlm-base-plus-sv` cosine between synthesis and the codec-decoded prompt.
Corpus WER/CER over 48 cases of 10 held-out speakers; the ASR floor on real codec-decoded speech is WER ≈ 0.05 / CER ≈ 0.016.

## Results

{table}

## Configuration

```json
{json.dumps(config.get("model", {}), indent=2)}
```

```json
{json.dumps(config.get("train", {}), indent=2)}
```
"""


def push_once(api, run, repo, title, notes, checkpoints):
    log = run / "monitor.jsonl"
    if not log.exists():
        return 0
    records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    state_path = run / "pushed.json"
    pushed = set(json.loads(state_path.read_text())) if state_path.exists() else set()
    pending = [r for r in records if folder_for(r) not in pushed and (run / "monitor" / folder_for(r)).exists()]
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    existing = set(api.list_repo_files(repo, repo_type="dataset"))
    for checkpoint in checkpoints:  # explicit checkpoint uploads happen regardless of pending audio
        path = Path(checkpoint)
        if path.exists() and f"checkpoints/{path.name}" not in existing:
            api.upload_file(path_or_fileobj=str(path), path_in_repo=f"checkpoints/{path.name}", repo_id=repo,
                            repo_type="dataset", commit_message=f"{run.name}: {path.name}")
            print(f"pushed checkpoints/{path.name}", flush=True)
    if not pending and (run / "README.md").exists() and state_path.exists():
        return 0
    for record in pending:
        folder = folder_for(record)
        api.upload_folder(folder_path=str(run / "monitor" / folder), repo_id=repo, repo_type="dataset",
                          path_in_repo=f"monitor/{folder}", allow_patterns=["*.wav", "*.json", "*.jsonl"],
                          commit_message=f"{run.name}: audio of {folder}")
        pushed.add(folder)
        state_path.write_text(json.dumps(sorted(pushed)))
        print(f"pushed monitor/{folder}", flush=True)
    # Shared files: prompts, cases, logs, README (small; re-uploaded each time so the table stays current).
    api.upload_folder(folder_path=str(run / "monitor"), repo_id=repo, repo_type="dataset", path_in_repo="monitor",
                      allow_patterns=["prompt-*.wav", "cases.json"], commit_message=f"{run.name}: prompts and cases")
    (run / "README.md").write_text(readme(run, records, title, notes))
    api.upload_folder(folder_path=str(run), repo_id=repo, repo_type="dataset", path_in_repo="",
                      allow_patterns=["README.md", "monitor.jsonl", "train.jsonl", "config.json", "notes.md"],
                      commit_message=f"{run.name}: metrics and README")
    return len(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--checkpoints", nargs="*", default=[], help="Checkpoint files to upload as well")
    parser.add_argument("--watch", type=int, default=0, help="Seconds between passes; 0 = one pass")
    args = parser.parse_args()
    run = Path(args.run).resolve()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    title = args.title or run.name
    while True:
        try:
            count = push_once(api, run, args.repo, title, args.notes, args.checkpoints)
            if count:
                print(f"{time.strftime('%H:%M')} pushed {count} folder(s) -> https://huggingface.co/datasets/{args.repo}", flush=True)
        except Exception as error:  # network hiccups: keep watching
            print(f"push failed: {error}", flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
