"""Landing page of the single experiments repo: one table over every run folder, rebuilt from their scores.json.

push_snapshot.py --subdir keeps <folder>/scores.json next to each run's checkpoints and evaluations and calls
rebuild() after every push, so the page always lists every folder. scripts/trc/hub_header.md (the current conclusions,
edited by hand) goes on top. Single-sample scores only: no reranking anywhere.

  python scripts/trc/hub_index.py --repo VoiceHub/dacvae-tts-tr-combined
"""

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms import ARMS, CROSS, EXECUTION  # noqa: E402

HEADER = Path(__file__).resolve().parent / "hub_header.md"
FULL = 495  # sentences of a final evaluation; fewer = the quick intermediate check
COLUMNS = (("wer", "WER %", 100), ("cer", "CER %", 100), ("sim_o", "SIM-o", 1), ("dnsmos_ovrl", "DNSMOS", 1),
           ("utmos", "UTMOS", 1))


OTHER = {"run-c-reference": "run C (VoiceHub/dacvae-tts-tr-w512, old data Vyvo/tr-dataset-12), same protocol",
         "base-s42-pad64": "#7 baseline with lengths padded to multiples of 64 (20 % of every batch was padding)"}


def describe(folder):
    """What the run changes: its purpose and its own overrides (the shared execution and cross-prompt options left out)."""
    if folder not in ARMS:
        return OTHER.get(folder, "")
    purpose, _, overrides, _ = ARMS[folder]
    shared = set(EXECUTION) | (set(CROSS) if "cross" in purpose or folder.startswith(("x-", "full-")) else set())
    own = [o.split(".", 1)[-1] for o in overrides if o not in shared]
    text = purpose + (f": `{', '.join(own)}`" if own else "")
    return text if len(text) <= 160 else text[:157] + "...`"


def final(steps):
    """The last step with a full evaluation, else None."""
    done = [int(step) for step, entry in steps.items() if entry.get("summary", {}).get("count", 0) >= FULL]
    return max(done) if done else None


def pooled(entry, key):
    """Seeds 42 and 1000 pooled: the same sentences and voices, so the corpus rate of both is their mean."""
    first, second = entry.get("summary", {}).get(key), entry.get("summary_s1000", {}).get(key)
    if first is None:
        return None
    return first if second is None else (first + second) / 2


def cell(value, scale):
    return "" if value is None else f"{value * scale:.2f}" if scale == 100 else f"{value:.3f}"


def order(folder):
    for rank, prefix in enumerate(("full-", "run-c", "ft-", "grpo", "base-", "pairs-", "x-")):
        if folder.startswith(prefix):
            return rank, folder
    return 9, folder


def page(repo, folders, scores, header):
    link = lambda folder: f"[`{folder}`](https://huggingface.co/{repo}/tree/main/{folder})"  # noqa: E731
    lines = ["---", "language: tr", "tags: [text-to-speech, voice-cloning, flow-matching]", "---", "",
             "# DACVAE-TTS Turkish: every tr-combined experiment in one place", "",
             "Training on [Codyfederer/tr-combined](https://huggingface.co/datasets/Codyfederer/tr-combined), code and "
             "write-up: https://github.com/kadirnar/dacvae-tts (branch `trc/tr-combined-experiments`). One folder per "
             "run: `checkpoints/step-*.pt`, `eval/step-*/` (scores, per-sentence Whisper transcripts, audio), "
             "`train.jsonl`, `config.json` and its own README.", ""]
    if header:
        lines += [header.strip(), ""]
    lines += ["## Final evaluation of every run", "",
              "Freya-TR-Eval (495 sentences) spoken by 48 leak-free Common Voice test voices, **one sample per "
              "sentence** (no reranking), guidance 5, 32 Euler steps, prompt-rate duration rule, Whisper large-v3, "
              "turkish-v2 metric. Seeds: sampling seeds of the final evaluation (2 = 42 and 1000 pooled). A/B arms stop "
              "at 20k of the 60k schedule; training-seed spread there is ~5 WER points.", "",
              "| run | what | updates | seeds | " + " | ".join(label for _, label, _ in COLUMNS) + " |",
              "|---|---|---:|---:|" + "---:|" * len(COLUMNS)]
    for folder in sorted(scores, key=order):
        steps = scores[folder]["steps"]
        step = final(steps)
        if step is None:
            continue
        entry = steps[str(step)]
        seeds = 2 if entry.get("summary_s1000") else 1
        lines.append(f"| {link(folder)} | {describe(folder)} | {step} | {seeds} | "
                     + " | ".join(cell(pooled(entry, key), scale) for key, _, scale in COLUMNS) + " |")
    trajectory = {folder: {int(s): e["summary"] for s, e in data["steps"].items() if e.get("summary", {}).get("wer")
                           is not None and int(s) % 5000 == 0}
                  for folder, data in scores.items()}
    columns = sorted({step for steps in trajectory.values() for step in steps})
    if columns:
        lines += ["", "## Training trajectories", "",
                  "WER / CER % per snapshot: the first 96 sentences (quick check) before the final step, seed 42.", "",
                  "| run | " + " | ".join(f"{step // 1000}k" for step in columns) + " |",
                  "|---|" + "---:|" * len(columns)]
        for folder in sorted(trajectory, key=order):
            steps = trajectory[folder]
            if len(steps) < 2:
                continue
            lines.append(f"| {link(folder)} | " + " | ".join(
                f"{100 * steps[s]['wer']:.1f} / {100 * steps[s]['cer']:.1f}" if s in steps else "" for s in columns)
                + " |")
    others = [folder for folder in folders if folder not in scores]
    if others:
        lines += ["", "## Other folders", ""] + [f"- {link(folder)}" for folder in sorted(others)]
    return "\n".join(lines) + "\n"


def rebuild(api, repo, header_path=HEADER):
    files = api.list_repo_files(repo)
    folders = sorted({f.split("/")[0] for f in files if "/" in f})
    scores = {}
    for file in files:
        parts = file.split("/")
        if len(parts) == 2 and parts[1] == "scores.json":
            scores[parts[0]] = json.loads(Path(hf_hub_download(repo, file, token=api.token)).read_text())
    header = header_path.read_text() if header_path and Path(header_path).exists() else ""
    api.upload_file(path_or_fileobj=page(repo, folders, scores, header).encode(), path_in_repo="README.md",
                    repo_id=repo, commit_message="landing page")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="VoiceHub/dacvae-tts-tr-combined")
    parser.add_argument("--header", default=str(HEADER))
    args = parser.parse_args()
    rebuild(HfApi(token=os.environ.get("HF_TOKEN")), args.repo, Path(args.header))
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
