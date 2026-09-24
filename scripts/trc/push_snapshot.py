"""Push one training snapshot of a run, its evaluation and a listening set to a Hugging Face model repo.

Layout of the repo (created on first use, private unless --public):
  checkpoints/step-XXXXXXX.pt           the snapshot as written by `dacvae-tts train` (resumable)
  eval/step-XXXXXXX/summary.json        eval_sentences.py summary (WER/CER/SIM-o/DNSMOS/UTMOS ...)
  eval/step-XXXXXXX/results.jsonl       per-sentence scores and Whisper transcripts
  eval/step-XXXXXXX/audio/*.wav         the first --audio generated sentences + the prompts they used
  config.json, train.jsonl, README.md   run configuration, training log and a table over all pushed steps

Progress is kept in RUN/pushed-<repo name>.json, so re-running only uploads what is new.

  python scripts/trc/push_snapshot.py --run /workspace/runs/trc-base-s42 --repo VoiceHub/dacvae-tts-trc-base-s42 \
      --step 20000 --eval /workspace/outputs/trc/base-s42/step-0020000 --title "baseline (run C recipe)"
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

METRICS = (
    ("wer", "WER %", 100), ("cer", "CER %", 100), ("freya_wer", "Freya WER %", 100), ("freya_cer", "Freya CER %", 100),
    ("sim_o", "SIM-o", 1), ("dnsmos_ovrl", "DNSMOS OVRL", 1), ("dnsmos_sig", "SIG", 1), ("dnsmos_bak", "BAK", 1),
    ("utmos", "UTMOS", 1), ("clipped_fraction", "clipped", 1),
)


def summary_value(summary, key):
    """eval_sentences.py summaries nest some protocol-v2 keys; look in the usual places."""
    for scope in (summary, summary.get("summary", {}), summary.get("v2", {}), summary.get("protocol_v2", {})):
        if isinstance(scope, dict) and isinstance(scope.get(key), (int, float)):
            return scope[key]
    return None


def readme(title, repo, state, notes):
    lines = [f"# {title}", "", f"Checkpoints and evaluations of `{repo.split('/')[-1]}` (DACVAE-TTS, Turkish, trained on "
             "[Codyfederer/tr-combined](https://huggingface.co/datasets/Codyfederer/tr-combined)).", ""]
    if notes:
        lines += [notes, ""]
    lines += ["Evaluation: Freya-TR-Eval sentences spoken by leak-free Common Voice test voices (48 speakers), "
              "Whisper large-v3 (deterministic), turkish-v2 metric normalization, guidance 5, 32 Euler steps. "
              "Steps with fewer than 495 sentences are the quick intermediate check.", ""]
    header = "| step | sentences | " + " | ".join(label for _, label, _ in METRICS) + " | audio |"
    lines += [header, "|" + "---:|" * (len(METRICS) + 2) + "---|"]
    for step in sorted(state["steps"], key=int):
        entry = state["steps"][step]
        summary = entry.get("summary") or {}
        cells = []
        for key, _, scale in METRICS:
            value = summary_value(summary, key)
            cells.append("" if value is None else f"{value * scale:.2f}" if scale == 100 else f"{value:.3f}")
        audio = f"[listen](https://huggingface.co/{repo}/tree/main/eval/step-{int(step):07d}/audio)" if entry.get("audio") else ""
        count = summary_value(summary, "sentences") or summary_value(summary, "count") or ""
        lines.append(f"| {int(step)} | {count} | " + " | ".join(cells) + f" | {audio} |")
    lines += ["", "Load a checkpoint: `dacvae_tts.training.load_model(path)` or `dacvae-tts infer --checkpoint path ...` "
              "from https://github.com/kadirnar/dacvae-tts."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--eval", help="eval_sentences.py output directory of this step")
    parser.add_argument("--audio", type=int, default=24, help="Generated sentences to upload for listening")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--title", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    run = Path(args.run)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    state_path = run / f"pushed-{args.repo.split('/')[-1]}.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"steps": {}}
    entry = state["steps"].setdefault(str(args.step), {})
    name = f"step-{args.step:07d}"
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        if not args.no_checkpoint and not entry.get("checkpoint"):
            checkpoint = run / f"{name}.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            (staging / "checkpoints").mkdir()
            os.link(checkpoint, staging / "checkpoints" / f"{name}.pt") if checkpoint.stat().st_dev == staging.stat().st_dev \
                else shutil.copy2(checkpoint, staging / "checkpoints" / f"{name}.pt")
            entry["checkpoint"] = True
        if args.eval and Path(args.eval, "summary.json").exists():
            source = Path(args.eval)
            target = staging / "eval" / name
            (target / "audio").mkdir(parents=True)
            for file in ("summary.json", "results.jsonl", "cases.json"):
                if (source / file).exists():
                    shutil.copy2(source / file, target / file)
            rows = [json.loads(line) for line in (source / "results.jsonl").read_text().splitlines() if line.strip()]
            prompts = set()
            for row in rows[: args.audio]:
                wav = source / f"{row['id']}.wav"
                if wav.exists():
                    shutil.copy2(wav, target / "audio" / wav.name)
                prompt = row.get("prompt_file") or row.get("prompt")
                if isinstance(prompt, str) and (source / Path(prompt).name).exists():
                    prompts.add(Path(prompt).name)
            for prompt in sorted(prompts) or [p.name for p in sorted(source.glob("prompt-*.wav"))[:8]]:
                shutil.copy2(source / prompt, target / "audio" / prompt)
            listing = [f"| {r['id']} | {r.get('text', '')} | {r.get('hypothesis', '')} | "
                       f"{r.get('wer', float('nan')):.3f} | {r.get('prompt_file', r.get('prompt', ''))} |"
                       for r in rows[: args.audio]]
            (target / "audio" / "README.md").write_text(
                "| id | text | Whisper large-v3 | WER | prompt |\n|---|---|---|---:|---|\n" + "\n".join(listing) + "\n")
            entry["summary"] = json.loads((source / "summary.json").read_text())
            entry["audio"] = True
        for file in ("config.json", "train.jsonl", "validation.jsonl"):
            if (run / file).exists():
                shutil.copy2(run / file, staging / file)
        (staging / "README.md").write_text(readme(args.title or run.name, args.repo, state, args.notes))
        api.upload_folder(repo_id=args.repo, folder_path=str(staging), repo_type="model",
                          commit_message=f"{run.name}: step {args.step}")
    state_path.write_text(json.dumps(state, indent=1))
    print(json.dumps({"repo": args.repo, "step": args.step, "checkpoint": entry.get("checkpoint", False),
                      "eval": entry.get("audio", False)}))


if __name__ == "__main__":
    main()
