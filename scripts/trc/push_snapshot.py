"""Push one training snapshot of a run, its evaluation and a listening set to a Hugging Face model repo.

Layout of the run's folder (--subdir; the repo root without it), repo created on first use, private unless --public:
  checkpoints/step-XXXXXXX.pt           the snapshot as written by `dacvae-tts train` (resumable)
  eval/step-XXXXXXX/summary.json        eval_sentences.py summary (WER/CER/SIM-o/DNSMOS/UTMOS ...)
  eval/step-XXXXXXX/results.jsonl       per-sentence scores and Whisper transcripts
  eval/step-XXXXXXX/audio/*.wav         the first --audio generated sentences + the prompts they used
  eval/step-XXXXXXX-s1000/              the second sampling seed of a final evaluation (--second-eval, no audio)
  config.json, train.jsonl, README.md   run configuration, training log and a table over all pushed steps
  scores.json                           the table as data; scripts/trc/hub_index.py builds the repo's landing page
                                        from the scores.json of every folder

Progress is kept in RUN/pushed-<repo name>.json, so re-running only uploads what is new.

  python scripts/trc/push_snapshot.py --run /workspace/runs/trc-base-s42 --repo VoiceHub/dacvae-tts-tr-combined \
      --subdir base-s42 --step 20000 --eval /workspace/outputs/trc/base-s42/step-0020000 --title "baseline"
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))

METRICS = (
    ("wer", "WER %", 100), ("cer", "CER %", 100), ("freya_wer", "Freya WER %", 100), ("freya_cer", "Freya CER %", 100),
    ("sim_o", "SIM-o", 1), ("dnsmos_ovrl", "DNSMOS OVRL", 1), ("dnsmos_sig", "SIG", 1), ("dnsmos_bak", "BAK", 1),
    ("sim_o_speechbrain", "SIM (SpeechBrain)", 1), ("utmos", "UTMOS", 1), ("files_clipping", "clipped files", 1),
)


def summary_value(summary, key):
    """eval_sentences.py summaries nest some protocol-v2 keys; look in the usual places."""
    for scope in (summary, summary.get("summary", {}), summary.get("v2", {}), summary.get("protocol_v2", {})):
        if isinstance(scope, dict) and isinstance(scope.get(key), (int, float)):
            return scope[key]
    return None


SCORES = ("count", "wer", "cer", "sim_o", "sim_o_speechbrain", "dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak", "utmos",
          "files_clipping")


def compact(summary):
    """The scores the landing page shows, from a full eval_sentences.py summary."""
    values = {key: summary_value(summary, key) for key in SCORES}
    return {key: value for key, value in values.items() if value is not None}


def scores(state):
    """scores.json of a folder: its title and notes, and per step the checkpoint kind and the compact scores."""
    steps = {}
    for step, entry in state["steps"].items():
        steps[step] = {"checkpoint": entry.get("checkpoint", False), "summary": compact(entry.get("summary") or {})}
        if entry.get("summary_s1000"):
            steps[step]["summary_s1000"] = compact(entry["summary_s1000"])
    return {"title": state.get("title", ""), "notes": state.get("notes", ""), "steps": steps}


def tree(repo, subdir, path):
    return f"https://huggingface.co/{repo}/tree/main/{subdir + '/' if subdir else ''}{path}"


def readme(title, repo, state, notes, subdir=""):
    name = subdir or repo.split("/")[-1]
    lines = [f"# {title}", "", f"Checkpoints and evaluations of `{name}` (DACVAE-TTS, Turkish zero-shot voice cloning, "
             "https://github.com/kadirnar/dacvae-tts)." + (f" All experiments: https://huggingface.co/{repo}." if subdir
                                                             else ""), ""]
    if subdir:
        lines += [f"Result against its baseline and verdict: [RESULT.md](https://huggingface.co/{repo}/blob/main/{subdir}/"
                  f"RESULT.md) · logs: [logs/]({tree(repo, subdir, 'logs')})", ""]
    if notes:
        lines += [notes, ""]
    lines += ["Evaluation: Freya-TR-Eval sentences spoken by leak-free Common Voice test voices (48 speakers), "
              "Whisper large-v3 (deterministic), turkish-v2 metric normalization, guidance 5, 32 Euler steps. "
              "Steps with fewer than 495 sentences are the quick intermediate check.", ""]
    header = "| step | sentences | " + " | ".join(label for _, label, _ in METRICS) + " | audio |"
    lines += [header, "|" + "---:|" * (len(METRICS) + 2) + "---|"]
    for step in sorted(state["steps"], key=int):
        entry = state["steps"][step]
        rows = [(str(int(step)), entry.get("summary") or {},
                 f"[listen]({tree(repo, subdir, f'eval/step-{int(step):07d}/audio')})" if entry.get("audio") else "")]
        if entry.get("summary_s1000"):
            rows.append((f"{int(step)} (sampling seed 1000)", entry["summary_s1000"], ""))
        for label, summary, audio in rows:
            cells = []
            for key, _, scale in METRICS:
                value = summary_value(summary, key)
                cells.append("" if value is None else f"{value * scale:.2f}" if scale == 100 else f"{value:.3f}")
            count = summary_value(summary, "sentences") or summary_value(summary, "count") or ""
            lines.append(f"| {label} | {count} | " + " | ".join(cells) + f" | {audio} |")
    lines += ["", "Load a checkpoint: `dacvae_tts.training.load_model(path)` or `dacvae-tts infer --checkpoint path ...` "
              "from https://github.com/kadirnar/dacvae-tts."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--subdir", default="", help="Folder of this run in the repo (one repo for all experiments)")
    parser.add_argument("--eval", help="eval_sentences.py output directory of this step")
    parser.add_argument("--second-eval", help="The same step evaluated with sampling seed 1000 (scores only)")
    parser.add_argument("--audio", type=int, default=24, help="Generated sentences to upload for listening")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--slim", action="store_true",
                        help="Upload the weights without optimizer/RNG state (~1/3 of the size; loadable, --init-from works)")
    parser.add_argument("--title", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    run = Path(args.run)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    state_path = run / f"pushed-{args.repo.split('/')[-1]}.json"
    legacy = run / f"pushed-dacvae-tts-trc-{args.subdir}.json"  # the run's former one-repo-per-arm state
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:  # steps pushed to the former repo are moved into the folder by scripts/trc/migrate_hub.py
        state = {"steps": json.loads(legacy.read_text())["steps"] if args.subdir and legacy.exists() else {}}
    state["title"] = args.title or state.get("title") or run.name
    state["notes"] = args.notes or state.get("notes", "")
    entry = state["steps"].setdefault(str(args.step), {})
    name = f"step-{args.step:07d}"
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        if not args.no_checkpoint and not entry.get("checkpoint"):
            checkpoint = run / f"{name}.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            (staging / "checkpoints").mkdir()
            target = staging / "checkpoints" / f"{name}.pt"
            if args.slim:
                import torch

                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                torch.save({k: v for k, v in saved.items() if k not in ("optimizer", "rng")}, target)
            elif checkpoint.stat().st_dev == staging.stat().st_dev:
                os.link(checkpoint, target)
            else:
                shutil.copy2(checkpoint, target)
            entry["checkpoint"] = "slim" if args.slim else "full"
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
        if args.second_eval and Path(args.second_eval, "summary.json").exists():
            source = Path(args.second_eval)
            target = staging / "eval" / f"{name}-s1000"
            target.mkdir(parents=True)
            for file in ("summary.json", "results.jsonl"):
                if (source / file).exists():
                    shutil.copy2(source / file, target / file)
            entry["summary_s1000"] = json.loads((source / "summary.json").read_text())
        for file in ("config.json", "train.jsonl", "validation.jsonl"):
            if (run / file).exists():
                shutil.copy2(run / file, staging / file)
        (staging / "README.md").write_text(readme(state["title"], args.repo, state, state["notes"], args.subdir))
        (staging / "scores.json").write_text(json.dumps(scores(state), indent=1))
        api.upload_folder(repo_id=args.repo, folder_path=str(staging), repo_type="model",
                          path_in_repo=args.subdir or None,
                          commit_message=f"{args.subdir or run.name}: step {args.step}")
    state_path.write_text(json.dumps(state, indent=1))
    if args.subdir:  # the landing page lists every folder; a failed refresh must not fail the push
        try:
            from hub_index import rebuild

            rebuild(api, args.repo)
        except Exception as error:  # noqa: BLE001
            print(f"landing page not refreshed: {error!r}", file=sys.stderr)
    print(json.dumps({"repo": args.repo, "step": args.step, "checkpoint": entry.get("checkpoint", False),
                      "eval": entry.get("audio", False)}))


if __name__ == "__main__":
    main()
