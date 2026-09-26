"""Move the former one-repo-per-run layout (VoiceHub/dacvae-tts-trc-<run>) into folders of the experiments repo.

Each former repo is downloaded, uploaded as <run>/ of --repo, and its push state is merged into the run's state for the
new repo, so later push_snapshot.py --subdir pushes keep listing the moved steps. Seed-1000 final evaluations found
under OUT/<run>/step-*-s1000 are added (scores and per-sentence results). --systems adds the inference comparisons
(OUT/systems/*: summary and per-sentence results of each system). Idempotent: unchanged files are not uploaded again.
The former repos are left untouched.

  python scripts/trc/migrate_hub.py --runs full-cross x-repa ...   # or --all
  python scripts/trc/migrate_hub.py --systems
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hub_index import rebuild  # noqa: E402
from push_snapshot import readme, scores  # noqa: E402

RUNS = Path("/workspace/runs")
OUT = Path("/workspace/outputs/trc")
RUN_DIRS = {"run-c-reference": Path("/workspace/models/run-c")}
SYSTEMS = {"old": "run C (VoiceHub/dacvae-tts-tr-w512, Vyvo/tr-dataset-12)", "new": "full-cross 60k",
           "v2": "full-v2 60k"}
DURATIONS = {"auto": "automatic duration", "predictor": "duration predictor", "trc": "refit on tr-combined"}


def title_and_notes(text):
    lines = text.splitlines()
    title = lines[0][2:].strip() if lines and lines[0].startswith("# ") else ""
    notes = []
    for line in lines[3:]:
        if line.startswith("Evaluation:"):
            break
        notes.append(line)
    return title, "\n".join(notes).strip()


def migrate(api, repo, folder, work):
    former = f"VoiceHub/dacvae-tts-trc-{folder}"
    local = Path(snapshot_download(former, local_dir=work / folder, token=api.token))
    shutil.rmtree(local / ".cache", ignore_errors=True)
    (local / ".gitattributes").unlink(missing_ok=True)
    run = RUN_DIRS.get(folder, RUNS / f"trc-{folder}")
    legacy = run / f"pushed-dacvae-tts-trc-{folder}.json"
    if not legacy.exists():
        found = sorted(run.glob("pushed-dacvae-tts-trc-*.json")) if run.exists() else []
        legacy = found[0] if found else None
    if legacy is None:  # not a push_snapshot.py run (e.g. the listening set): keep its README, point links here
        if (local / "README.md").exists():
            text = (local / "README.md").read_text()
            (local / "README.md").write_text(text.replace(f"huggingface.co/{former}/tree/main/",
                                                          f"huggingface.co/{repo}/tree/main/{folder}/"))
    else:
        state_path = run / f"pushed-{repo.split('/')[-1]}.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"steps": {}}
        for step, entry in json.loads(legacy.read_text())["steps"].items():
            merged = state["steps"].setdefault(step, {})
            for key, value in entry.items():
                merged.setdefault(key, value)
        title, notes = title_and_notes((local / "README.md").read_text()) if (local / "README.md").exists() else ("", "")
        state["title"] = state.get("title") or title or folder
        state["notes"] = state.get("notes") or notes
        for step, entry in state["steps"].items():
            second = OUT / folder / f"step-{int(step):07d}-s1000"
            if (second / "summary.json").exists():
                target = local / "eval" / second.name
                target.mkdir(parents=True, exist_ok=True)
                for file in ("summary.json", "results.jsonl"):
                    if (second / file).exists():
                        shutil.copy2(second / file, target / file)
                entry["summary_s1000"] = json.loads((second / "summary.json").read_text())
        (local / "README.md").write_text(readme(state["title"], repo, state, state["notes"], folder))
        (local / "scores.json").write_text(json.dumps(scores(state), indent=1))
        state_path.write_text(json.dumps(state, indent=1))
    if any(path.is_file() for path in local.rglob("*")):
        api.upload_folder(repo_id=repo, folder_path=str(local), path_in_repo=folder, repo_type="model",
                          commit_message=f"{folder}: moved from {former}")
    shutil.rmtree(local)


def systems(api, repo, work):
    root = work / "systems"
    rows = []
    for source in sorted((OUT / "systems").iterdir()):
        if not source.is_dir() or not (source / "summary.json").exists() or "bo3" in source.name:
            continue  # best-of-N reranking is not reported: single samples only
        (root / source.name).mkdir(parents=True)
        for file in ("summary.json", "results.jsonl"):
            if (source / file).exists():
                shutil.copy2(source / file, root / source.name / file)
        summary = json.loads((source / "summary.json").read_text())
        system, *rest = source.name.split("-")
        seed = "1000" if rest and rest[-1] == "s1000" else "42"
        duration = " ".join(DURATIONS.get(part, part) for part in rest if part != "s1000") or "automatic duration"
        rows.append(f"| `{source.name}` | {SYSTEMS.get(system, system)} | {duration} | {seed} | "
                    f"{100 * summary['wer']:.2f} | {100 * summary['cer']:.2f} | {summary.get('sim_o', 0):.3f} | "
                    f"{summary.get('dnsmos_ovrl', 0):.3f} | {summary.get('utmos', 0):.3f} |")
    (root / "README.md").write_text(
        "# Inference comparisons (single sample)\n\nThe same 495 Freya-TR-Eval sentences and 48 Common Voice voices, "
        "each system with the same duration model; scores and per-sentence Whisper transcripts. Audio: `comparison/`."
        "\n\n| folder | model | duration | sampling seed | WER % | CER % | SIM-o | DNSMOS | UTMOS |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(rows) + "\n")
    api.upload_folder(repo_id=repo, folder_path=str(root), path_in_repo="systems", repo_type="model",
                      commit_message="systems: inference comparisons")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="VoiceHub/dacvae-tts-tr-combined")
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--all", action="store_true", help="Every VoiceHub/dacvae-tts-trc-* repo")
    parser.add_argument("--skip", nargs="*", default=[], help="With --all: runs still pushing to their former repo")
    parser.add_argument("--systems", action="store_true")
    args = parser.parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    folders = list(args.runs)
    if args.all:
        folders += [m.id.split("dacvae-tts-trc-", 1)[1] for m in api.list_models(author="VoiceHub", search="dacvae-tts-trc-")
                    if re.fullmatch(r"VoiceHub/dacvae-tts-trc-[\w.-]+", m.id)]
    folders = [folder for folder in dict.fromkeys(folders) if folder not in args.skip]
    with tempfile.TemporaryDirectory(dir="/workspace") as tmp:
        for folder in folders:
            migrate(api, args.repo, folder, Path(tmp))
            print(f"moved {folder}", flush=True)
        if args.systems:
            systems(api, args.repo, Path(tmp))
            print("systems", flush=True)
    rebuild(api, args.repo)
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
