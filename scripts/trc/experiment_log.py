"""Result log of every tr-combined experiment: what it asked, what it measured, what was decided and why.

Reads scripts/trc/experiments.yaml (question, evidence, baseline, verdict, conclusion per run) and adds the numbers:
the final evaluation paired against the run's baseline (both sampling seeds pooled, speaker-clustered jackknife-t
intervals, dacvae_tts.comparison), the quick-set trajectory and training statistics (final validation flow, seconds per
update, target frames per update). Writes EXPERIMENTS.md (all runs) and, in the experiments repo, <run>/RESULT.md and
<run>/logs/ (trainer output, runner log, evaluation logs); the local running notebook goes to logs/notebook.md.
Files that contain a credential from the environment are never uploaded.

  python scripts/trc/experiment_log.py                       # every run, upload
  python scripts/trc/experiment_log.py --runs x-long-skip    # one run (e.g. after its final evaluation)
  python scripts/trc/experiment_log.py --no-upload --output docs/experiments-log.md
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms import ARMS, CROSS, EXECUTION, V2  # noqa: E402
from run_arm import setting  # noqa: E402

from dacvae_tts.comparison import compare_evaluations  # noqa: E402
from dacvae_tts.data import jsonl  # noqa: E402

REGISTRY = Path(__file__).resolve().parent / "experiments.yaml"
FULL = 495
METRICS = (("wer", "WER %", 100), ("cer", "CER %", 100), ("sim_o", "SIM-o", 1), ("dnsmos_ovrl", "DNSMOS", 1),
           ("utmos", "UTMOS", 1))
SECRETS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN", "WANDB_API_KEY")
# Inference comparison of the models with one duration model (the refit predictor): OUT/systems/<dir>[-s1000].
HEADLINE = (("run C", "old-predictor-trc"), ("full-cross", "new-predictor-trc"), ("full-v2", "v2-predictor-trc"))
PROGRESS = re.compile(r"\d+%\|")  # tqdm upload/download bars in the runner logs


def evaluated(directory):
    return (directory / "results.jsonl").exists() and (directory / "summary.json").exists()


def final_dirs(out, run, entry):
    """The run's final evaluation: sampling seeds 42 and 1000 when both exist."""
    if entry.get("dirs"):
        return [out / d for d in entry["dirs"] if evaluated(out / d)]
    steps = []
    for summary in (out / run).glob("step-*/summary.json"):
        if not summary.parent.name.endswith("-s1000") and json.loads(summary.read_text()).get("count", 0) >= FULL:
            steps.append(int(summary.parent.name.split("-")[1]))
    if not steps:
        return []
    name = f"step-{max(steps):07d}"
    return [d for d in (out / run / name, out / run / f"{name}-s1000") if evaluated(d)]


def compare(run_dirs, base_dirs):
    """Pooled values of both systems and the paired differences (None when there is no baseline)."""
    systems = {"run": [list(jsonl(d / "results.jsonl")) for d in run_dirs]}
    if base_dirs:
        systems["baseline"] = [list(jsonl(d / "results.jsonl")) for d in base_dirs]
    try:
        report = compare_evaluations(systems, baseline="baseline" if base_dirs else "run")
    except ValueError as error:
        return None, None, f"not compared: {error}"
    values = {key: report["systems"]["run"]["metrics"].get(key, {}).get("value") for key, _, _ in METRICS}
    paired = report["comparisons"].get("run", {}).get("metrics", {}) if base_dirs else None
    return values, paired, ""


def fmt(value, scale):
    return "" if value is None else f"{value * scale:.2f}" if scale == 100 else f"{value:.3f}"


def change(run):
    """The run's own overrides (shared execution and cross-prompt options left out)."""
    if run not in ARMS:
        return ""
    purpose, _, overrides, _ = ARMS[run]
    shared = set(EXECUTION) | (set(CROSS) if run.startswith(("x-", "full-")) or "cross" in purpose else set())
    shared |= set(V2) if run.startswith("y-") else set()
    return ", ".join(f"`{o}`" for o in overrides if o not in shared)


def trajectory(out, run):
    points = []
    for summary in sorted((out / run).glob("step-*/summary.json")):
        if summary.parent.name.endswith("-s1000"):
            continue
        data = json.loads(summary.read_text())
        step = int(summary.parent.name.split("-")[1])
        points.append(f"{step // 1000 if step >= 1000 else step}{'k' if step >= 1000 else ''}: "
                      f"{100 * data['wer']:.1f} / {100 * data['cer']:.1f}" + ("" if data.get("count", 0) >= FULL else "*"))
    return ", ".join(points)


def training(runs, run):
    path = runs / f"trc-{run}" / "train.jsonl"
    if not path.exists():
        return ""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    steps = [r for r in records if "flow" in r and "elapsed_seconds" in r and r["step"] > 1]
    validation = [r for r in records if "validation_flow" in r]
    parts = []
    if validation:
        parts.append(f"final validation flow {validation[-1]['validation_flow']:.4f} (step {validation[-1]['step']})")
    if steps:
        parts.append(f"{np.median([r['elapsed_seconds'] / 100 for r in steps]):.3f} s/update")
        frames = [r["valid_target_frames"] for r in steps if "valid_target_frames" in r]
        if frames:
            parts.append(f"{np.mean(frames):,.0f} target frames/update")
    return "; ".join(parts)


def section(run, entry, registry, out, runs, repo):
    lines = [f"## `{run}` ({entry.get('issue', '')}): {entry.get('verdict', 'pending')}", ""]
    lines.append(f"**Question.** {entry['question']}")
    if change(run):
        lines.append(f"**Change.** {change(run)}")
    if entry.get("evidence"):
        lines.append(f"**Evidence.** {entry['evidence']}")
    base = entry.get("baseline")
    lines.append(f"**Baseline.** `{base}`" if base else "**Baseline.** none (reference run)")
    lines.append("")
    run_dirs = final_dirs(out, run, entry)
    if run_dirs:
        base_dirs = final_dirs(out, base, registry.get(base, {})) if base else []
        values, paired, note = compare(run_dirs, base_dirs)
        seeds = len(run_dirs)
        lines += [f"Final evaluation, {'sampling seeds 42 + 1000 pooled' if seeds == 2 else 'sampling seed 42'} "
                  f"({', '.join(d.parent.name + '/' + d.name for d in run_dirs)}):", ""]
        if values is None:
            lines += [note, ""]
        elif paired:
            lines += [f"| metric | `{base}` | `{run}` | difference [95 % CI] | verdict |", "|---|---:|---:|---|---|"]
            for key, label, scale in METRICS:
                m = paired.get(key)
                if m:
                    ci = f"[{fmt(m['ci'][0], scale)}, {fmt(m['ci'][1], scale)}]"
                    lines.append(f"| {label} | {fmt(m['baseline'], scale)} | {fmt(m['system'], scale)} | "
                                 f"{'+' if m['delta'] >= 0 else ''}{fmt(m['delta'], scale)} {ci} | {m['verdict']} |")
            lines.append("")
        else:
            lines += ["| " + " | ".join(label for _, label, _ in METRICS) + " |", "|" + "---:|" * len(METRICS),
                      "| " + " | ".join(fmt(values[key], scale) for key, _, scale in METRICS) + " |", ""]
    else:
        lines += ["No final evaluation yet.", ""]
    if trajectory(out, run):
        lines.append(f"**Trajectory** (WER / CER %; * = quick check, first 96 sentences): {trajectory(out, run)}")
    if training(runs, run):
        lines.append(f"**Training.** {training(runs, run)}")
    if entry.get("conclusion"):
        lines.append(f"**Verdict: {entry.get('verdict')}.** {entry['conclusion']}")
    lines += ["", f"Files: [checkpoints, evaluations, audio](https://huggingface.co/{repo}/tree/main/{run}) · "
                  f"[logs](https://huggingface.co/{repo}/tree/main/{run}/logs)", ""]
    return "\n".join(lines).replace("\n**", "\n\n**")  # every labelled field its own paragraph


def headline(out):
    rows = []
    for label, name in HEADLINE:
        dirs = [d for d in (out / "systems" / name, out / "systems" / f"{name}-s1000") if evaluated(d)]
        if dirs:
            values, _, _ = compare(dirs, [])
            if values:
                rows.append(f"| {label} | " + " | ".join(fmt(values[key], scale) for key, _, scale in METRICS) + " |")
    if not rows:
        return ""
    return "\n".join(["## Models side by side (refit duration predictor, seeds 42 + 1000 pooled)", "",
                      "| model | " + " | ".join(label for _, label, _ in METRICS) + " |", "|---|" + "---:|" * len(METRICS),
                      *rows, ""])


def document(registry, runs_to_show, out, runs, repo, sections):
    verdicts = {}
    for run in runs_to_show:
        verdicts.setdefault(registry[run].get("verdict", "pending"), []).append(f"`{run}`")
    lines = ["# tr-combined experiment log", "",
             "Every run of the tr-combined study (DACVAE-TTS, Turkish zero-shot TTS): the question, the evidence it came "
             "from, the measured result against its baseline and the decision. Protocol: Freya-TR-Eval, 495 sentences x "
             "48 leak-free Common Voice voices, one sample per sentence (no reranking), guidance 5, 32 steps, prompt-rate "
             "duration rule, Whisper large-v3, turkish-v2 metric; paired speaker-clustered jackknife-t 95 % intervals. "
             "A/B arms stop at 20k of the 60k schedule, where the training-seed spread is ~5 WER points: an option "
             "counts only when it beats both base seeds. Code: https://github.com/kadirnar/dacvae-tts (branch "
             "`trc/tr-combined-experiments`, registry `scripts/trc/experiments.yaml`).", "",
             "| verdict | runs |", "|---|---|"]
    lines += [f"| {verdict} | {', '.join(names)} |" for verdict, names in verdicts.items()]
    lines += ["", headline(out), *sections]
    return "\n".join(lines) + "\n"


def clean_copy(source, target, secrets):
    """Copy a log without progress bars; refuse files that contain a credential."""
    text = source.read_text(errors="replace")
    if any(secret and secret in text for secret in secrets):
        print(f"skipped {source}: contains a credential", file=sys.stderr)
        return False
    lines = [line for line in text.replace("\r", "\n").splitlines() if not PROGRESS.search(line)]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")
    return True


def stage_run(run, text, out, runs, stage, secrets):
    folder = stage / run
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "RESULT.md").write_text(text + "\n")
    sources = {"logs/train.log": runs / f"trc-{run}.log", "logs/runner.log": out / f"queue-{run}.log"}
    for log in sorted((out / run).glob("step-*.log")):
        sources[f"logs/eval-{log.stem}.log"] = log
    for name, source in sources.items():
        if source.exists():
            clean_copy(source, folder / name, secrets)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", help="Runs to refresh on the Hub (default: every run of the registry)")
    parser.add_argument("--repo", default=setting("HUB_REPO"))
    parser.add_argument("--output", default="", help="Also write EXPERIMENTS.md here (e.g. docs/experiments-log.md)")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    registry = yaml.safe_load(REGISTRY.read_text())
    unknown = [run for run in args.runs or [] if run not in registry]
    if unknown:
        sys.exit(f"not in {REGISTRY.name}: {unknown}")
    out, runs = Path(setting("OUT")), Path(setting("RUNS"))
    texts = {run: section(run, entry, registry, out, runs, args.repo) for run, entry in registry.items()}
    text = document(registry, list(registry), out, runs, args.repo, [texts[run] for run in registry])
    if args.output:
        Path(args.output).write_text(text)
    if args.no_upload:
        print(text)
        return
    secrets = [os.environ.get(name, "") for name in SECRETS]
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        (stage / "EXPERIMENTS.md").write_text(text)
        for run in args.runs or list(registry):
            stage_run(run, texts[run], out, runs, stage, secrets)
        if args.runs is None and (out / "RESULTS.md").exists():
            clean_copy(out / "RESULTS.md", stage / "logs" / "notebook.md", secrets)
        HfApi(token=os.environ.get("HF_TOKEN")).upload_folder(
            repo_id=args.repo, folder_path=str(stage), repo_type="model",
            commit_message=f"experiment log: {', '.join(args.runs) if args.runs else 'all runs'}")
    print(f"https://huggingface.co/{args.repo}/blob/main/EXPERIMENTS.md")


if __name__ == "__main__":
    main()
