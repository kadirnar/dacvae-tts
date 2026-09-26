"""Rescore every arm evaluation whose rows failed in scoring (e.g. CUDA OOM next to other GPU jobs).

Finds OUT/<arm>/step-<N>[-s1000]/results.jsonl files with "score: ..." errors and reruns eval_sentences.py --rescore
on them through the evaluation lock (scripts/trc/eval_slot.py). The generated audio is reused; nothing is
resynthesized. Quick checks (every step but the final one) are the first 96 sentences.

  python scripts/trc/rescore_failed.py [--push]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_arm import PY, log, scoring_errors, setting  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--push", action="store_true", help="Refresh the Hub summary of each rescored step")
    args = parser.parse_args()
    out, stop = Path(setting("OUT")), int(setting("AB_STOP"))
    for results in sorted(out.glob("*/step-*/results.jsonl")):
        directory = results.parent
        if not scoring_errors(directory):
            continue
        arm, name = directory.parent.name, directory.name
        step = int(name.split("-")[1])
        seed = 1000 if name.endswith("-s1000") else 42
        command = [PY, "scripts/eval_sentences.py", "--checkpoint", "unused-on-rescore", "--prompt-set",
                   setting("PROMPTS"), "--sentences", setting("FREYA"), "--output", str(directory), "--guidance",
                   setting("GUIDANCE"), "--steps", setting("SAMPLE_STEPS"), "--asr-backend", "faster-whisper",
                   "--asr-device", "cuda", "--dnsmos", setting("DNSMOS"), "--protocol-v2", "--metric-normalization",
                   "turkish-v2", "--freya-metric", "--seed", str(seed), "--rescore"]
        if step != stop:
            command += ["--limit", setting("QUICK")]
        for attempt in range(3):
            before = scoring_errors(directory)
            if not before:
                break
            log(f"{arm}/{name}: {before} rows failed in scoring; rescoring (attempt {attempt + 1})")
            with open(directory.parent / f"{name}.rescore.log", "a") as stream:
                subprocess.run([PY, "scripts/trc/eval_slot.py", "--", *command], cwd=REPO, stdout=stream,
                               stderr=subprocess.STDOUT)
        left = scoring_errors(directory)
        summary = json.loads((directory / "summary.json").read_text())
        log(f"{arm}/{name}: {left} unscored rows left; WER {100 * summary['wer']:.2f} CER {100 * summary['cer']:.2f}")
        if args.push and not left and seed == 42:
            subprocess.run([PY, "scripts/trc/push_snapshot.py", "--run", str(Path(setting("RUNS")) / f"trc-{arm}"),
                            "--repo", setting("HUB_REPO"), "--subdir", arm, "--step", str(step), "--eval",
                            str(directory), "--no-checkpoint", "--title", f"DACVAE-TTS tr-combined A/B arm `{arm}`"],
                           cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
