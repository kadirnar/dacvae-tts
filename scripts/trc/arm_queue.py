"""Run A/B arms (scripts/trc/arms.py) through scripts/trc/run_arm.py, `--parallel` at a time on one GPU.

`--parallel` counts training slots: an arm frees its slot when its final snapshot exists, so its final evaluations
overlap the next arm's training (one compiled arm already saturates an RTX 5090: two at once ran at 0.93x the
sequential throughput). Arms that share a comparison share the GPU model, cache, frame budget and schedule. Finished
arms (OUT/<arm>/done) are skipped; an arm whose stores are missing waits until the end of the queue and is reported.

  python scripts/trc/arm_queue.py base-s42 base-eager base-s43 --parallel 2
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms import ARMS  # noqa: E402
from run_arm import PY, REPO, log, setting  # noqa: E402


def training_active(arm):
    """An arm occupies a training slot while its run_arm.py lives (any queue) and its final snapshot is missing;
    its evaluations may then overlap the next arm's training."""
    final = Path(setting("RUNS")) / f"trc-{arm}" / f"step-{int(setting('AB_STOP')):07d}.pt"
    if final.exists():
        return False
    found = subprocess.run(["pgrep", "-f", f"scripts/trc/run_arm.py --arm {arm} --config"], capture_output=True)
    return found.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arms", nargs="+")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    unknown = [arm for arm in args.arms if arm not in ARMS]
    if unknown:
        sys.exit(f"unknown arms: {unknown}; known: {sorted(ARMS)}")
    out, cache = Path(setting("OUT")), Path(setting("CACHE"))
    pending = [arm for arm in args.arms if not (out / arm / "done").exists()]
    running, failed = {}, []
    while pending or running:
        for arm, process in list(running.items()):
            if process.poll() is not None:
                del running[arm]
                if process.returncode:
                    failed.append(arm)
                    log(f"FAILED {arm} ({process.returncode})")
                else:
                    log(f"done {arm}")
        while pending and sum(training_active(arm) for arm in ARMS) < args.parallel:
            ready = [arm for arm in pending if all((cache / need).exists() for need in ARMS[arm][3])
                     and not training_active(arm)]
            if not ready:
                break
            arm = ready[0]
            pending.remove(arm)
            issue, config, overrides, _ = ARMS[arm]
            command = [PY, "scripts/trc/run_arm.py", "--arm", arm, "--config", config, "--gpu", args.gpu,
                       "--notes", f"Trained on [Codyfederer/tr-combined](https://huggingface.co/datasets/Codyfederer/"
                                  f"tr-combined). Issue {issue}. Base `{config}`; overrides: {', '.join(overrides) or 'none'}. "
                                  f"Frame budget {setting('FRAME_BUDGET')}, {setting('AB_STEPS')}-update LR schedule "
                                  f"stopped at {setting('AB_STOP')}; cache {cache.name} of tr-combined."]
            if overrides:
                command += ["--set", *overrides]
            stream = open(out / f"queue-{arm}.log", "a")
            running[arm] = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, env=os.environ)
            log(f"start {arm} ({issue})")
            time.sleep(90)  # stagger compilation and loader start-up
        if pending and not running and not any(all((cache / n).exists() for n in ARMS[a][3]) for a in pending):
            log(f"waiting for stores of {pending}")
            break
        time.sleep(30)
    if failed:
        log(f"failed arms: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
