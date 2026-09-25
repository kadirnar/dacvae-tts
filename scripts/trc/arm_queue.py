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
from arms import ARMS, FINETUNE, POSTTRAIN  # noqa: E402
from run_arm import PY, REPO, log, setting  # noqa: E402


def trainers():
    """Training processes on this machine (any launcher): the GPU is saturated by one."""
    # The pattern must not start with "-": pgrep would read it as its own option.
    # Anchored at the start of the command line: a shell whose text merely mentions the command must not count.
    found = subprocess.run(["pgrep", "-f", "^[^ ]*python[^ ]* (-u -m dacvae_tts train|[^ ]*dacvae-tts post-train)"],
                           capture_output=True, text=True)
    return len(found.stdout.split())


def training_active(arm):
    """An arm occupies a training slot while a trainer with its config lives (whoever launched it); its evaluations
    then overlap the next arm's training (one compiled arm already saturates an RTX 5090)."""
    found = subprocess.run(["pgrep", "-f", f"^[^ ]*python[^ ]* -u -m dacvae_tts train --config .*/trc-{arm}.yaml "],
                           capture_output=True)
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
    running, failed, started, streak = {}, [], {}, 0
    while pending or running:
        for arm, process in list(running.items()):
            if process.poll() is not None:
                del running[arm]
                if process.returncode:
                    failed.append(arm)
                    log(f"FAILED {arm} ({process.returncode})")
                    quick = time.time() - started[arm] < 900
                    streak = streak + 1 if quick else 0
                    if streak >= 2:  # two arms died within 15 minutes of starting: a shared cause, stop
                        log(f"stopping: {failed[-2:]} failed right after starting; see their trc-*.log")
                        pending.clear()
                else:
                    log(f"done {arm}")
        while pending and max(trainers(), sum(training_active(arm) for arm in ARMS)) < args.parallel:
            ready = [arm for arm in pending if all((cache / need).exists() for need in ARMS[arm][3])
                     and not training_active(arm) and (arm not in FINETUNE or Path(FINETUNE[arm]["init"]).exists())
                     and (arm not in POSTTRAIN or Path(POSTTRAIN[arm]["init"]).exists())]
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
            if arm in POSTTRAIN:  # an exclusive post-training job with its own script (training, evaluation, push)
                command = ["bash", POSTTRAIN[arm]["script"]]
            stream = open(out / f"queue-{arm}.log", "a")
            # "full-*" arms train the whole 60k schedule (the model candidates), the others stop at AB_STOP.
            env = {**os.environ, **({"AB_STOP": os.environ.get("AB_STEPS", "60000")} if arm.startswith("full-") else {})}
            if arm in FINETUNE:  # warm-started fine-tune: its own schedule length, guidance and initial weights
                fine = FINETUNE[arm]
                env.update(AB_STEPS=fine["steps"], AB_STOP=fine["steps"], GUIDANCE=fine["guidance"])
                command += ["--init-from", fine["init"]]
            running[arm] = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, env=env)
            started[arm] = time.time()
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
