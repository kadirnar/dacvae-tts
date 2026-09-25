"""Run a command while holding one of N evaluation slots (a flock semaphore shared by every launcher).

Two evaluations (~6 GB each) and a training run (~9 GB) fit a 32 GB GPU; the slots keep a third from starting.

  python scripts/trc/eval_slot.py [--slots 2] -- .venv/bin/python scripts/eval_sentences.py ...
"""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

LOCKS = Path(os.environ.get("EVAL_LOCKS", "/workspace/outputs/trc"))


def acquire(slots):
    while True:
        for index in range(slots):
            handle = open(LOCKS / f".evaluation-slot-{index}.lock", "w")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return handle
            except BlockingIOError:
                handle.close()
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slots", type=int, default=int(os.environ.get("EVAL_SLOTS", "1")))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    held = acquire(args.slots)
    try:
        sys.exit(subprocess.run(command).returncode)
    finally:
        held.close()


if __name__ == "__main__":
    main()
