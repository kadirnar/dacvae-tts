"""Train one A/B arm on the tr-combined cache, evaluate every kept snapshot while training continues, and push
each snapshot (checkpoint, scores, listening set) to a VoiceHub model repo.

  python scripts/trc/run_arm.py --arm base-s42 --config configs/trc/base.yaml
  python scripts/trc/run_arm.py --arm swiglu --config configs/trc/base.yaml --set model.ffn_activation=swiglu

Protocol (shared by every arm, so arms compare): the arm's config trains on $CACHE with frame budget $FRAME_BUDGET on
the unchanged $AB_STEPS-update LR schedule and stops at $AB_STOP. Snapshots every keep_every updates get the quick
evaluation ($QUICK Freya-TR-Eval sentences), the last one the full 495 sentences; both use the 48 leak-free Common
Voice voices, guidance 5, 32 Euler steps, protocol v2 (Whisper large-v3, SIM-o, DNSMOS, UTMOS) and the turkish-v2
metric. Everything is idempotent: a finished step is skipped, an interrupted run resumes from last.pt.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".venv" / "bin" / "python")
ENV = {
    "WORK": "/workspace",
    "CACHE": "/workspace/data/trc/clean",
    "RUNS": "/workspace/runs",
    "OUT": "/workspace/outputs/trc",
    "PROMPTS": "/workspace/data/eval/cv-tr-prompts/prompts.json",
    "FREYA": "/workspace/data/eval/freya_tr_eval.jsonl",
    "DNSMOS": "/workspace/models/sig_bak_ovr.onnx",
    "FRAME_BUDGET": "6000",
    "AB_STEPS": "60000",
    "AB_STOP": "20000",
    "QUICK": "96",
    "GUIDANCE": "5.0",
    "SAMPLE_STEPS": "32",
    "HF_ORG": "VoiceHub",
    "WANDB_PROJECT": "dacvae-tts-tr-combined",
}


def setting(name):
    return os.environ.get(name, ENV[name])


def log(message):
    print(time.strftime("%F %T"), message, flush=True)


def derive(base, output, overrides):
    config = yaml.safe_load(Path(base).read_text())
    for override in overrides:
        key, _, value = override.partition("=")
        section, _, field = key.partition(".")
        config[section][field] = yaml.safe_load(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = f"# Derived from {base}; changed: {', '.join(overrides) or 'nothing'}\n"
    output.write_text(header + yaml.safe_dump(config, sort_keys=False))
    subprocess.run([PY, "-c", f"from dacvae_tts.config import Config; Config.load({str(output)!r})"], check=True)
    return output


def evaluate(checkpoint, out, limit, gpu, seed=42):
    if (out / "summary.json").exists():
        return
    command = [
        PY, "scripts/eval_sentences.py", "--checkpoint", str(checkpoint), "--prompt-set", setting("PROMPTS"),
        "--sentences", setting("FREYA"), "--output", str(out), "--guidance", setting("GUIDANCE"),
        "--steps", setting("SAMPLE_STEPS"), "--asr-backend", os.environ.get("ASR_BACKEND", "faster-whisper"),
        "--asr-device", "cuda", "--dnsmos", setting("DNSMOS"), "--protocol-v2", "--metric-normalization", "turkish-v2",
        "--freya-metric", "--seed", str(seed),
    ]
    if limit:
        command += ["--limit", str(limit)]
    out.mkdir(parents=True, exist_ok=True)
    # At most EVAL_SLOTS (2) evaluations at once on the GPU next to the training run (~6 GB each, ~9 GB training).
    with open(out.parent / f"{out.name}.log", "a") as stream:
        subprocess.run([PY, "scripts/trc/eval_slot.py", "--", *command], check=True, cwd=REPO, stdout=stream,
                       stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})


def wandb_log(arm, step, out, final):
    project = setting("WANDB_PROJECT")
    if not project or not os.environ.get("WANDB_API_KEY"):
        return
    code = f"""
import json, wandb
summary = json.load(open({str(out / 'summary.json')!r}))
flat = {{k: v for k, v in summary.items() if isinstance(v, (int, float))}}
run = wandb.init(project={project!r}, id={(arm + '-eval')!r}, name={(arm + '-eval')!r}, group='trc-ab', resume='allow',
                 job_type='eval')
prefix = {('freya_full/' if final else 'freya_quick/')!r}
run.log({{prefix + k: v for k, v in flat.items()}} | {{'train_step': {step}}}, step={step})
run.finish()
"""
    subprocess.run([PY, "-c", code], cwd=REPO, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def push(arm, run, step, out, title, notes, slim=False):
    """Returns True once the snapshot is on the Hub."""
    if os.environ.get("NO_PUSH"):
        return False
    command = [PY, "scripts/trc/push_snapshot.py", "--run", str(run), "--repo", f"{setting('HF_ORG')}/dacvae-tts-trc-{arm}",
               "--step", str(step), "--title", title, "--notes", notes, *(["--slim"] if slim else [])]
    if out is not None and (out / "summary.json").exists():
        command += ["--eval", str(out)]
    for attempt in range(3):
        if subprocess.run(command, cwd=REPO).returncode == 0:
            return True
        time.sleep(30)
    log(f"push failed for {arm} step {step}; continuing")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="section.field=value overrides of --config")
    parser.add_argument("--init-from")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--title", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--train-args", nargs="*", default=[], help="extra `dacvae-tts train` flags")
    args = parser.parse_args()
    runs, out_root = Path(setting("RUNS")), Path(setting("OUT"))
    run, results = runs / f"trc-{args.arm}", out_root / args.arm
    config = derive(args.config, runs / "configs" / f"trc-{args.arm}.yaml", args.set)
    keep = yaml.safe_load(config.read_text())["train"].get("keep_every", 5000)
    stop = int(setting("AB_STOP"))
    steps = list(range(keep, stop + 1, keep))
    final = run / f"step-{stop:07d}.pt"
    trainer = None
    if not final.exists():
        command = ["bash", "scripts/train_2gpu.sh", str(config), setting("CACHE"), str(run),
                   "--frame-budget", setting("FRAME_BUDGET"), "--steps", setting("AB_STEPS"), "--stop-after", str(stop),
                   "--wandb-project", setting("WANDB_PROJECT"), "--wandb-group", "trc-ab", *args.train_args]
        if (run / "last.pt").exists():
            command += ["--resume", str(run / "last.pt")]
        elif args.init_from:
            command += ["--init-from", args.init_from]
        log(f"{args.arm}: training -> {run}")
        stream = open(runs / f"trc-{args.arm}.log", "a")
        trainer = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu, "TRAIN_GPUS": "1",
                                        "MASTER_PORT": str(29500 + hash(args.arm) % 1000)})
    title = args.title or f"DACVAE-TTS tr-combined A/B arm `{args.arm}`"
    notes = args.notes or f"Trained on [Codyfederer/tr-combined](https://huggingface.co/datasets/Codyfederer/tr-combined). " \
                          f"Config: `{config.name}` ({', '.join(args.set) or 'as given'}); frame budget " \
                          f"{setting('FRAME_BUDGET')}, {setting('AB_STEPS')}-update schedule stopped at {stop}."
    for step in steps:
        snapshot = run / f"step-{step:07d}.pt"
        while not snapshot.exists():
            if trainer is not None and trainer.poll() is not None:
                if not snapshot.exists():
                    log(f"{args.arm}: trainer exited ({trainer.returncode}) before step {step}")
                    sys.exit(1)
            time.sleep(20)
        is_final = step == stop
        out = results / f"step-{step:07d}"
        log(f"{args.arm}: evaluating step {step} ({'full' if is_final else 'quick'})")
        try:
            evaluate(snapshot, out, 0 if is_final else int(setting("QUICK")), args.gpu)
        except subprocess.CalledProcessError as error:
            log(f"{args.arm}: evaluation of step {step} failed ({error.returncode}); see {out}.log")
            out = None
        if out is not None:
            wandb_log(args.arm, step, out, is_final)
        pushed = push(args.arm, run, step, out, title, notes, slim=not is_final)
        if pushed and not is_final:
            snapshot.unlink()  # on the Hub (weights + EMA); local disk keeps the final snapshot only
        if is_final:
            # A second sampling seed: one 495-sentence generation cannot resolve < ~1 WER point (run C: seed 42 vs
            # 1000 differed by +0.9, a tie); compare_evals.py pools LABEL=dir,dir-s1000 as replicates.
            try:
                evaluate(snapshot, results / f"step-{step:07d}-s1000", 0, args.gpu, seed=1000)
            except subprocess.CalledProcessError as error:
                log(f"{args.arm}: seed-1000 evaluation failed ({error.returncode})")
    if trainer is not None:
        trainer.wait()
    if final.exists() and (run / "last.pt").exists():
        (run / "last.pt").unlink()  # the final snapshot is the same state
    (results / "done").touch()
    log(f"{args.arm}: done")


if __name__ == "__main__":
    main()
