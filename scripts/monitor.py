"""Watch a training run, synthesize a frozen held-out case set for every kept checkpoint and score it.

Cases are cross-utterance pairs from the cache's validation split (unseen speakers): the prompt is one
recording of a speaker, the target text belongs to another recording of the same speaker. Scores are
corpus WER/CER (faster-whisper), speaker similarity against the codec-decoded prompt (SIM-r) and the
generated/ground-truth duration ratio. Results go to RUN/monitor.jsonl; audio to RUN/monitor/step-N/.

  python scripts/monitor.py --run runs/nano --cache data/corpus/merged --cases 48 --asr-model small.en
"""

import argparse
import json
import random
import sqlite3
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dacvae_tts.data import LatentDataset
from dacvae_tts.inference import Synthesizer, VoiceReference
from dacvae_tts.metrics import Evaluator, summarize


def select_cases(cache, count, seed, min_frames=75, max_frames=375):
    """Deterministic (prompt, target) pairs of distinct validation recordings per speaker."""
    data = LatentDataset(cache, "val", pairing="within", layout="joined")
    db = sqlite3.connect(f"file:{Path(cache) / 'index.sqlite'}?mode=ro", uri=True)
    rows = db.execute(
        "SELECT id, uid, speaker, text, frames FROM samples WHERE split='val' AND frames BETWEEN ? AND ? "
        "ORDER BY speaker, id",
        (min_frames, max_frames),
    ).fetchall()
    by_speaker = {}
    for row in rows:
        by_speaker.setdefault(row[2], []).append(row)
    speakers = sorted(s for s, items in by_speaker.items() if len(items) >= 2)
    rng = random.Random(seed)
    rng.shuffle(speakers)
    index = {int(v): i for i, v in enumerate(data.ids)}
    cases = []
    for speaker in speakers:
        prompt, target = rng.sample(by_speaker[speaker], 2)
        cases.append(
            {
                "speaker": speaker,
                "prompt_uid": prompt[1],
                "prompt_text": prompt[3],
                "prompt_index": index[int(prompt[0])],
                "uid": target[1],
                "text": target[3],
                "target_index": index[int(target[0])],
                "ground_truth_seconds": target[4] / 25,
            }
        )
        if len(cases) >= count:
            break
    return data, cases


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cases", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--asr-model", default="small.en")
    parser.add_argument("--asr-device", default="cpu")
    parser.add_argument("--speaker-model", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument("--once", action="store_true", help="Score the checkpoints present now, then exit")
    parser.add_argument("--checkpoint", help="Score one checkpoint file instead of watching the run")
    parser.add_argument("--poll", type=int, default=120)
    parser.add_argument(
        "--prompt-audio",
        help="Directory of original prompt WAVs from export_case_audio.py; scores become SIM-o",
    )
    args = parser.parse_args()

    run = Path(args.run)
    monitor = run / "monitor"
    monitor.mkdir(parents=True, exist_ok=True)
    data, cases = select_cases(args.cache, args.cases, args.seed)
    (monitor / "cases.json").write_text(json.dumps(cases, indent=1))
    print(f"{len(cases)} cases from {len({c['speaker'] for c in cases})} held-out speakers", flush=True)
    evaluator = Evaluator(args.asr_model, None, args.speaker_model, args.asr_device)
    scored = set()
    log = run / "monitor.jsonl"
    if log.exists():
        scored = {json.loads(line)["checkpoint"] for line in log.read_text().splitlines() if line.strip()}
    prompts_written = False

    while True:
        if args.checkpoint:
            checkpoints = [Path(args.checkpoint)]
        else:
            checkpoints = sorted(run.glob("step-*.pt"))
        pending = [c for c in checkpoints if str(c) not in scored]
        for checkpoint in pending:
            started = time.time()
            tts = Synthesizer(str(checkpoint), device=args.device)
            step = tts.checkpoint.get("step")
            folder = monitor / checkpoint.stem
            folder.mkdir(exist_ok=True)
            if not prompts_written:
                for case in cases:  # decoded once: the same prompt audio for every checkpoint
                    latents = data.row(case["prompt_index"])["latents"].to(tts.device)
                    wave = tts.codec.decode(latents * tts.std + tts.mean)
                    sf.write(
                        monitor / f"prompt-{case['uid'].replace('/', '_').replace(':', '_')}.wav",
                        wave.numpy(),
                        tts.codec.sample_rate,
                    )
                prompts_written = True
            rows = []
            for number, case in enumerate(cases):
                prompt = data.row(case["prompt_index"])
                voice = VoiceReference(prompt["latents"], case["prompt_text"], "cache", {})
                output = folder / f"{number:03d}.wav"
                try:
                    result = tts.synthesize(
                        case["text"],
                        reference=voice,
                        output=output,
                        steps=args.steps,
                        guidance=args.guidance,
                        seed=args.seed + number,
                    )
                except ValueError as error:
                    rows.append({**case, "error": str(error)})
                    continue
                prompt_wav = monitor / f"prompt-{case['uid'].replace('/', '_').replace(':', '_')}.wav"
                if args.prompt_audio:
                    original = Path(args.prompt_audio) / (
                        case["prompt_uid"].replace("/", "_").replace(":", "_") + ".wav"
                    )
                    if original.exists():
                        prompt_wav = original
                score = evaluator.score(output, case["text"], prompt_wav)
                rows.append(
                    {
                        **case,
                        **{k: v for k, v in score.items() if k != "evaluator"},
                        "audio_seconds": result.metadata["audio_seconds"],
                        "duration_ratio": result.metadata["audio_seconds"] / case["ground_truth_seconds"],
                        "rtf": result.metadata["rtf"],
                    }
                )
            (folder / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            good = [r for r in rows if "error" not in r]
            summary = summarize(good) if good else {"count": 0}
            record = {
                "checkpoint": str(checkpoint),
                "step": step,
                "cases": len(rows),
                "failed": len(rows) - len(good),
                "wer": summary.get("wer"),
                "cer": summary.get("cer"),
                "speaker_similarity": summary.get("speaker_similarity"),
                "duration_ratio_mean": float(np.mean([r["duration_ratio"] for r in good])) if good else None,
                "rtf": summary.get("rtf"),
                "asr_model": args.asr_model,
                "similarity_reference": "original" if args.prompt_audio else "codec_decoded",
                "guidance": args.guidance,
                "sampler_steps": args.steps,
                "seconds": time.time() - started,
            }
            with open(log, "a") as stream:
                stream.write(json.dumps(record) + "\n")
            scored.add(str(checkpoint))
            print(json.dumps(record), flush=True)
            del tts
            torch.cuda.empty_cache()
        if args.once or args.checkpoint:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
