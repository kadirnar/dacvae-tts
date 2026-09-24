"""Watch a training run, synthesize a frozen held-out case set for every kept checkpoint and score it.

Cases are cross-utterance pairs from the cache's validation split (unseen speakers): the prompt is one
recording of a speaker, the target text belongs to another recording of the same speaker. Scores are
corpus WER/CER (faster-whisper), speaker similarity against the codec-decoded prompt (SIM-r) and the
generated/ground-truth duration ratio. Results go to RUN/monitor.jsonl; audio to RUN/monitor/step-N/.

  python scripts/monitor.py --run runs/nano --cache data/corpus/merged --cases 48 --asr-model small.en

Evaluation protocol v2 (issue #3) is opt-in (`--protocol-v2` or the individual flags, see dacvae_tts.eval_protocol).
Its seed-tts-eval SIM is reported as `sim_r` (vs the codec-decoded prompt, always available) and `sim_o` (vs the
original prompt from `--prompt-audio`, never a codec fallback); the record then also carries the per-utterance and
v2 summary keys. `speaker_similarity` keeps its v1 meaning.
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
from dacvae_tts.eval_protocol import add_protocol_args, protocol_from_args
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
    cases, used = [], set()
    # Round-robin over speakers so a small validation split (few diarized speaker labels) still yields
    # `count` distinct (prompt, target) pairs; every target recording is used at most once.
    for _ in range(count):
        if not speakers:
            break
        speaker = speakers[len(cases) % len(speakers)] if speakers else None
        for _attempt in range(len(speakers)):
            speaker = speakers[(len(cases) + _attempt) % len(speakers)]
            free = [r for r in by_speaker[speaker] if r[1] not in used]
            if len(free) >= 1 and len(by_speaker[speaker]) >= 2:
                target = rng.choice(free)
                prompt = rng.choice([r for r in by_speaker[speaker] if r[1] != target[1]])
                break
        else:
            break
        used.add(target[1])
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
    parser.add_argument("--guidance-until", type=float, default=1.0, help="CFG only while t < this (0.5 = noisy half)")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--sway", type=float, default=-1.0)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default="en", help="Whisper language code")
    parser.add_argument("--asr-model", default="small.en")
    parser.add_argument("--asr-device", default="cpu")
    parser.add_argument(
        "--metric-normalization",
        choices=["english-unicode-v2", "legacy-ascii-v1", "turkish-v1", "turkish-v2"],
        help="WER/CER normalization (default follows --language: turkish-v1 for tr)",
    )
    parser.add_argument("--speaker-model", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument("--dnsmos", help="Path to sig_bak_ovr.onnx; adds DNSMOS SIG/BAK/OVRL per case and to the summary")
    parser.add_argument("--once", action="store_true", help="Score the checkpoints present now, then exit")
    parser.add_argument("--checkpoint", help="Score one checkpoint file instead of watching the run")
    parser.add_argument("--poll", type=int, default=120)
    parser.add_argument(
        "--prompt-audio",
        help="Directory of original prompt WAVs from export_case_audio.py; speaker_similarity is then scored "
        "against them (same wavlm-base-plus-sv model) and protocol v2 adds the real SIM-o as sim_o",
    )
    parser.add_argument("--wandb-project", help="Log scores and a few audio samples to this W&B project")
    parser.add_argument("--wandb-id", help="W&B run id to attach to (default: the run directory name)")
    parser.add_argument("--wandb-audio", type=int, default=4, help="Samples per checkpoint uploaded as audio")
    add_protocol_args(parser)
    args = parser.parse_args()
    protocol = protocol_from_args(args)

    run = Path(args.run)
    monitor = run / "monitor"
    monitor.mkdir(parents=True, exist_ok=True)
    data, cases = select_cases(args.cache, args.cases, args.seed)
    (monitor / "cases.json").write_text(json.dumps(cases, indent=1))
    print(f"{len(cases)} cases from {len({c['speaker'] for c in cases})} held-out speakers", flush=True)
    evaluator = Evaluator(
        args.asr_model,
        args.dnsmos,
        args.speaker_model,
        args.asr_device,
        metric_normalization=args.metric_normalization,
        language=args.language,
        protocol=protocol,
    )
    scored = set()
    log = run / "monitor.jsonl"
    if log.exists():
        scored = {
            (r["checkpoint"], r.get("guidance"), r.get("guidance_until", 1.0), r.get("noise_scale", 1.0), r.get("sway", -1.0), r.get("duration_scale", 1.0), r.get("sampler_steps"))
            for r in (json.loads(line) for line in log.read_text().splitlines() if line.strip())
        }
    prompts_written = False
    from dacvae_tts.tracking import Tracker

    tracker = Tracker(bool(args.wandb_project), args.wandb_project, run.name, {}, resume_id=args.wandb_id)

    while True:
        if args.checkpoint:
            checkpoints = [Path(args.checkpoint)]
        else:
            checkpoints = sorted(run.glob("step-*.pt"))
        key = lambda c: (str(c), args.guidance, args.guidance_until, args.noise_scale, args.sway, args.duration_scale, args.steps)  # noqa: E731
        pending = [c for c in checkpoints if key(c) not in scored]
        for checkpoint in pending:
            started = time.time()
            tts = Synthesizer(str(checkpoint), device=args.device)
            step = tts.checkpoint.get("step")
            tag = "" if (args.guidance, args.guidance_until, args.noise_scale, args.sway, args.duration_scale, args.steps) == (2.0, 1.0, 1.0, -1.0, 1.0, 16) else f"-g{args.guidance:g}-u{args.guidance_until:g}-n{args.noise_scale:g}-s{args.sway:g}-d{args.duration_scale:g}-k{args.steps}"
            folder = monitor / (checkpoint.stem + tag)
            folder.mkdir(parents=True, exist_ok=True)
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
                        sway=args.sway,
                        guidance_until=args.guidance_until,
                        noise_scale=args.noise_scale,
                        duration_scale=args.duration_scale,
                    )
                except ValueError as error:
                    rows.append({**case, "error": str(error)})
                    continue
                prompt_wav = codec_prompt = monitor / f"prompt-{case['uid'].replace('/', '_').replace(':', '_')}.wav"
                original = None
                if args.prompt_audio:
                    original = Path(args.prompt_audio) / (
                        case["prompt_uid"].replace("/", "_").replace(":", "_") + ".wav"
                    )
                    if original.exists():
                        prompt_wav = original
                    else:
                        original = None
                try:
                    score = evaluator.score(
                        output, case["text"], prompt_wav, original_prompt=original, codec_prompt=codec_prompt
                    )
                except (RuntimeError, ValueError, OSError) as error:  # ASR OOM, empty transcript, bad file
                    rows.append({**case, "error": f"score: {error}"[:300]})
                    continue
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
                "dnsmos_ovrl": summary.get("dnsmos_ovrl"),
                "dnsmos_sig": summary.get("dnsmos_sig"),
                "dnsmos_bak": summary.get("dnsmos_bak"),
                "duration_ratio_mean": float(np.mean([r["duration_ratio"] for r in good])) if good else None,
                "rtf": summary.get("rtf"),
                "asr_model": args.asr_model,
                "similarity_reference": "original" if args.prompt_audio else "codec_decoded",
                "guidance": args.guidance,
                "guidance_until": args.guidance_until,
                "noise_scale": args.noise_scale,
                "sway": args.sway,
                "duration_scale": args.duration_scale,
                "sampler_steps": args.steps,
                "seconds": time.time() - started,
            }
            if protocol is not None:  # v2: per-utterance WER/CER, S/D/I, sim_o/sim_r, UTMOS, signal statistics
                record.update({k: v for k, v in summary.items() if k not in record and k != "count"})
                record["protocol"] = evaluator.identity["protocol"]["options"]
            with open(log, "a") as stream:
                stream.write(json.dumps(record) + "\n")
            tracker.log(record, step=step, prefix="monitor/")
            for index, row in enumerate(rows[: args.wandb_audio]):
                if "error" in row:
                    continue
                tracker.audio(
                    f"monitor/audio/{index:03d}",
                    folder / f"{index:03d}.wav",
                    f"step {step} | WER {row['wer']:.2f} | {row['text'][:80]}",
                    48000,
                    step=step,
                )
            scored.add(key(checkpoint))
            print(json.dumps(record), flush=True)
            del tts
            torch.cuda.empty_cache()
        if args.once or args.checkpoint:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
