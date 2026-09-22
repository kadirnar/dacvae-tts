"""Synthesize arbitrary sentences and random word lists with a checkpoint, transcribe them and score WER.

  python scripts/random_word_test.py --checkpoint checkpoints/nano-v2-40k.pt \
      --prompt data/seedtts/en/prompt-wavs/common_voice_en_10119832.wav \
      --prompt-text "We asked over twenty different people, and they all said it was his." --output outputs/random-test
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch

from dacvae_tts.codec import read_audio
from dacvae_tts.inference import Synthesizer
from dacvae_tts.metrics import word_edit_counts

SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Please remember to water the plants before you leave for the airport tomorrow morning.",
    "Scientists discovered a new species of deep sea jellyfish that glows bright green.",
    "My grandmother's recipe for apple pie uses cinnamon, nutmeg, and a pinch of salt.",
    "The train to Edinburgh was delayed by almost an hour because of heavy snow.",
    "Honestly, I never expected the meeting to end with everyone singing happy birthday.",
    "Turn left at the old library, then follow the road until you see a yellow house.",
    "Neural networks learn patterns from data instead of following hand written rules.",
]

WORDS = (
    "banana telescope whisper granite umbrella velvet cactus lantern marble thunder pillow "
    "compass ribbon walnut engine meadow silver puzzle harbor candle dragon falcon garden hammer "
    "island jacket kettle ladder magnet needle orange parcel quiver rocket saddle tunnel violin "
    "window yogurt zipper anchor blanket crystal dolphin feather glacier helmet ivory jungle"
).split()


def normalize(text):
    import re

    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True, help="Prompt WAV")
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--random-lists", type=int, default=6)
    parser.add_argument("--words-per-list", type=int, default=6)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--asr", default="openai/whisper-large-v3")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    texts = [("sentence", s) for s in SENTENCES]
    for _ in range(args.random_lists):
        words = rng.sample(WORDS, args.words_per_list)
        texts.append(("random_words", " ".join(words).capitalize() + "."))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tts = Synthesizer(args.checkpoint, device=args.device)
    voice = tts.prepare_reference(args.prompt, args.prompt_text)
    rows = []
    for number, (kind, text) in enumerate(texts):
        started = time.time()
        result = tts.synthesize(
            text,
            reference=voice,
            output=output / f"{number:02d}-{kind}.wav",
            steps=args.steps,
            guidance=args.guidance,
            duration_scale=args.duration_scale,
            seed=args.seed + number,
        )
        rows.append(
            {
                "file": f"{number:02d}-{kind}.wav",
                "kind": kind,
                "text": text,
                "audio_seconds": round(result.metadata["audio_seconds"], 2),
                "rtf": round(result.metadata["rtf"], 3),
                "wall_seconds": round(time.time() - started, 2),
            }
        )
    del tts
    torch.cuda.empty_cache()

    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model=args.asr,
        device=0 if args.device == "cuda" else -1,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    for row in rows:
        audio = read_audio(output / row["file"], 16000)
        row["hypothesis"] = asr(audio.numpy(), generate_kwargs={"language": "en", "task": "transcribe"})[
            "text"
        ].strip()
        reference, hypothesis = normalize(row["text"]).split(), normalize(row["hypothesis"]).split()
        row["word_errors"], row["words"] = word_edit_counts(reference, hypothesis)[0], len(reference)
        row["wer"] = round(row["word_errors"] / len(reference), 3)
    summary = {}
    for kind in ("sentence", "random_words"):
        subset = [r for r in rows if r["kind"] == kind]
        summary[kind] = {
            "count": len(subset),
            "wer_corpus": round(sum(r["word_errors"] for r in subset) / sum(r["words"] for r in subset), 3),
        }
    report = {
        "checkpoint": args.checkpoint,
        "prompt": args.prompt,
        "prompt_text": args.prompt_text,
        "steps": args.steps,
        "guidance": args.guidance,
        "duration_scale": args.duration_scale,
        "asr": args.asr,
        "summary": summary,
        "samples": rows,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for row in rows:
        print(f"[{row['kind']:12}] WER {row['wer']:.2f} | {row['text']}\n{'':16}ASR: {row['hypothesis']}")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
