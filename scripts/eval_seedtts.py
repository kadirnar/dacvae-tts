"""Zero-shot evaluation on the Seed-TTS test-en list (1,088 Common Voice prompts + target texts).

Synthesizes every case, then scores WER with Whisper-large-v3 (transformers) using the Seed-TTS-eval
text convention (lower case, punctuation removed except apostrophes; both corpus-level and
mean-per-utterance WER are reported) and speaker similarity with the repository's evaluator. The
similarity model here is not the WavLM-large ECAPA checkpoint used by published SIM-o numbers, so
treat that column as a development metric; `--sim-o` adds the published metric (dacvae_tts.sim_o,
seed-tts-eval's WavLM-Large ECAPA-TDNN) against the original prompt wav as `sim_o`.

  python scripts/eval_seedtts.py --checkpoint runs/nano/step-0200000.pt \
      --meta seedtts_testset/en/meta.lst --audio-root seedtts_testset --output outputs/seedtts-nano
"""

import argparse
import json
import string
import time
from pathlib import Path

import numpy as np
import torch

from dacvae_tts.codec import read_audio
from dacvae_tts.inference import Synthesizer
from dacvae_tts.metrics import word_edit_counts

# seed-tts-eval run_wer.py deletes every character of zhon.hanzi.punctuation + string.punctuation except the ASCII
# apostrophe (so "twenty-one" is one word and "don’t" becomes "dont"), lower-cases, and jiwer splits at whitespace.
# zhon.hanzi.punctuation (zhon 2.x), copied to avoid the dependency: CJK/fullwidth marks, dashes, curly quotes, ….
HANZI_PUNCTUATION = (
    "\uFF02\uFF03\uFF04\uFF05\uFF06\uFF07\uFF08\uFF09\uFF0A\uFF0B\uFF0C\uFF0D"
    "\uFF0F\uFF1A\uFF1B\uFF1C\uFF1D\uFF1E\uFF20\uFF3B\uFF3C\uFF3D\uFF3E\uFF3F"
    "\uFF40\uFF5B\uFF5C\uFF5D\uFF5E\uFF5F\uFF60\uFF62\uFF63\uFF64\u3000\u3001\u3003"
    "\u3008\u3009\u300A\u300B\u300C\u300D\u300E\u300F\u3010\u3011"
    "\u3014\u3015\u3016\u3017\u3018\u3019\u301A\u301B\u301C\u301D\u301E\u301F"
    "\u3030\u303E\u303F\u2013\u2014\u2018\u2019\u201B\u201C\u201D\u201E\u201F\u2026\u2027\uFE4F"
    "\uFE51\uFE54\u00B7\uFF0E\uFF01\uFF1F\uFF61\u3002"
)
SEED_PUNCTUATION = str.maketrans("", "", HANZI_PUNCTUATION + string.punctuation.replace("'", ""))


def seed_normalize(text):
    """seed-tts-eval English WER text: punctuation deleted (not replaced by a space), apostrophes kept, lower case."""
    return " ".join(text.translate(SEED_PUNCTUATION).lower().split())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--meta", required=True, help="Seed-TTS meta.lst: id|prompt_text|prompt_wav|text")
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default="en", help="Whisper language code")
    parser.add_argument("--asr", default="openai/whisper-large-v3")
    parser.add_argument("--asr-device", default="cuda")
    parser.add_argument("--speaker-model", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument("--skip-synthesis", action="store_true", help="Only score existing WAVs")
    parser.add_argument("--sim-o", action="store_true",
                        help="Also score SIM-o (seed-tts-eval WavLM-Large ECAPA-TDNN) against the original prompt")
    parser.add_argument("--sim-o-checkpoint", help="Local wavlm_large_finetune.pth (default: pinned HF mirror)")
    parser.add_argument("--sim-o-backend", choices=["transformers", "s3prl"], default="transformers")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for line in Path(args.meta).read_text().splitlines():
        if not line.strip():
            continue
        utt, prompt_text, prompt_wav, text = line.split("|")
        cases.append(
            dict(
                id=utt, prompt_text=prompt_text, prompt_wav=str(Path(args.audio_root) / prompt_wav), text=text
            )
        )
    cases = cases[: args.limit] if args.limit else cases

    if not args.skip_synthesis:
        tts = Synthesizer(args.checkpoint, device=args.device)
        started = time.time()
        for number, case in enumerate(cases):
            target = output / f"{case['id']}.wav"
            if target.exists():
                continue
            try:
                result = tts.synthesize(
                    case["text"],
                    ref_audio=case["prompt_wav"],
                    reference_text=case["prompt_text"],
                    output=target,
                    steps=args.steps,
                    guidance=args.guidance,
                    seed=args.seed,
                )
                case["audio_seconds"] = result.metadata["audio_seconds"]
                case["rtf"] = result.metadata["rtf"]
            except ValueError as error:
                case["error"] = str(error)
            if number % 50 == 0:
                print(f"synthesized {number + 1}/{len(cases)} in {time.time() - started:.0f}s", flush=True)
        del tts
        torch.cuda.empty_cache()

    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model=args.asr,
        device=0 if args.asr_device == "cuda" else -1,
        torch_dtype=torch.float16 if args.asr_device == "cuda" else torch.float32,
    )
    from dacvae_tts.metrics import Evaluator

    similarity = Evaluator.__new__(Evaluator)  # only the speaker branch is needed
    similarity.device = torch.device(args.asr_device)
    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

    similarity.extractor = AutoFeatureExtractor.from_pretrained(args.speaker_model)
    similarity.speaker = (
        AutoModelForAudioXVector.from_pretrained(args.speaker_model).to(similarity.device).eval()
    )

    sim_o = None
    if args.sim_o:
        from dacvae_tts.sim_o import SimO

        sim_o = SimO(args.sim_o_checkpoint, args.sim_o_backend, args.asr_device)
    rows, edits, words, per_utterance, sims, sims_o = [], 0, 0, [], [], []
    for case in cases:
        target = output / f"{case['id']}.wav"
        if not target.exists():
            rows.append({**case, "error": case.get("error", "missing audio")})
            continue
        audio = read_audio(target, 16000)
        hypothesis = asr(audio.numpy(), generate_kwargs={"language": args.language, "task": "transcribe"})[
            "text"
        ]
        reference, predicted = seed_normalize(case["text"]).split(), seed_normalize(hypothesis).split()
        distance = word_edit_counts(reference, predicted)[0]
        edits += distance
        words += len(reference)
        per_utterance.append(distance / max(len(reference), 1))
        prompt = read_audio(case["prompt_wav"], 16000)
        with torch.inference_mode():
            sim = float((similarity.embedding(audio) * similarity.embedding(prompt)).sum())
        sims.append(sim)
        row = {**case, "hypothesis": hypothesis, "wer": per_utterance[-1], "speaker_similarity": sim}
        if sim_o is not None:  # generated audio (no prompt frames) vs the ORIGINAL prompt recording
            row["sim_o"] = sim_o.similarity(target, case["prompt_wav"])
            sims_o.append(row["sim_o"])
        rows.append(row)
    (output / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary = {
        "checkpoint": args.checkpoint,
        "cases": len(cases),
        "scored": len(per_utterance),
        "failed": len(cases) - len(per_utterance),
        "wer_corpus": edits / max(words, 1),
        "wer_mean_per_utterance": float(np.mean(per_utterance)) if per_utterance else None,
        "speaker_similarity_dev_model": float(np.mean(sims)) if sims else None,
        "similarity_model": args.speaker_model,
        "asr": args.asr,
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "text_normalization": "seed-tts-eval: lower case, punctuation removed except apostrophes",
    }
    if sim_o is not None:
        summary.update(sim_o=float(np.mean(sims_o)) if sims_o else None, sim_o_model=sim_o.identity)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
