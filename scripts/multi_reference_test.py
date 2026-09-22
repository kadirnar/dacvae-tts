"""Synthesize emotionally loaded texts with many different reference voices and score every sample.

Input: a JSON list of prompts [{"name", "audio", "text", "tag"}, ...] and a JSON list of target texts
[{"name", "text"}, ...]. Every prompt is combined with every text. Output: WAVs under OUTPUT/<prompt>/,
WER/CER (Whisper-large-v3), DNSMOS (official ONNX) and speaker similarity against the original
prompt (dev model), per sample and aggregated per prompt.

  python scripts/multi_reference_test.py --checkpoint checkpoints/nano-v2-40k.pt --prompts prompts.json \
      --texts texts.json --dnsmos /path/sig_bak_ovr.onnx --output outputs/emotion-test
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dacvae_tts.codec import read_audio
from dacvae_tts.inference import Synthesizer
from dacvae_tts.metrics import DNSMOS, Evaluator, error_counts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--texts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dnsmos", help="Path to the official sig_bak_ovr.onnx")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default="en", help="Whisper language code")
    parser.add_argument("--asr", default="openai/whisper-large-v3")
    parser.add_argument("--speaker-model", default="microsoft/wavlm-base-plus-sv")
    args = parser.parse_args()

    prompts = json.loads(Path(args.prompts).read_text())
    texts = json.loads(Path(args.texts).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tts = Synthesizer(args.checkpoint, device=args.device)
    rows = []
    started = time.time()
    for prompt in prompts:
        folder = output / prompt["name"]
        folder.mkdir(exist_ok=True)
        voice = tts.prepare_reference(prompt["audio"], prompt["text"])
        for number, target in enumerate(texts):
            path = folder / f"{target['name']}.wav"
            try:
                result = tts.synthesize(
                    target["text"],
                    reference=voice,
                    output=path,
                    steps=args.steps,
                    guidance=args.guidance,
                    seed=args.seed + number,
                )
                rows.append(
                    {
                        "prompt": prompt["name"],
                        "prompt_tag": prompt.get("tag", ""),
                        "prompt_audio": prompt["audio"],
                        "prompt_text": prompt["text"],
                        "target": target["name"],
                        "text": target["text"],
                        "file": str(path),
                        "audio_seconds": round(result.metadata["audio_seconds"], 2),
                    }
                )
            except ValueError as error:
                rows.append(
                    {
                        "prompt": prompt["name"],
                        "target": target["name"],
                        "text": target["text"],
                        "error": str(error),
                    }
                )
    print(f"synthesized {len(rows)} samples in {time.time() - started:.0f}s", flush=True)
    del tts
    torch.cuda.empty_cache()

    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector, pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model=args.asr,
        device=0 if args.device == "cuda" else -1,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    similarity = Evaluator.__new__(Evaluator)
    similarity.device = torch.device(args.device)
    similarity.extractor = AutoFeatureExtractor.from_pretrained(args.speaker_model)
    similarity.speaker = AutoModelForAudioXVector.from_pretrained(args.speaker_model).to(args.device).eval()
    dnsmos = DNSMOS(args.dnsmos) if args.dnsmos else None
    prompt_cache = {}
    for row in rows:
        if "error" in row:
            continue
        audio = read_audio(row["file"], 16000)
        row["hypothesis"] = asr(
            audio.numpy(), generate_kwargs={"language": args.language, "task": "transcribe"}
        )["text"].strip()
        row.update(error_counts(row["text"], row["hypothesis"]))
        if row["prompt_audio"] not in prompt_cache:
            prompt_wave = read_audio(row["prompt_audio"], 16000)
            prompt_cache[row["prompt_audio"]] = {
                "embedding": similarity.embedding(prompt_wave),
                **({f"prompt_{k}": v for k, v in dnsmos(prompt_wave.numpy()).items()} if dnsmos else {}),
            }
        cached = prompt_cache[row["prompt_audio"]]
        row["speaker_similarity"] = float((similarity.embedding(audio) * cached["embedding"]).sum())
        if dnsmos:
            row.update(dnsmos(audio.numpy()))
            row.update({k: v for k, v in cached.items() if k.startswith("prompt_")})

    def aggregate(subset):
        good = [r for r in subset if "error" not in r]
        if not good:
            return {"count": 0, "failed": len(subset)}
        result = {
            "count": len(good),
            "failed": len(subset) - len(good),
            "wer": round(sum(r["word_edits"] for r in good) / sum(r["words"] for r in good), 4),
            "cer": round(sum(r["char_edits"] for r in good) / sum(r["chars"] for r in good), 4),
            "speaker_similarity": round(float(np.mean([r["speaker_similarity"] for r in good])), 4),
        }
        if dnsmos:
            for key in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"):
                result[key] = round(float(np.mean([r[key] for r in good])), 3)
            result["prompt_dnsmos_ovrl"] = round(float(np.mean([r["prompt_dnsmos_ovrl"] for r in good])), 3)
        return result

    summary = {
        "overall": aggregate(rows),
        "per_prompt": {
            p["name"]: {"tag": p.get("tag", ""), **aggregate([r for r in rows if r["prompt"] == p["name"]])}
            for p in prompts
        },
        "per_text": {t["name"]: aggregate([r for r in rows if r["target"] == t["name"]]) for t in texts},
    }
    report = {
        "checkpoint": args.checkpoint,
        "steps": args.steps,
        "guidance": args.guidance,
        "asr": args.asr,
        "speaker_model": args.speaker_model,
        "dnsmos": args.dnsmos,
        "summary": summary,
        "samples": rows,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
