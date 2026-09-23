"""Codec/ASR ceiling on the validation split: decode cached latents through DACVAE and score them.

Tells how much of the measured WER is due to Whisper + the codec rather than the TTS model, and gives the
DNSMOS of real (codec-reconstructed) speech as the quality reference. Uses the same case selection as monitor.py.

  python scripts/eval_ceiling.py --cache data/tr55/merged --cases 48 --output outputs/ceiling --dnsmos models/sig_bak_ovr.onnx
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor import select_cases  # noqa: E402

from dacvae_tts.codec import Codec  # noqa: E402
from dacvae_tts.data import load_stats  # noqa: E402
from dacvae_tts.metrics import Evaluator, summarize  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cases", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default="tr")
    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument("--asr-device", default="cuda")
    parser.add_argument("--dnsmos")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data, cases = select_cases(args.cache, args.cases, args.seed)
    meta = json.loads((Path(args.cache) / "metadata.json").read_text())
    stats = load_stats(args.cache)
    mean, std = stats["mean"].to(args.device), stats["std"].to(args.device)
    codec = Codec(meta["checkpoint"], args.device, loudness=meta.get("loudness_lufs"))
    evaluator = Evaluator(args.asr_model, args.dnsmos, "microsoft/wavlm-base-plus-sv", args.asr_device, language=args.language)
    rows = []
    for number, case in enumerate(cases):
        target = data.row(case["target_index"])["latents"].to(args.device)
        wave = codec.decode(target * std + mean)
        path = out / f"{number:03d}.wav"
        sf.write(path, wave.numpy(), codec.sample_rate)
        prompt = data.row(case["prompt_index"])["latents"].to(args.device)
        prompt_path = out / f"prompt-{number:03d}.wav"
        sf.write(prompt_path, codec.decode(prompt * std + mean).numpy(), codec.sample_rate)
        score = evaluator.score(path, case["text"], prompt_path)
        rows.append({**case, **{k: v for k, v in score.items() if k != "evaluator"}})
        print(number, f"wer={score['wer']:.2f}", case["text"][:60], "|", score["hypothesis"][:60], flush=True)
    (out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary = summarize(rows)
    summary["per_case_wer_median"] = float(np.median([r["wer"] for r in rows]))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
