"""Codec/ASR ceiling on the validation split: decode cached latents through DACVAE and score them.

Tells how much of the measured WER is due to Whisper + the codec rather than the TTS model, and gives the
DNSMOS of real (codec-reconstructed) speech as the quality reference. Uses the same case selection as monitor.py.

  python scripts/eval_ceiling.py --cache data/tr55/merged --cases 48 --output outputs/ceiling --dnsmos models/sig_bak_ovr.onnx

Protocol v2 (issue #3, opt-in): with `--protocol-v2 --prompt-audio DIR` (original recordings exported with
`export_case_audio.py --cases OUTPUT/cases.json`) the codec-resynthesis ceiling also gets sim_o against the original
prompt and sim_r against the decoded one; `--real-audio` scores the original target recordings themselves (the
real-speech ceiling of SIM-o, WER and UTMOS).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor import select_cases  # noqa: E402

from dacvae_tts.codec import Codec  # noqa: E402
from dacvae_tts.data import load_stats  # noqa: E402
from dacvae_tts.eval_protocol import add_protocol_args, protocol_from_args  # noqa: E402
from dacvae_tts.metrics import METRIC_NORMALIZATIONS, Evaluator, summarize  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cases", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default="tr")
    parser.add_argument(
        "--metric-normalization", choices=METRIC_NORMALIZATIONS,
        help="WER/CER text normalization (default: turkish-v1 for --language tr, else english-unicode-v2)",
    )
    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument("--asr-device", default="cuda")
    parser.add_argument("--dnsmos")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-audio", help="Original recordings (export_case_audio.py) for sim_o / --real-audio")
    parser.add_argument("--real-audio", action="store_true",
                        help="Score the original target recordings instead of their codec resynthesis")
    add_protocol_args(parser)
    args = parser.parse_args()
    protocol = protocol_from_args(args)
    if args.real_audio and not args.prompt_audio:
        parser.error("--real-audio needs --prompt-audio")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data, cases = select_cases(args.cache, args.cases, args.seed)
    (out / "cases.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False))  # input of export_case_audio.py
    meta = json.loads((Path(args.cache) / "metadata.json").read_text())
    stats = load_stats(args.cache)
    mean, std = stats["mean"].to(args.device), stats["std"].to(args.device)
    codec = Codec(meta["checkpoint"], args.device, loudness=meta.get("loudness_lufs"))
    evaluator = Evaluator(args.asr_model, args.dnsmos, "microsoft/wavlm-base-plus-sv", args.asr_device,
                          metric_normalization=args.metric_normalization, language=args.language, protocol=protocol)

    def original(uid):
        path = Path(args.prompt_audio) / (uid.replace("/", "_").replace(":", "_") + ".wav") if args.prompt_audio else None
        return path if path is not None and path.exists() else None

    rows = []
    for number, case in enumerate(cases):
        path = out / f"{number:03d}.wav"
        if args.real_audio:
            path = original(case["uid"])
            if path is None:
                print(number, "skipped: no original target recording", flush=True)
                continue
        else:
            target = data.row(case["target_index"])["latents"].to(args.device)
            sf.write(path, codec.decode(target * std + mean).numpy(), codec.sample_rate)
        prompt = data.row(case["prompt_index"])["latents"].to(args.device)
        prompt_path = out / f"prompt-{number:03d}.wav"
        sf.write(prompt_path, codec.decode(prompt * std + mean).numpy(), codec.sample_rate)
        score = evaluator.score(path, case["text"], prompt_path, original_prompt=original(case["prompt_uid"]),
                                codec_prompt=prompt_path)
        rows.append({**case, **score, "evaluator": evaluator.row_identity})  # compact scorer identity
        print(number, f"wer={score['wer']:.2f}", case["text"][:60], "|", score["hypothesis"][:60], flush=True)
    (out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary = summarize(rows)
    summary["per_case_wer_median"] = float(np.median([r["wer"] for r in rows]))
    summary["metric_normalization"] = evaluator.metric_normalization
    if protocol is not None:
        summary.update(protocol=evaluator.identity["protocol"], audio="original" if args.real_audio else "codec")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
