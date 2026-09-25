"""Real-speech and codec ceilings of the leak-free Common Voice protocol (issue #3).

For each of the prompt set's speakers, up to --per-speaker other validated Common Voice test recordings of that speaker
(reading other sentences: not the prompt clip, no digits, no Freya-TR-Eval sentence) are scored exactly like generated
speech in eval_sentences.py: Whisper large-v3 WER/CER against the Common Voice sentence, DNSMOS, UTMOS, clipping and
SIM-o against the speaker's prompt recording. `real/` scores the recordings as they are, `codec/` their DACVAE round
trip (-16 LUFS, encode, decode) -- what a perfect latent generator could reach at best. The rows keep the
eval_sentences.py format (id, speaker, prompt ...), so compare_evals.py reads them.

  python scripts/trc/cv_ceiling.py --parquet data/cv17/tr/test --prompt-set data/eval/cv-tr-prompts/prompts.json \
      --exclude-sentences data/eval/freya_tr_eval.jsonl --output outputs/trc/ceiling --dnsmos models/sig_bak_ovr.onnx \
      --protocol-v2
"""

import argparse
import io
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from make_prompt_set import parquet_rows  # noqa: E402

from dacvae_tts.codec import Codec, read_audio  # noqa: E402
from dacvae_tts.datafilters import load_sentences, match_key  # noqa: E402
from dacvae_tts.eval_protocol import add_protocol_args, protocol_from_args  # noqa: E402
from dacvae_tts.metrics import Evaluator, summarize  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", nargs="+", required=True)
    parser.add_argument("--prompt-set", required=True)
    parser.add_argument("--exclude-sentences", nargs="*", default=[])
    parser.add_argument("--per-speaker", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dnsmos")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--codec", default="facebook/dacvae-watermarked")
    add_protocol_args(parser)
    args = parser.parse_args()
    prompts = json.loads(Path(args.prompt_set).read_text())
    root = Path(args.prompt_set).parent
    by_speaker = {p["speaker"]: (f"prompt-{i:02d}.wav", p) for i, p in enumerate(prompts)}
    excluded = {match_key(t) for path in args.exclude_sentences for t in load_sentences(path)}
    chosen = {}
    for row in parquet_rows(args.parquet):
        speaker = "cv/" + row["client_id"][:16]
        if speaker not in by_speaker or row["clip"] == by_speaker[speaker][1]["uid"]:
            continue
        sentence = row["sentence"]
        if (row["down_votes"] > 0 or any(c.isdigit() for c in sentence) or len(sentence.split()) < 3
                or match_key(sentence) in excluded):
            continue
        clips = chosen.setdefault(speaker, [])
        if len(clips) < args.per_speaker:
            clips.append(row)
    out = Path(args.output)
    codec = Codec(args.codec, args.device)
    evaluator = Evaluator("large-v3", args.dnsmos, device=args.device, metric_normalization="turkish-v2",
                          language="tr", protocol=protocol_from_args(args))
    for kind in ("real", "codec"):
        (out / kind).mkdir(parents=True, exist_ok=True)
        rows = []
        for speaker, clips in sorted(chosen.items()):
            prompt_file, prompt = by_speaker[speaker]
            for number, row in enumerate(clips):
                name = f"{speaker[3:]}-{number}"
                path = out / kind / f"{name}.wav"
                if kind == "real":
                    audio, rate = sf.read(io.BytesIO(row["bytes"]), dtype="float32", always_2d=True)
                    sf.write(path, audio.mean(1), rate, subtype="FLOAT")
                else:
                    wave = read_audio(io.BytesIO(row["bytes"]), codec.sample_rate, -16.0).to(args.device)
                    with torch.inference_mode():
                        decoded = codec.decode(codec.encode(wave))
                    sf.write(path, decoded.float().cpu().numpy(), codec.sample_rate, subtype="FLOAT")
                score = evaluator.score(path, row["sentence"], original_prompt=root / prompt["audio"])
                rows.append({"id": name, "text": row["sentence"], "speaker": speaker, "prompt": prompt_file,
                             "audio": str(path), **score})
        (out / kind / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        summary = summarize(rows)
        (out / kind / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        print(kind, json.dumps({k: summary[k] for k in ("count", "wer", "cer", "sim_o", "dnsmos_ovrl", "utmos")
                                if k in summary}))


if __name__ == "__main__":
    main()
