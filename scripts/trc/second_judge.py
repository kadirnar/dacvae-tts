"""Rescore eval_sentences.py outputs with an ASR judge from another model family (issues #12/#13: best-of-N).

Best-of-N picks candidates with Whisper-large-v3-turbo and the protocol judges them with Whisper-large-v3: shared
Whisper preferences can inflate the gain. This rescoring uses Meta's MMS-1b-all CTC model with its Turkish adapter
(wav2vec 2.0, trained on other data, no autoregressive decoder), the same turkish-v2 metric normalization and the
same speakers, and writes DIR/judge-mms/results.jsonl in the eval_sentences.py row format, so compare_evals.py pairs
it like any run:

  python scripts/trc/second_judge.py outputs/trc/inference-runc/base outputs/trc/inference-runc/bo3
  python scripts/compare_evals.py base=outputs/.../base/judge-mms bo3=outputs/.../bo3/judge-mms
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from dacvae_tts.metrics import error_counts

MODEL = "facebook/mms-1b-all"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normalization", default="turkish-v2")
    args = parser.parse_args()
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    processor = AutoProcessor.from_pretrained(MODEL, target_lang="tur")
    model = Wav2Vec2ForCTC.from_pretrained(MODEL, target_lang="tur", ignore_mismatched_sizes=True)
    model = model.to(args.device).eval()
    for directory in map(Path, args.dirs):
        rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines() if line.strip()]
        output = directory / "judge-mms"
        output.mkdir(exist_ok=True)
        scored = []
        for row in rows:
            audio, rate = sf.read(directory / f"{row['id']}.wav", dtype="float32", always_2d=True)
            audio = audio.mean(1)
            if rate != 16000:
                g = np.gcd(rate, 16000)
                audio = resample_poly(audio, 16000 // g, rate // g).astype(np.float32)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                ids = model(**inputs).logits.argmax(-1)[0]
            hypothesis = processor.decode(ids)
            counts = error_counts(row["text"], hypothesis, args.normalization)
            scored.append({"id": row["id"], "text": row["text"], "speaker": row["speaker"], "prompt": row.get("prompt"),
                           "hypothesis": hypothesis, **counts,
                           "evaluator": {"asr_backend": "transformers-ctc", "asr_model": MODEL,
                                         "metric_normalization": args.normalization, "language": "tur"}})
        (output / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in scored) + "\n")
        words = sum(r["words"] for r in scored)
        chars = sum(r["chars"] for r in scored)
        summary = {"wer": sum(r["word_edits"] for r in scored) / words, "cer": sum(r["char_edits"] for r in scored) / chars,
                   "count": len(scored), "asr_model": MODEL}
        (output / "summary.json").write_text(json.dumps(summary, indent=1))
        print(directory.name, json.dumps(summary))


if __name__ == "__main__":
    main()
