"""Leak-free prompt voices for `scripts/eval_sentences.py --prompt-set`, from Common Voice test speakers.

Why: the cache prompts of eval_sentences.py/monitor.py are the held-out labels of the podcast corpus, and labels are
per-episode diarization (`<episode>_speaker_k`). An acoustic audit of the Freya-TR-Eval prompt set of run C
(WavLM-large ECAPA centroids, the SIM-o model) found train labels of the same shows with cosine 0.90-0.97 for 6 of
its 10 speakers, within the same-label split-half range (median 0.94; other-show controls at most 0.60): the same
hosts recorded in other episodes. "Unseen speaker" SIM and WER on those prompts are partly seen-speaker numbers, and
10 voices leave speaker-clustered intervals wide. Common Voice contributors never occur in the podcast corpus.

Selection, one prompt per contributor (client_id): validated clips (up-votes >= --min-up-votes, no down-votes),
--min-seconds..--max-seconds, >= --min-words words, no digits (the transcript must be read as written), sentence not
in --exclude-sentences (Freya-TR-Eval's short-native part comes from Common Voice texts; datafilters.match_key), then
per speaker the clip with the best DNSMOS OVRL (--dnsmos; else the longest), speakers ranked by that score and
balanced between the male and female gender tags (untagged speakers fill up). Deterministic.

Inputs: Hugging Face Common Voice parquet shards (`--parquet`, e.g. fixie-ai/common_voice_17_0 tr/test) or an
extracted release (`--cv-dir`, its test.tsv and clips/). Writes OUTPUT/prompt-NN.wav (the original recording,
decoded, native rate), OUTPUT/prompts.json (the --prompt-set list) and OUTPUT/summary.json.

  python scripts/data/make_prompt_set.py --parquet data/cv17-tr/test --exclude-sentences data/eval/freya_tr_eval.jsonl \
      --speakers 48 --dnsmos models/sig_bak_ovr.onnx --output data/eval/cv-tr-prompts
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dacvae_tts.datafilters import load_sentences, match_key  # noqa: E402


def parquet_rows(paths):
    import pyarrow.parquet as pq

    files = []
    for path in map(Path, paths):
        files += sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for file in files:
        for row in pq.read_table(file).to_pylist():
            audio = row["audio"]
            yield {"client_id": row["client_id"], "sentence": row["sentence"] or "", "up_votes": row["up_votes"],
                   "down_votes": row["down_votes"], "gender": row.get("gender") or "",
                   "clip": Path(row.get("path") or audio.get("path") or "").name, "bytes": audio["bytes"]}


def release_rows(cv_dir, split):
    from prepare_common_voice import read_tsv

    cv_dir = Path(cv_dir)
    for row in read_tsv(cv_dir / f"{split}.tsv"):
        path = cv_dir / "clips" / row["path"]
        yield {"client_id": row["client_id"], "sentence": row["sentence"] or "", "up_votes": int(row["up_votes"] or 0),
               "down_votes": int(row["down_votes"] or 0), "gender": row.get("gender") or "", "clip": row["path"],
               "bytes": path.read_bytes()}


def gender(tag):
    return "male" if tag.startswith("male") else "female" if tag.startswith("female") else "unknown"


def select(candidates, count):
    """Best clip per speaker, speakers by score, half male / half female where tagged, untagged fill up."""
    ranked = sorted(candidates.values(), key=lambda c: (-c["score"], c["client_id"]))
    quota = {"male": count // 2, "female": count - count // 2}
    chosen = []
    for c in ranked:
        g = gender(c["gender"])
        if quota.get(g, 0) > 0:
            chosen.append(c)
            quota[g] -= 1
    for c in ranked:
        if len(chosen) >= count:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen[:count]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parquet", nargs="+", help="Hugging Face Common Voice parquet files or directories")
    source.add_argument("--cv-dir", help="Extracted Common Voice release language directory (uses --split).tsv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--speakers", type=int, default=48)
    parser.add_argument("--exclude-sentences", nargs="*", default=[], help="Evaluation sentence files (jsonl/txt)")
    parser.add_argument("--dnsmos", help="sig_bak_ovr.onnx: rank clips by DNSMOS OVRL (else by duration)")
    parser.add_argument("--min-seconds", type=float, default=3.5)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--min-up-votes", type=int, default=2)
    args = parser.parse_args(argv)
    excluded = {match_key(text) for path in args.exclude_sentences for text in load_sentences(path)}
    scorer = None
    if args.dnsmos:
        from dacvae_tts.metrics import DNSMOS

        scorer = DNSMOS(args.dnsmos)
    from scipy.signal import resample_poly

    rows = parquet_rows(args.parquet) if args.parquet else release_rows(args.cv_dir, args.split)
    candidates, reasons, total = {}, {}, 0
    for row in rows:
        total += 1
        sentence = row["sentence"]
        reason = None
        if row["down_votes"] > 0 or row["up_votes"] < args.min_up_votes:
            reason = "votes"
        elif any(c.isdigit() for c in sentence) or len(sentence.split()) < args.min_words:
            reason = "text"
        elif match_key(sentence) in excluded:
            reason = "evaluation_sentence"
        if reason is None:
            audio, rate = sf.read(io.BytesIO(row["bytes"]), dtype="float32", always_2d=True)
            audio = audio.mean(1)
            seconds = len(audio) / rate
            if not args.min_seconds <= seconds <= args.max_seconds:
                reason = "duration"
        if reason is not None:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        if scorer is not None:
            g = np.gcd(rate, 16000)
            score = scorer(resample_poly(audio, 16000 // g, rate // g).astype(np.float32))["dnsmos_ovrl"]
        else:
            score = seconds
        best = candidates.get(row["client_id"])
        if best is None or score > best["score"]:
            candidates[row["client_id"]] = dict(row, score=float(score), seconds=seconds, audio=audio, rate=rate)
    chosen = select(candidates, args.speakers)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    prompts = []
    for number, c in enumerate(chosen):
        name = f"prompt-{number:02d}.wav"
        sf.write(out / name, c["audio"], c["rate"], subtype="FLOAT")
        prompts.append({"audio": name, "text": c["sentence"], "speaker": "cv/" + c["client_id"][:16], "uid": c["clip"],
                        "gender": gender(c["gender"]), "seconds": round(c["seconds"], 2),
                        ("dnsmos_ovrl" if scorer else "rank_seconds"): round(c["score"], 3)})
    (out / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=1))
    summary = {"clips": total, "eligible_speakers": len(candidates), "chosen": len(prompts), "removed": reasons,
               "genders": {g: sum(p["gender"] == g for p in prompts) for g in ("male", "female", "unknown")},
               "excluded_sentences": len(excluded), "ranked_by": "dnsmos_ovrl" if scorer else "seconds"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
