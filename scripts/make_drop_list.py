"""Turn corpus transcription/DNSMOS scores into a uid drop list for `dacvae-tts merge --drop-uids`.

  python scripts/make_drop_list.py --scores outputs/corpus-scores/scores.jsonl --max-cer 0.1 --min-ovrl 2.8 \
      --min-words 3 --output data/drop-clean.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cer", type=float, default=0.1, help="Whisper-large-v3 CER vs the given transcript")
    parser.add_argument("--max-wer", type=float, default=None)
    parser.add_argument("--min-ovrl", type=float, default=2.8, help="DNSMOS OVRL floor")
    parser.add_argument("--min-quality", type=float, default=None, help="Dataset quality_score floor")
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--drop-errors", action="store_true", help="Drop rows Whisper/DNSMOS could not score")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.scores).read_text().splitlines() if line.strip()]
    drop, reasons = [], {}

    def add(uid, reason):
        drop.append(uid)
        reasons[reason] = reasons.get(reason, 0) + 1

    for r in rows:
        if "cer" not in r or "dnsmos_ovrl" not in r:
            if args.drop_errors:
                add(r["uid"], "unscored")
            continue
        if r["cer"] > args.max_cer:
            add(r["uid"], "cer")
        elif args.max_wer is not None and r["wer"] > args.max_wer:
            add(r["uid"], "wer")
        elif r["dnsmos_ovrl"] < args.min_ovrl:
            add(r["uid"], "dnsmos")
        elif args.min_quality is not None and r["quality_score"] < args.min_quality:
            add(r["uid"], "quality")
        elif len(r["text"].split()) < args.min_words:
            add(r["uid"], "short")
    Path(args.output).write_text(json.dumps(drop))
    print(json.dumps({"rows": len(rows), "dropped": len(drop), "kept_fraction": 1 - len(drop) / max(len(rows), 1), "reasons": reasons}))


if __name__ == "__main__":
    main()
