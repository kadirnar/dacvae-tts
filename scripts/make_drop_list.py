"""Turn corpus transcription/DNSMOS scores into a uid drop list for `dacvae-tts merge --drop-uids`.

  python scripts/make_drop_list.py --scores outputs/corpus-scores/scores.jsonl --max-cer 0.1 --min-ovrl 2.8 \
      --min-words 3 --output data/drop-clean.json

Every threshold is applied on its own whenever the row has that score, so a row with CER 0.6 is dropped even when
DNSMOS failed on it. A row missing a score that other rows have (Whisper/decode errors, DNSMOS errors) is dropped as
"unscored" unless `--keep-unscored`; the summary counts the missing scores per metric. A threshold whose score no row
has at all (scores written without `transcribe_corpus.py --dnsmos`) cannot be applied: it is switched off with a
warning and listed under `disabled_thresholds`, instead of dropping every row.
"""

import argparse
import json
import math
import sys
from pathlib import Path


def score(row, key):
    """The row's finite numeric score, or None when it is absent, null, NaN or not a number."""
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cer", type=float, default=0.1, help="Whisper-large-v3 CER vs the given transcript")
    parser.add_argument("--max-wer", type=float, default=None)
    parser.add_argument("--min-ovrl", type=float, default=2.8, help="DNSMOS OVRL floor")
    parser.add_argument("--min-quality", type=float, default=None, help="Dataset quality_score floor")
    parser.add_argument("--min-words", type=int, default=2)
    unscored = parser.add_mutually_exclusive_group()
    unscored.add_argument(
        "--keep-unscored", action="store_true",
        help="Keep rows missing a score an active threshold needs (the scores they have are still applied)",
    )
    unscored.add_argument("--drop-errors", action="store_true", help="No-op: unscored rows are dropped by default")
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.scores).read_text().splitlines() if line.strip()]
    # (score key, drop reason, threshold, whether a value fails it); None thresholds are inactive.
    checks = [
        ("cer", "cer", args.max_cer, lambda value: value > args.max_cer),
        ("wer", "wer", args.max_wer, lambda value: value > args.max_wer),
        ("dnsmos_ovrl", "dnsmos", args.min_ovrl, lambda value: value < args.min_ovrl),
        ("quality_score", "quality", args.min_quality, lambda value: value < args.min_quality),
    ]
    checks = [check for check in checks if check[2] is not None]
    # A score absent from every row was never computed: the threshold is off (and said so), not a reason to drop all.
    disabled = [key for key, *_ in checks if rows and all(score(r, key) is None for r in rows)]
    for key in disabled:
        print(f"warning: no row has {key}; its threshold is disabled (score the corpus with it to apply it)",
              file=sys.stderr)
    checks = [check for check in checks if check[0] not in disabled]
    drop, reasons, missing, unscored_rows = [], {}, {}, 0

    def add(uid, reason):
        drop.append(uid)
        reasons[reason] = reasons.get(reason, 0) + 1

    for r in rows:
        values = {key: score(r, key) for key, *_ in checks}
        absent = [key for key, value in values.items() if value is None]
        for key in absent:
            missing[key] = missing.get(key, 0) + 1
        unscored_rows += bool(absent)
        failed = next((reason for key, reason, _, fails in checks if key not in absent and fails(values[key])), None)
        if failed is not None:
            add(r["uid"], failed)
        elif len(r["text"].split()) < args.min_words:
            add(r["uid"], "short")
        elif absent and not args.keep_unscored:
            add(r["uid"], "unscored")
    Path(args.output).write_text(json.dumps(drop))
    summary = {
        "rows": len(rows),
        "dropped": len(drop),
        "kept_fraction": 1 - len(drop) / max(len(rows), 1),
        "reasons": reasons,
        "unscored_rows": unscored_rows,
        "missing_scores": missing,
        "disabled_thresholds": disabled,
    }
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
