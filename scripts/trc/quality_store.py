"""Write the quality-condition store of a cache: CACHE/quality/dnsmos.json = {uid: [SIG, BAK, OVRL]}.

Scores come from scripts/transcribe_corpus.py (scores.jsonl, keyed by the cache uids); rows without all three DNSMOS
values are left out and train as "unknown quality". Set `model.quality_condition: true` and
`train.quality_scores: quality/dnsmos.json`.

  python scripts/trc/quality_store.py --scores outputs/trc-scores/scores.jsonl --cache data/trc/clean
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", default="quality/dnsmos.json", help="Relative to the cache")
    args = parser.parse_args()
    keys = ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl")
    scores = {}
    for line in open(args.scores):
        row = json.loads(line)
        if all(isinstance(row.get(k), (int, float)) for k in keys):
            scores[row["uid"]] = [round(float(row[k]), 4) for k in keys]
    cache = Path(args.cache)
    with sqlite3.connect(f"file:{cache / 'index.sqlite'}?mode=ro", uri=True) as db:
        uids = [uid for (uid,) in db.execute("SELECT uid FROM samples")]
    store = {uid: scores[uid] for uid in uids if uid in scores}
    output = cache / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(store))
    values = np.array(list(store.values()))
    summary = {"rows": len(uids), "scored": len(store),
               "percentiles_50_90": {k: np.percentile(values[:, i], [50, 90]).round(3).tolist()
                                     for i, k in enumerate(("sig", "bak", "ovrl"))}}
    (output.parent / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
