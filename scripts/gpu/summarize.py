"""Headline numbers of every evaluation under a directory, as a tab-separated table (one row per summary.json).

  python scripts/gpu/summarize.py /workspace/outputs/inference > summary.tsv

Columns that no summary has are left out; a missing value is empty. Nested summaries (oracle_best_of_n.py) keep
only their flat numeric fields. The table is for reading; decisions come from compare_evals.py's paired intervals.
"""

import json
import sys
from pathlib import Path

COLUMNS = (
    "count", "failed", "cer", "wer", "freya_cer", "freya_wer", "wer_filtered", "hallucination_rate", "sim_o",
    "sim_r", "sim_o_speechbrain", "speaker_similarity", "utmos", "utmosv2", "dnsmos_ovrl", "dnsmos_sig",
    "dnsmos_bak", "files_clipping", "clipped_fraction_fullband", "median_lufs", "loudness_lufs", "bandwidth_hz",
    "selection_changed", "rtf",
)


def main(root):
    rows = []
    for path in sorted(Path(root).glob("*/summary.json")):
        summary = json.loads(path.read_text())
        flat = {k: v for k, v in summary.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        rows.append((path.parent.name, flat))
    columns = [c for c in COLUMNS if any(c in flat for _, flat in rows)]
    print("\t".join(["run", *columns]))
    for name, flat in rows:
        cells = [f"{flat[c]:.4f}" if isinstance(flat.get(c), float) else str(flat.get(c, "")) for c in columns]
        print("\t".join([name, *cells]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
