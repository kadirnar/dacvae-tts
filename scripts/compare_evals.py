r"""Markdown comparison of evaluation runs: speaker-clustered bootstrap intervals and paired differences.

Inputs are per-utterance result JSONL files: eval_sentences.py `results.jsonl` (Freya-TR-Eval), monitor.py
`RUN/monitor/step-N/results.jsonl`, `dacvae-tts evaluate` or `run-eval` outputs. Give each run as LABEL=PATH
(a directory means PATH/results.jsonl); several comma-separated paths under one label are generation-seed
replicates, averaged per utterance. The first run is the baseline unless --baseline names another.

  python scripts/compare_evals.py base=outputs/freya-tr-w512-clean-60000 clamp=outputs/demo-clamp \
      predictor=outputs/demo-predictor --stratify length --markdown compare.md --output compare.json

  # Both prompt draws (seeds 42 and 1000) pooled per system: 990 sentences, one paired test
  python scripts/compare_evals.py base=outputs/freya-tr-w512-clean-60000,outputs/demo-base-s1000 \
      clamp=outputs/demo-clamp,outputs/demo-clamp-s1000

Per system: n, CER, corpus and per-utterance-mean WER with intervals, S/D/I rates, SIM/SIM-o, DNSMOS OVRL,
UTMOS, clipped-sample fraction and RTF whenever present (missing metrics are skipped, never imputed). Paired
table: difference vs the baseline with the interval from identical resampled clusters and a win/loss/tie
verdict (tie when the interval contains 0). Why speaker clusters: sentences of one voice are correlated; on
the 495 Freya sentences (10 held-out speakers) clustering widens the WER interval by 40-70 % and differences
below ~0.7 WER points are unresolvable -- see dacvae_tts.comparison for the evidence and the method.
"""

import argparse
import json
import sys
from pathlib import Path

from dacvae_tts.comparison import DEFAULT_SAMPLES, METRIC_SPECS, compare_evaluations, markdown_report
from dacvae_tts.data import jsonl


def results_file(path):
    path = Path(path)
    path = path / "results.jsonl" if path.is_dir() else path
    if not path.is_file():
        raise SystemExit(f"No result file at {path}")
    return path


def parse_run(spec):
    """LABEL=PATH[,PATH...] or PATH[,PATH...]; the default label is the run directory (or file stem)."""
    label, sep, paths = spec.partition("=")
    if not sep or not label or "/" in label:
        label, paths = None, spec
    files = [results_file(p) for p in paths.split(",") if p]
    if not files:
        raise SystemExit(f"No result files in {spec!r}")
    if label is None:
        label = files[0].parent.name if files[0].name == "results.jsonl" else files[0].stem
    return label, files


def fields(value):
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="+", help="LABEL=PATH[,PATH...] (PATH: results.jsonl or its directory)")
    parser.add_argument("--baseline", help="Label of the reference run (default: the first run)")
    parser.add_argument("--key", help="Comma-separated pairing fields (default: auto, e.g. id)")
    parser.add_argument(
        "--cluster", default="auto",
        help="Bootstrap cluster field: auto (speaker, then prompt IDs), a field such as speaker or prompt, "
             "or none for the plain utterance-level bootstrap",
    )
    parser.add_argument(
        "--seed-field", default="seed",
        help="Row field telling generation seeds apart within one file (none: every file is one seed)",
    )
    parser.add_argument(
        "--metrics",
        help=f"Comma-separated metrics of the paired/stratum tables (default: headline set); one of "
             f"{', '.join(METRIC_SPECS)}",
    )
    parser.add_argument(
        "--stratify", action="append", default=[],
        help="Add a per-stratum table for a row field (e.g. register); 'length' buckets reference word "
             "counts into 1-5, 6-9, 10+. Repeatable",
    )
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_SAMPLES, help="Resamples (%(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed (results are deterministic)")
    parser.add_argument("--level", type=float, default=0.95, help="Interval coverage (default %(default)s)")
    parser.add_argument(
        "--utterance-ci", action="store_true",
        help="Also show the utterance-level (unclustered) interval for reference",
    )
    parser.add_argument("--markdown", help="Write the markdown report here (it is always printed)")
    parser.add_argument("--output", help="Write the full JSON report here")
    args = parser.parse_args(argv)
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")

    systems, sources = {}, {}
    for spec in args.runs:
        label, files = parse_run(spec)
        if label in systems:
            parser.error(f"Duplicate run label {label!r}; name runs as LABEL=PATH")
        systems[label] = [list(jsonl(path)) for path in files]
        sources[label] = [str(path) for path in files]
    try:
        report = compare_evaluations(
            systems,
            baseline=args.baseline,
            metrics=fields(args.metrics),
            key=fields(args.key),
            cluster=args.cluster,
            seed_field=None if args.seed_field == "none" else args.seed_field,
            stratify=args.stratify,
            samples=args.bootstrap,
            seed=args.seed,
            level=args.level,
            utterance_ci=args.utterance_ci,
        )
    except ValueError as error:
        raise SystemExit(f"compare_evals: {error}") from error
    report["config"]["sources"] = sources
    text = markdown_report(report)
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
