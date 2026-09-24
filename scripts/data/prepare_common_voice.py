"""Common Voice Turkish manifests for `dacvae-tts prepare` (CC0-1.0), from an extracted release.

Common Voice releases (v26/v27 at the time of writing) are distributed by the Mozilla Data Collective
(datacollective.mozillafoundation.org) after accepting the terms; download the Turkish archive and extract it:
  cv-corpus-<version>/tr/validated.tsv, clip_durations.tsv, clips/*.mp3
Only `validated.tsv` is used (up-votes > down-votes; ~130 h, ~1,840 speakers, mean clip 3.9 s): it is the superset
of the official train/dev/test clips, and the speaker-hash split of `prepare` replaces Mozilla's split. Clips with
any down-vote are dropped by default (`--max-down-votes 0`, reason `cv_down_votes`), clip durations come from
clip_durations.tsv so the duration filter runs before any MP3 is decoded, and the MP3 bytes are embedded as they are.

Speakers: speaker = "cv/<client_id>" (one id per contributor account). A few contributors read thousands of prompts;
`--max-per-speaker` caps them if the mix should favour speaker variety.
Evaluation hygiene: Freya-TR-Eval's short-native sentences come from Common Voice 17 / CoVoST2 texts, and every
Common Voice prompt is read by many speakers, so the Freya exclusion matters most here (freya_exact in summary.json).
Common Voice is read speech on consumer microphones: expect a wide bandwidth spread (kept_bandwidth_histogram) and a
lower DNSMOS than the podcasts, which is what the `--min-ovrl 2.8` cut of make_drop_list.py is for.

Chain (the 5% sample needs no download beyond the release):
  python scripts/data/prepare_common_voice.py --cv-dir raw/cv-corpus-27.0/tr --sample-fraction 0.05 \
      --output data/sources/cv-tr-sample --freya-sentences data/eval/freya_tr_eval.jsonl --vad
  python scripts/data/prepare_common_voice.py --cv-dir raw/cv-corpus-27.0/tr \
      --output data/sources/cv-tr --freya-sentences data/eval/freya_tr_eval.jsonl --vad
  python scripts/transcribe_corpus.py --raw data/sources/cv-tr/manifest --output outputs/scores-cv-tr --device cuda \
      --dnsmos models/sig_bak_ovr.onnx
  python scripts/make_drop_list.py --scores outputs/scores-cv-tr/scores.jsonl --max-cer 0.1 --min-ovrl 2.8 \
      --min-words 2 --output data/drop-cv-tr.json
  dacvae-tts prepare --manifest data/sources/cv-tr/manifest --output data/cache/cv-tr --device cuda \
      --speaker-column speaker --text-normalization turkish-v1 --languages tr --loudness -16 --min-seconds 1 --max-seconds 20
  dacvae-tts merge --inputs data/tr55/parts/part-* data/cache/cv-tr --output data/tr-mix \
      --drop-uids data/drop-all.json --keep-singletons      # data/drop-all.json = jq -s add data/drop-*.json
`--text-normalization turkish-v1`, `--loudness -16` and the default seed must match the podcast cache.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest_pipeline import ManifestBuilder, add_arguments  # noqa: E402

from dacvae_tts.datafilters import check_training_source  # noqa: E402

LICENSE = "CC0-1.0"
META_COLUMNS = ("sentence_id", "up_votes", "down_votes", "age", "gender", "accents", "variant", "locale", "segment",
                "sentence_domain")


def read_tsv(path):
    # Common Voice TSVs are unquoted: a sentence may contain '"' and must not open a quoted field.
    csv.field_size_limit(sys.maxsize)
    with open(path, encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)


def clip_durations(path):
    if not path.exists():
        return {}
    durations = {}
    for row in read_tsv(path):
        value = row.get("duration[ms]") or row.get("duration")
        if row.get("clip") and value:
            durations[row["clip"]] = float(value) / 1000
    return durations


def votes(value):
    try:
        return int(value or 0)
    except ValueError:
        return 0


def candidates(cv_dir, tsv, max_down_votes, locale):
    durations = clip_durations(cv_dir / "clip_durations.tsv")
    for row in read_tsv(cv_dir / tsv):
        clip = row["path"]
        reject = None
        if votes(row.get("down_votes")) > max_down_votes:
            reject = "cv_down_votes"
        elif row.get("locale") and row["locale"] != locale:
            reject = "cv_locale"
        yield dict(
            id=f"common-voice/{locale}/{Path(clip).stem}",
            text=row.get("sentence", ""),
            speaker=f"cv/{row['client_id']}" if row.get("client_id") else "",
            path=cv_dir / "clips" / clip,
            duration=durations.get(clip),
            source_recording=clip,
            meta={key: row[key] for key in META_COLUMNS if row.get(key) not in (None, "")},
            reject=reject,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cv-dir", required=True, help="Extracted locale directory: .../cv-corpus-<version>/tr")
    parser.add_argument("--tsv", default="validated.tsv", help="Clip list inside --cv-dir")
    parser.add_argument("--locale", default="tr")
    parser.add_argument("--max-down-votes", type=int, default=0, help="Drop clips with more down-votes than this")
    add_arguments(parser)
    args = parser.parse_args(argv)
    cv_dir = Path(args.cv_dir)
    check_training_source(str(cv_dir.resolve()))
    if not (cv_dir / args.tsv).exists():
        raise SystemExit(f"{cv_dir / args.tsv} not found; point --cv-dir at the extracted locale directory")
    release = next((m.group(0) for m in [re.search(r"cv-corpus-[\w.\-]+", str(cv_dir.resolve()))] if m), "unknown")
    builder = ManifestBuilder(
        args, source=f"common-voice-{args.locale}", license=LICENSE,
        origin={"release": release, "cv_dir": str(cv_dir.resolve()), "tsv": args.tsv,
                "max_down_votes": args.max_down_votes},
    )
    return builder.run(candidates(cv_dir, args.tsv, args.max_down_votes, args.locale))


if __name__ == "__main__":
    main()
