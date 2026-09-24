"""ISSAI Turkish Speech Corpus (TSC) manifests for `dacvae-tts prepare` (218.2 h, 186,171 utterances).

License: MIT per the Hugging Face card of issai/Turkish_Speech_Corpus (the TurkicASR code repository that
introduced it is CC-BY-4.0); recorded as "MIT" in every row. Paper: Mussakhojayeva et al., "Multilingual Speech
Recognition for Turkic Languages", Information 14(2):74, 2023.

The corpus is one 21 GB archive; extract it first (its Train/Dev/Test folders hold <utt>.wav + <utt>.txt pairs,
the layout TurkicASR's training/local/prep_data.py reads):
  huggingface-cli download issai/Turkish_Speech_Corpus ISSAI_TSC_218.tar.gz --repo-type dataset --local-dir raw/tsc
  tar -xzf raw/tsc/ISSAI_TSC_218.tar.gz -C raw/tsc        # -> raw/tsc/ISSAI_TSC_218/{Train,Dev,Test}
Train and Dev are used by default; Test stays out so the corpus' ASR test set remains untouched (add it with
--splits Train,Dev,Test if needed; Freya-TR-Eval exclusion applies either way).

Speakers: the release has no speaker labels (TurkicASR maps every utterance to its own speaker), so by default
speaker = "tsc/<utt>" and only within-utterance prompting is possible (merge with --keep-singletons). If the file
names on the 5% sample turn out to encode a recording/speaker, pass --speaker-regex with one capture group (e.g.
'^(.+)_\\d+$') to group them; verify groups with speaker embeddings before trusting cross-utterance pairs.
Transcripts: check a sample for casing/punctuation. ASR-style lower-case unpunctuated text differs from the
punctuated podcast transcripts the model is trained on; if so, prefer Whisper's punctuated hypothesis from
transcribe_corpus.py for this source or keep its mix share small.

Chain:
  python scripts/data/prepare_issai_tsc.py --root raw/tsc/ISSAI_TSC_218 --sample-fraction 0.05 \
      --output data/sources/tsc-sample --freya-sentences data/eval/freya_tr_eval.jsonl --vad
  python scripts/data/prepare_issai_tsc.py --root raw/tsc/ISSAI_TSC_218 \
      --output data/sources/tsc --freya-sentences data/eval/freya_tr_eval.jsonl --vad
  python scripts/transcribe_corpus.py --raw data/sources/tsc/manifest --output outputs/scores-tsc --device cuda \
      --dnsmos models/sig_bak_ovr.onnx
  python scripts/make_drop_list.py --scores outputs/scores-tsc/scores.jsonl --max-cer 0.1 --min-ovrl 2.8 \
      --min-words 2 --output data/drop-tsc.json
  dacvae-tts prepare --manifest data/sources/tsc/manifest --output data/cache/tsc --device cuda \
      --speaker-column speaker --text-normalization turkish-v1 --languages tr --loudness -16 --min-seconds 1 --max-seconds 20
  dacvae-tts merge --inputs data/tr55/parts/part-* data/cache/tsc --output data/tr-mix \
      --drop-uids data/drop-all.json --keep-singletons      # data/drop-all.json = jq -s add data/drop-*.json
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest_pipeline import ManifestBuilder, add_arguments  # noqa: E402

from dacvae_tts.datafilters import check_training_source  # noqa: E402

LICENSE = "MIT"


def split_directory(root, split):
    """TSC folders are capitalized (Train/Dev/Test); accept any casing."""
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == split.lower():
            return child
    raise FileNotFoundError(f"{root} has no {split} folder")


def candidates(root, splits, speaker_regex=None):
    pattern = re.compile(speaker_regex) if speaker_regex else None
    for split in splits:
        directory = split_directory(root, split)
        for wav in sorted(directory.rglob("*.wav")):
            stem = wav.stem
            transcript = wav.with_suffix(".txt")
            text = transcript.read_text(encoding="utf-8").strip() if transcript.exists() else ""
            speaker = f"tsc/{stem}"
            reject = None if transcript.exists() else "missing_transcript"
            if pattern:
                match = pattern.search(stem)
                if match:
                    speaker = f"tsc/{match.group(1)}"
                else:
                    reject = reject or "speaker_regex"
            yield dict(
                id=f"issai-tsc/{directory.name}/{wav.relative_to(directory).with_suffix('')}",
                text=text,
                speaker=speaker,
                path=wav,
                source_recording=str(wav.relative_to(root)),
                meta={"split": directory.name},
                reject=reject,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Extracted ISSAI_TSC_218 directory")
    parser.add_argument("--splits", default="Train,Dev", help="Comma-separated TSC folders (Test is held out)")
    parser.add_argument("--speaker-regex", help="Regex with one group mapping a file stem to a speaker/recording")
    add_arguments(parser)
    args = parser.parse_args(argv)
    root = Path(args.root)
    check_training_source(str(root.resolve()))
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        split_directory(root, split)  # fail before any work if a folder is missing
    builder = ManifestBuilder(
        args, source="issai-tsc", license=LICENSE,
        origin={"repo": "issai/Turkish_Speech_Corpus", "root": str(root.resolve()), "splits": splits,
                "speaker_regex": args.speaker_regex},
    )
    return builder.run(candidates(root, splits, args.speaker_regex))


if __name__ == "__main__":
    main()
