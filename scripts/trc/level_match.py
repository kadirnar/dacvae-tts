"""Copy an eval_sentences.py output directory with every generated sentence WAV normalized to one loudness.

UTMOS and DNSMOS move with level, so options that change loudness (the pre-tanh gain, APG, CFG scale) must be
compared at equal loudness (#13). The copy keeps cases.json, results.jsonl and the prompt WAVs, so
`eval_sentences.py --rescore` scores it with the same prompts:

  python scripts/trc/level_match.py outputs/trc/inference-runc/base outputs/trc/inference-runc/base-lufs16
  python scripts/eval_sentences.py --checkpoint CKPT --prompt-set PROMPTS --sentences FREYA \
      --output outputs/trc/inference-runc/base-lufs16 --rescore --protocol-v2 ...
"""

import argparse
import json
import shutil
from pathlib import Path

import soundfile as sf

from dacvae_tts.codec import normalize_loudness


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--lufs", type=float, default=-16.0)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(source.iterdir()):
        if path.is_dir() or path.name == "summary.json":
            continue
        if path.suffix == ".wav" and not path.name.startswith("prompt-"):
            audio, rate = sf.read(path, dtype="float32")
            sf.write(output / path.name, normalize_loudness(audio, rate, args.lufs), rate, subtype="FLOAT")
            count += 1
        elif path.name == "results.jsonl":
            # --rescore reuses each row's `audio` path: point it at the normalized copy, not the source WAV.
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for row in rows:
                row["audio"] = str(output / Path(row["audio"]).name)
            (output / path.name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        else:
            shutil.copy2(path, output / path.name)
    print(f"{count} sentence WAVs normalized to {args.lufs} LUFS -> {output}")


if __name__ == "__main__":
    main()
