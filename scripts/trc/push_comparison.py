"""Side-by-side listening set: the same Freya sentences and voices synthesized by several systems, on one Hub page.

Each system is an eval_sentences.py output directory with the same prompt set and seed. The repo gets
samples/<id>/{prompt.wav, <system>.wav}, a README table with the text, each system's Whisper transcript and WER, and the
systems' corpus metrics, so differences can be heard sentence by sentence.

  python scripts/trc/push_comparison.py --count 40 \
      old=outputs/trc/systems/old-auto new=outputs/trc/systems/new-auto

It goes to the comparison/ folder of the experiments repo (--subdir; empty = the repo root).
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("systems", nargs="+", help="LABEL=eval_sentences output directory")
    parser.add_argument("--repo", default="VoiceHub/dacvae-tts-tr-combined")
    parser.add_argument("--subdir", default="comparison")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--title", default="DACVAE-TTS Turkish: old vs new, side by side")
    parser.add_argument("--notes", default="")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    systems = [(label, Path(path)) for label, path in (item.split("=", 1) for item in args.systems)]
    rows = {label: {r["id"]: r for r in map(json.loads, (path / "results.jsonl").read_text().splitlines()) if r}
            for label, path in systems}
    first = systems[0][0]
    # Every 12th sentence: all registers and lengths, and every prompt voice appears (the prompts rotate).
    ids = [i for i in sorted(rows[first]) if all(i in rows[label] for label, _ in systems)]
    chosen = ids[:: max(1, len(ids) // args.count)][: args.count]
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, private=not args.public, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lines = [f"# {args.title}", "", args.notes, "",
                 "Leak-free protocol: Freya-TR-Eval sentences, 48 Common Voice test voices never seen in training, "
                 "Whisper large-v3 transcripts, turkish-v2 metric.", "", "## Corpus metrics (495 sentences)", "",
                 "| system | WER % | CER % | SIM-o | DNSMOS OVRL | UTMOS |", "|---|---:|---:|---:|---:|---:|"]
        for label, path in systems:
            s = json.loads((path / "summary.json").read_text())
            lines.append(f"| {label} | {100 * s['wer']:.2f} | {100 * s['cer']:.2f} | {s.get('sim_o', 0):.3f} | "
                         f"{s.get('dnsmos_ovrl', 0):.3f} | {s.get('utmos', 0):.3f} |")
        lines += ["", "## Samples", "", "| id | text | " + " | ".join(f"{label} (Whisper, WER)" for label, _ in systems) + " |",
                  "|---|---|" + "---|" * len(systems)]
        for sentence in chosen:
            folder = root / "samples" / sentence
            folder.mkdir(parents=True)
            prompt = rows[first][sentence]["prompt"]
            shutil.copy2(systems[0][1] / prompt, folder / "prompt.wav")
            cells = []
            for label, path in systems:
                shutil.copy2(path / f"{sentence}.wav", folder / f"{label}.wav")
                row = rows[label][sentence]
                cells.append(f"{row.get('hypothesis', '').strip()} ({100 * row.get('wer', 0):.0f} %)")
            prefix = f"{args.subdir}/" if args.subdir else ""
            link = f"[{sentence}](https://huggingface.co/{args.repo}/tree/main/{prefix}samples/{sentence})"
            lines.append(f"| {link} | {rows[first][sentence]['text']} | " + " | ".join(cells) + " |")
        (root / "README.md").write_text("\n".join(lines) + "\n")
        api.upload_folder(repo_id=args.repo, folder_path=str(root), path_in_repo=args.subdir or None,
                          commit_message="side-by-side listening set")
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
