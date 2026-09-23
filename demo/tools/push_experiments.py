"""Publish the demo experiments (Freya-TR-Eval with the published checkpoint) to the run's VoiceHub dataset.

Every experiment folder contributes summary.json and results.jsonl (per-sentence transcripts and scores); the folders
named with --with-audio also upload their WAVs and prompt WAVs for listening. A README table summarizes all runs.

  python push_experiments.py --repo VoiceHub/dacvae-tts-tr-w512-clean --with-audio demo-final-s42
"""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

OUT = Path("/workspace/outputs")
EXTRA = {
    "freya-tr-w512-clean-60000": "baseline: rule duration, CFG 5 (published numbers)",
    "freya-w512-dur1.15": "duration x1.15 for every prompt",
    "freya-w512-dur1.3": "duration x1.3 for every prompt",
    "freya-w512-rate15": "fixed 15 chars/s (old demo 'Sabit hız')",
    "freya-w512-rate13": "fixed 13 chars/s",
}


def describe(summary):
    keys = [("duration_mode", "rule"), ("candidates", 1), ("cfg_rescale", 0.0), ("apg_eta", 1.0), ("apg_momentum", 0.0),
            ("speaker_guidance", None), ("guidance_until", 1.0), ("duration_scale", 1.0), ("chars_per_second", 0.0)]
    parts = [f"{k}={summary.get(k)}" for k, default in keys if summary.get(k, default) not in (default, None)]
    return ", ".join(parts) or "defaults"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--with-audio", nargs="*", default=[])
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    folders = sorted(p for p in OUT.iterdir() if p.is_dir() and (p / "summary.json").exists()
                     and (p.name.startswith("demo-") or p.name in EXTRA))
    rows = []
    for folder in folders:
        s = json.loads((folder / "summary.json").read_text())
        seed = 1000 if folder.name.endswith("s1000") else 42
        rows.append((folder.name, seed, EXTRA.get(folder.name, describe(s)), s))
    table = ["| run | prompt seed | settings | WER % | CER % | SIM | DNSMOS OVRL | error-free | files clipping |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for name, seed, settings, s in sorted(rows, key=lambda r: (r[1], r[3]["wer"])):
        clip = s.get("files_clipping")
        table.append(f"| `{name}` | {seed} | {settings} | {100 * s['wer']:.2f} | {100 * s['cer']:.2f} | "
                     f"{s.get('speaker_similarity', 0):.3f} | {s.get('dnsmos_ovrl', 0):.3f} | {s.get('sentences_wer_zero')} | "
                     f"{'' if clip is None else f'{100 * clip:.0f}%'} |")
    readme = f"""# Demo experiments on Freya-TR-Eval (checkpoint tr-w512-clean step 60k)

All runs: 495 Freya-TR-Eval sentences (unseen text), 24 held-out prompt voices (unseen speakers; prompt seed 42 is the set
used for every published number, seed 1000 a second, disjoint draw of voices and noise), guidance 5, 32 Euler steps,
sway -1, scored with faster-whisper large-v3 (beam 5, Turkish normalization), WavLM-base-plus-SV similarity and DNSMOS.
{args.notes}

{chr(10).join(table)}

Each folder holds `summary.json` and `results.jsonl` (per sentence: text, prompt, Whisper hypothesis, WER/CER, similarity,
DNSMOS, duration, peak/clipping/loudness){"; folders with audio also hold the WAVs" if args.with_audio else ""}.
"""
    staging = OUT / "_push_demo_experiments"
    staging.mkdir(exist_ok=True)
    (staging / "README.md").write_text(readme)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for name, *_ in rows:
        patterns = ["summary.json", "results.jsonl"] + (["*.wav", "*.json"] if name in args.with_audio else [])
        api.upload_folder(folder_path=str(OUT / name), repo_id=args.repo, repo_type="dataset",
                          path_in_repo=f"demo-experiments/{name}", allow_patterns=patterns,
                          commit_message=f"demo experiments: {name}")
        print("pushed", name, flush=True)
    api.upload_file(path_or_fileobj=str(staging / "README.md"), path_in_repo="demo-experiments/README.md",
                    repo_id=args.repo, repo_type="dataset", commit_message="demo experiments: summary table")
    print(readme)


if __name__ == "__main__":
    main()
