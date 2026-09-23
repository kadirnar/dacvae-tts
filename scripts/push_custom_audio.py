"""Push a custom-sentence synthesis folder (from eval_sentences.py) to a run's Hugging Face dataset, with the texts.

  python scripts/push_custom_audio.py --folder outputs/custom5-tr-nano-a-40k --repo VoiceHub/dacvae-tts-tr-nano-a \
      --name custom-sentences-step-0040000 --checkpoint runs/tr-nano-a/step-0040000.pt
"""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--name", required=True, help="Folder name in the repository")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--title", default="Custom sentences")
    args = parser.parse_args()
    folder = Path(args.folder)
    rows = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines() if line.strip()]
    summary = json.loads((folder / "summary.json").read_text()) if (folder / "summary.json").exists() else {}
    lines = [f"# {args.title}", "",
             f"Checkpoint: `{args.checkpoint}` · guidance {summary.get('guidance')} · {summary.get('steps')} Euler steps · "
             f"sway {summary.get('sway')} · prompts: held-out validation speakers (`prompt-NN.wav`, real speech decoded through the codec).", "",
             "These sentences were written for this test and do not occur in the training data. WER/CER: Whisper-large-v3 (Turkish "
             "normalization, numbers spelled out on both sides); DNSMOS OVRL: 16 kHz non-intrusive quality estimate.", ""]
    if summary.get("wer") is not None:
        lines += [f"Overall: WER {summary['wer']:.3f} · CER {summary['cer']:.3f} · SIM {summary.get('speaker_similarity', 0):.3f} · "
                  f"DNSMOS OVRL {summary.get('dnsmos_ovrl', 0):.2f}", ""]
    for r in rows:
        audio = Path(r.get("audio", "")).name
        lines += [f"## `{audio}` — prompt `{r.get('prompt', '')}`", "", f"**Metin:** {r['text']}", ""]
        if "hypothesis" in r:
            lines += [f"Whisper: *{r['hypothesis'].strip()}*", "",
                      f"WER {r['wer']:.2f} · CER {r['cer']:.2f} · SIM {r.get('speaker_similarity', 0):.3f} · "
                      f"DNSMOS OVRL {r.get('dnsmos_ovrl', 0):.2f} · {r.get('audio_seconds', 0):.1f} s", ""]
        elif "error" in r:
            lines += [f"Error: {r['error']}", ""]
    (folder / "README.md").write_text("\n".join(lines))
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=args.repo, repo_type="dataset", path_in_repo=args.name,
                      allow_patterns=["*.wav", "*.json", "*.jsonl", "README.md"], commit_message=f"{args.name}: custom sentences")
    print(f"https://huggingface.co/datasets/{args.repo}/tree/main/{args.name}")


if __name__ == "__main__":
    main()
