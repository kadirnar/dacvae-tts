"""Exact simulation of prompt-rate-dependent duration modes by mixing per-sentence outputs of Freya runs.

Every sentence's WAV depends only on (prompt, sentence, seed, duration mode), so choosing per prompt which run a sentence
comes from reproduces what a hybrid mode would generate. Prompt rates are computed the way the demo does it:
normalized reference characters / reference seconds.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/workspace/dacvae-tts/scripts")
from monitor import select_cases  # noqa: E402

from dacvae_tts.duration import speaking_rate  # noqa: E402
from dacvae_tts.text import normalize  # noqa: E402


def load(folder):
    rows = [json.loads(l) for l in open(f"/workspace/outputs/{folder}/results.jsonl") if l.strip()]
    return {r["id"]: r for r in rows if "wer" in r}


def prompt_rates(seed):
    data, cases = select_cases("/workspace/data/tr55/merged", 24, seed)
    return {f"prompt-{i:02d}.wav": speaking_rate(data.lengths[c["prompt_index"]], normalize(c["prompt_text"], "turkish-v1"))
            for i, c in enumerate(cases)}


def summary(rows):
    rows = list(rows)
    return (sum(r["word_edits"] for r in rows) / sum(r["words"] for r in rows),
            sum(r["char_edits"] for r in rows) / sum(r["chars"] for r in rows),
            float(np.mean([r["dnsmos_ovrl"] for r in rows])), sum(r["wer"] == 0 for r in rows),
            float(np.mean([r["speaker_similarity"] for r in rows])))


def main():
    seed, runs = int(sys.argv[1]), json.loads(sys.argv[2])  # {"rule": folder, "clamp": folder, "predictor": folder}
    rates = prompt_rates(seed)
    data = {k: load(v) for k, v in runs.items()}
    print("prompt rates:", sorted(round(r, 1) for r in rates.values()))
    policies = {
        "rule": lambda r: "rule",
        "clamp": lambda r: "clamp",
        "predictor": lambda r: "predictor",
        "hybrid slow<13 predictor, fast>17 clamp": lambda r: "predictor" if r < 13 else ("clamp" if r > 17 else "rule"),
        "hybrid slow<12 predictor, fast>17 clamp": lambda r: "predictor" if r < 12 else ("clamp" if r > 17 else "rule"),
        "hybrid slow<14 predictor, fast>17 clamp": lambda r: "predictor" if r < 14 else ("clamp" if r > 17 else "rule"),
        "predictor except normal": lambda r: "rule" if 13 <= r <= 17 else "predictor",
    }
    for name, policy in policies.items():
        if any(policy(r) not in data for r in rates.values()):
            continue
        rows = [data[policy(rates[row["prompt"]])][sid] for sid, row in data["rule"].items()]
        wer, cer, ovrl, zero, sim = summary(rows)
        print(f"  {name:42s} WER {wer:.4f} CER {cer:.4f} SIM {sim:.4f} OVRL {ovrl:.3f} zero {zero}")


if __name__ == "__main__":
    main()
