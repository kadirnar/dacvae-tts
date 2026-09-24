"""Pre-generate the showcase samples of the Space (Samples tab) with the default settings of a running app.

  python make_samples.py http://127.0.0.1:7861
"""

import json
import shutil
import sys
from pathlib import Path

from gradio_client import Client, handle_file

SPACE = Path("/workspace/release/space")
SENTENCES = [
    "Bu sabah erkenden kalkıp sahilde uzun bir yürüyüş yaptım; deniz o kadar sakindi ki martıların sesi bile net duyuluyordu.",
    "Toplantı 23.09.2026 tarihinde saat 14:30'da başlayacak; bütçenin %20'si Ar-Ge'ye ayrıldı.",
    "Anneannem her bayram sabahı mutfakta baklava açar, evin içi tereyağı ve fıstık kokusuyla dolar.",
]


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7861"
    candidates = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    import os

    client = Client(url, verbose=False, token=os.environ.get("HF_TOKEN"))
    examples = json.loads((SPACE / "examples" / "prompts.json").read_text())
    out = SPACE / "samples"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    rows = []
    for v, example in enumerate(examples):
        for s, sentence in enumerate(SENTENCES):
            if (v + s) % 3 == 2:  # two sentences per voice keep the tab short
                continue
            audio, info, normalized, metrics, _, _ = client.predict(
                handle_file(str(SPACE / example["audio"])), example["text"], sentence, "Automatic (recommended)", 15,
                candidates, 42 + s, True, "w512-clean 60k · published (Freya WER 4.3 %)", "", "", "dataset",
                5.0, 32, "CFG (lowest WER)", 0.7, 0.5, 3.0, 1.0, 1.0, "Automatic", api_name="/synthesize")
            name = f"voice{v + 1}-sentence{s + 1}.wav"
            shutil.copy(audio, out / name)
            label = (f"Voice {v + 1} · “{sentence}” · WER {metrics.get('wer', float('nan')):.2f}, "
                     f"similarity {metrics.get('similarity', float('nan')):.2f}")
            rows.append({"audio": f"samples/{name}", "label": label, "text": sentence, "voice": example["audio"],
                         "metrics": {k: metrics.get(k) for k in ("wer", "cer", "similarity", "dnsmos_ovrl", "seconds")}})
            print(label, flush=True)
    (out / "samples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
