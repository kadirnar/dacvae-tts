"""End-to-end check of a deployed demo through its public API (anonymous unless HF_TOKEN is given with --token).

  python verify_space.py Vyvo/dacvae-tts-tr-demo
"""

import json
import os
import sys
import time

import soundfile as sf
from gradio_client import Client, handle_file

SPACE = "/workspace/release/space"
MODEL = "w512-clean 60k · yayımlanan (Freya WER %4,3)"
ADVANCED = [5.0, 32, "CFG (en düşük WER)", 0.7, 0.5, 3.0, 1.0, 1.0, "Otomatik"]


def main():
    target = sys.argv[1]
    token = os.environ.get("HF_TOKEN") if "--token" in sys.argv else None
    client = Client(target, verbose=False, token=token)
    examples = json.load(open(f"{SPACE}/examples/prompts.json"))
    prompt = lambda i: handle_file(f"{SPACE}/{examples[i]['audio']}")  # noqa: E731

    def synthesize(label, ref, ref_text, text, candidates=3, rate="Otomatik (önerilen)", advanced=ADVANCED, choice=MODEL,
                   repo="", file=""):
        started = time.time()
        audio, info, normalized, metrics, shown, reference = client.predict(
            ref, ref_text, text, rate, 15, candidates, 42, True, choice, repo, file, "dataset", *advanced,
            api_name="/synthesize")
        data, rate_hz = sf.read(audio)
        print(f"{label}: {time.time() - started:.1f} s istek, {len(data) / rate_hz:.1f} s ses @ {rate_hz} ({sf.info(audio).subtype}), "
              f"GPU {metrics.get('gpu_seconds', 0):.1f} s, WER {metrics.get('wer', float('nan')):.3f}, "
              f"benzerlik {metrics.get('similarity', float('nan')):.3f}, süre modu {metrics.get('duration_mode')}", flush=True)
        return metrics, normalized

    started = time.time()
    text, report = client.predict(prompt(0), api_name="/transcribe")
    print(f"transkript: {time.time() - started:.1f} s · {text[:60]}… · {report.splitlines()[0]}")
    synthesize("varsayılan (otomatik transkript, 3 aday)", prompt(0), "",
               "Yarın öğleden sonra sağanak bekleniyormuş, şemsiyeni unutma.")
    _, normalized = synthesize("semboller + tarih + kısaltma", prompt(1), examples[1]["text"],
                               "Toplantı 23.09.2026 tarihinde saat 14:30'da; bütçe %20 artıp 1.250 TL'ye çıktı, Dr. Ayşe ABD'den katılacak.")
    print("   modelin okuduğu:", normalized)
    synthesize("uzun metin (3 parça)", prompt(2), examples[2]["text"],
               "Bu sabah erkenden kalkıp sahilde uzun bir yürüyüş yaptım; deniz o kadar sakindi ki martıların sesi bile net "
               "duyuluyordu. Sonra eve dönüp kahvaltı hazırladım. Anneannem her bayram sabahı mutfakta baklava açar, evin içi "
               "tereyağı ve fıstık kokusuyla dolar, biz çocuklar da sofranın kurulmasını sabırsızlıkla beklerdik.")
    synthesize("APG + süre tahmincisi", prompt(1), examples[1]["text"], "Kosova'da Kasım ayında seçimler yapılacak.",
               candidates=1, advanced=[5.0, 32, "APG (daha temiz ses)", 0.7, 0.5, 3.0, 1.0, 1.0, "Süre tahmincisi"])
    synthesize("özel checkpoint (Hub URL)", prompt(0), examples[0]["text"], "Özel checkpoint ile deneme cümlesi.", candidates=1,
               choice="Özel checkpoint (aşağıdaki depo)",
               repo="https://huggingface.co/datasets/VoiceHub/dacvae-tts-tr-w512-clean/blob/main/checkpoints/step-0060000.pt")
    result = client.predict([MODEL, "w512-stage2-hq 10k (Freya %4,6)"], prompt(2), examples[2]["text"],
                            "Hafta sonu kayınvalidemlere yemeğe gideceğiz.", "Otomatik (önerilen)", 15, 3, 7, "", "", "dataset",
                            *ADVANCED, api_name="/compare")
    print("A/B:", [row[:6] for row in result[4]["data"]])
    batch = client.predict(MODEL, "5 örnek cümle", "", "Örnek ses 1", None, "", "Otomatik (önerilen)", 15, 3, 42, "", "",
                           "dataset", *ADVANCED, api_name="/batch_test")
    print("toplu test:", batch[0])
    try:
        client.predict(prompt(0), "x", "Merhaba.", "Otomatik (önerilen)", 15, 1, 42, True, "Özel checkpoint (aşağıdaki depo)",
                       "yok/yok-depo", "a.pt", "dataset", *ADVANCED, api_name="/synthesize")
    except Exception as error:
        print("beklenen hata mesajı:", str(error)[:120])
    print(client.predict(api_name="/diag"))


if __name__ == "__main__":
    main()
