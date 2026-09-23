"""DACVAE-TTS Türkçe demo: sıfır-atış ses klonlama, checkpoint karşılaştırma ve Whisper ile doğrulama."""

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import engine as E  # noqa: E402  (imports spaces before torch, loads the models once)
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from dacvae_tts.duration import speaking_rate  # noqa: E402
from dacvae_tts.frontend import speakable  # noqa: E402

CUSTOM = "Özel checkpoint (aşağıdaki depo)"
RATE_AUTO, RATE_PROMPT, RATE_FIXED = "Otomatik (önerilen)", "Prompt'un hızı", "Sabit hız"
GUIDE_CFG, GUIDE_RESCALE, GUIDE_APG, GUIDE_SPLIT = "CFG (en düşük WER)", "CFG + rescale", "APG (daha temiz ses)", "Ayrı metin/konuşmacı"
DURATION_MODELS = {"Otomatik": None, "Prompt hızı (kural)": "rule", "Kural + hızlı prompt sınırı": "clamp",
                   "Süre tahmincisi": "predictor", "Karma (yavaşta tahminci, hızlıda sınır)": "auto",
                   "Hece kuralı": "syllable"}
EXAMPLES = json.loads((ROOT / "examples" / "prompts.json").read_text(encoding="utf-8"))
SAMPLES = json.loads((ROOT / "samples" / "samples.json").read_text(encoding="utf-8")) if (ROOT / "samples" / "samples.json").exists() else []
SENTENCES = [
    "Yarın öğleden sonra sağanak bekleniyormuş, şemsiyeni unutma.",
    "Toplantı 23.09.2026 tarihinde saat 14:30'da başlayacak; lütfen sunum dosyalarınızı öğleden önce paylaşın.",
    "Yapay zekâ modelleri, büyük miktarda veriden örüntüler öğrenerek konuşma sentezi gibi karmaşık görevleri yerine getirebiliyor.",
]
CUSTOM5 = [
    "Bu sabah erkenden kalkıp sahilde uzun bir yürüyüş yaptım; deniz o kadar sakindi ki martıların sesi bile net duyuluyordu.",
    "Yapay zeka modelleri, büyük miktarda veriden örüntüler öğrenerek konuşma sentezi gibi karmaşık görevleri artık şaşırtıcı bir doğallıkla yerine getirebiliyor.",
    "Anneannem her bayram sabahı mutfakta baklava açar, evin içi tereyağı ve fıstık kokusuyla dolar, biz çocuklar da sofranın kurulmasını sabırsızlıkla beklerdik.",
    "Toplantı yarın saat 14'te başlayacak; lütfen sunum dosyalarınızı öğleden önce paylaşın ve %20'lik bütçe artışı önerisini gündeme eklemeyi unutmayın.",
    "İstanbul'da akşam trafiği başlamadan Boğaz Köprüsü'nden geçmek istiyorsanız en geç dörtte yola çıkmalısınız, yoksa bir saatlik yol üçe katlanır.",
]
BATCH_SETS = ["5 örnek cümle", "Freya-TR-Eval · ilk 20", "Freya-TR-Eval · rastgele 20", "Kendi listem"]
VOICES = ["Sentez sekmesindeki referans"] + [f"Örnek ses {i + 1}" for i in range(len(EXAMPLES))]


def fail(error):
    raise gr.Error(str(error)) from error


def token_of(oauth_token):
    return getattr(oauth_token, "token", None) if oauth_token is not None else None


def checkpoint(choice, repo, file, repo_type, oauth_token):
    if choice == CUSTOM:
        return E.resolve_custom(repo, file, repo_type, token_of(oauth_token))
    return E.model_path(choice)


def reference_inputs(ref_audio, ref_text):
    """Main-process reference preparation: trimmed audio, normalized transcript (or None = transcribe on GPU)."""
    audio, notes = E.load_reference(ref_audio)
    transcript = None
    if ref_text and ref_text.strip():
        transcript, changes = speakable(ref_text)
        if changes:
            notes.append("referans transkripti okunuşa çevrildi")
    else:
        cut = E.cut_for_transcript(audio)
        if len(cut) < len(audio):
            notes.append(f"otomatik transkript için referans {len(cut) / E.SAMPLE_RATE:.1f} s'ye kısaltıldı")
            audio = cut
    return audio, transcript, notes


def sampler_settings(guide_mode, rescale, apg_eta, speaker_guidance, guidance_until):
    sampler = {"guidance_until": float(guidance_until)}
    if guide_mode == GUIDE_RESCALE:
        sampler["cfg_rescale"] = float(rescale)
    elif guide_mode == GUIDE_APG:
        sampler.update(apg_eta=float(apg_eta), apg_momentum=-0.3)
    elif guide_mode == GUIDE_SPLIT:
        sampler["speaker_guidance"] = float(speaker_guidance)
    return sampler


def build_job(path, audio, transcript, chunks, rate_mode, cps, duration_model, candidates, guidance, steps, sampler,
              duration_scale, seed, verify):
    mode = DURATION_MODELS.get(duration_model) or (E.DEFAULTS["duration_mode"] if rate_mode == RATE_AUTO else "rule")
    return {
        "model": path,
        "reference": audio.astype(np.float32),
        "reference_text": transcript,
        "chunks": chunks,
        "candidates": int(candidates),
        "guidance": float(guidance),
        "steps": int(steps),
        "sampler": sampler,
        "duration_mode": mode,
        "duration_scale": float(duration_scale) if rate_mode == RATE_PROMPT else 1.0,
        "seconds": [max(0.5, len(c) / float(cps)) for c in chunks] if rate_mode == RATE_FIXED else None,
        "seed": int(seed),
        "verify": bool(verify),
    }


def metrics_markdown(result, final_info, audio, label, changes, chunks):
    summary = E.summarize_chunks(result["chunks"])
    seconds = len(audio) / E.SAMPLE_RATE
    text_chars = sum(len(c) for c in chunks)
    quality = E.dnsmos(audio)
    lines = [
        f"**Model:** {label} · adım {result.get('step')} · {len(chunks)} parça · "
        f"{result['meta'].get('candidates', 1)} aday/parça",
        f"**Ses:** {seconds:.1f} s · konuşma hızı {text_chars / max(seconds - 0.32, 0.1):.1f} kar/s · "
        f"üretim {result['generation_seconds']:.1f} s · GPU {result.get('gpu_seconds', 0):.1f} s"
        + (f" (model yükleme {result['model_load_seconds']:.1f} s)" if result.get("model_load_seconds", 0) > 0.5 else ""),
    ]
    metrics = {"seconds": seconds, "chunks": len(chunks), "generation_seconds": result["generation_seconds"],
               "gpu_seconds": result.get("gpu_seconds"), "duration_mode": result["meta"].get("duration_mode"),
               "input_lufs": final_info.get("input_lufs"), "clipped_fraction": final_info.get("clipped_input_fraction")}
    if summary:
        selected = result["meta"].get("candidates", 1) > 1
        lines.append(f"**Whisper ({E.ASR_NAME.split('/')[-1]}):** WER {summary['wer']:.3f} · CER {summary['cer']:.3f}"
                     + (f" · konuşmacı benzerliği {summary['similarity']:.3f}" if "similarity" in summary else "")
                     + (" · *adaylar bu Whisper ile seçildiği için iyimser bir ölçüm*" if selected else ""))
        lines.append(f"> {summary['hypothesis']}")
        metrics.update(wer=summary["wer"], cer=summary["cer"], similarity=summary.get("similarity"),
                       hypothesis=summary["hypothesis"])
    if quality:
        lines.append(f"**DNSMOS:** OVRL {quality['dnsmos_ovrl']:.2f} · SIG {quality['dnsmos_sig']:.2f} · BAK {quality['dnsmos_bak']:.2f}")
        metrics.update(quality)
    if any(len(c["candidates"]) > 1 for c in result["chunks"]):
        rows = ["| Parça | Seçilen | Aday CER'leri |", "|---:|---:|---|"]
        for i, c in enumerate(result["chunks"], 1):
            cers = ", ".join(f"{r['counts']['cer']:.3f}" for r in c["candidates"])
            rows.append(f"| {i} | {c['selected'] + 1} | {cers} |")
        lines.append("\n".join(rows))
    if result.get("reference_asr"):
        lines.append(f"**Referans transkripti (Whisper):** {result['reference_asr']}")
    if changes:
        lines.append("<details><summary>Metin ön işleme ({} değişiklik)</summary>\n\n{}\n</details>".format(
            len(changes), "\n".join(f"- {c}" for c in changes[:40])))
    return "\n\n".join(lines), metrics


def on_reference(ref_audio):
    """Upload/recording: transcribe the (trimmed) reference into the editable transcript box."""
    if not ref_audio:
        return gr.update(), ""
    try:
        audio, notes = E.load_reference(ref_audio)
        cut = E.cut_for_transcript(audio)
        text = E.run_transcribe(cut)
        transcript = speakable(text)[0] if text.strip() else ""
        if len(cut) < len(audio):
            notes.append(f"transkript ilk {len(cut) / E.SAMPLE_RATE:.1f} s için; sentezde de o kısım kullanılır")
        return text, E.reference_report(cut, transcript, notes)
    except ValueError as error:
        return gr.update(), f"⚠️ {error}"


def synthesize(ref_audio, ref_text, text, rate_mode, cps, candidates, seed, verify, choice, repo, file, repo_type,
               guidance, steps, guide_mode, rescale, apg_eta, speaker_guidance, guidance_until, duration_scale,
               duration_model, oauth_token: gr.OAuthToken | None = None):
    try:
        path, label = checkpoint(choice, repo, file, repo_type, oauth_token)
        audio, transcript, notes = reference_inputs(ref_audio, ref_text)
        rate = speaking_rate(len(audio) / E.SAMPLE_RATE * 25, transcript) if transcript else 15.0
        chunks, pauses, changes = E.plan_text(text, len(audio) / E.SAMPLE_RATE, min(max(rate, 11.0), 17.0),
                                              float(cps) if rate_mode == RATE_FIXED else None)
        job = build_job(path, audio, transcript, chunks, rate_mode, cps, duration_model, candidates, guidance, steps,
                        sampler_settings(guide_mode, rescale, apg_eta, speaker_guidance, guidance_until),
                        duration_scale, seed, verify)
        result = E.run_generate(job)
        final, final_info = E.assemble(result, pauses)
        info, metrics = metrics_markdown(result, final_info, final, label, changes, chunks)
        metrics.update(model=label, reference_transcript=result["reference_text"], normalized_text=chunks,
                       settings={k: v for k, v in job.items() if k not in {"reference", "chunks", "model"}})
        report = E.reference_report(audio, result["reference_text"], notes)
        shown = ref_text if ref_text and ref_text.strip() else (result.get("reference_asr") or "")
        return E.write_wav(final), info, "\n\n".join(chunks), metrics, shown, report
    except ValueError as error:
        fail(error)


def compare(models, ref_audio, ref_text, text, rate_mode, cps, candidates, seed, repo, file, repo_type, guidance,
            steps, guide_mode, rescale, apg_eta, speaker_guidance, guidance_until, duration_scale, duration_model,
            oauth_token: gr.OAuthToken | None = None):
    try:
        models = list(models or [])[:4]
        if len(models) < 2:
            raise ValueError("Karşılaştırmak için en az iki model seçin")
        audio, transcript, _ = reference_inputs(ref_audio, ref_text)
        if transcript is None:
            transcript, _ = speakable(E.run_transcribe(audio))
        rate = speaking_rate(len(audio) / E.SAMPLE_RATE * 25, transcript)
        chunks, pauses, changes = E.plan_text(text, len(audio) / E.SAMPLE_RATE, min(max(rate, 11.0), 17.0),
                                              float(cps) if rate_mode == RATE_FIXED else None)
        sampler = sampler_settings(guide_mode, rescale, apg_eta, speaker_guidance, guidance_until)
        jobs = []
        for choice in models:
            path, label = checkpoint(choice, repo, file, repo_type, oauth_token)
            job = build_job(path, audio, transcript, chunks, rate_mode, cps, duration_model, candidates, guidance,
                            steps, sampler, duration_scale, seed, True)
            job["label"] = label
            jobs.append(job)
        results = E.run_compare(jobs)
        audios, table = [], []
        for result in results:
            final, info = E.assemble(result, pauses)
            summary = E.summarize_chunks(result["chunks"]) or {}
            quality = E.dnsmos(final) or {}
            audios.append(gr.update(value=E.write_wav(final, "karsilastirma"), label=result["label"], visible=True))
            table.append([result["label"], round(len(final) / E.SAMPLE_RATE, 1), round(summary.get("wer", float("nan")), 3),
                          round(summary.get("cer", float("nan")), 3), round(summary.get("similarity", float("nan")), 3),
                          round(quality.get("dnsmos_ovrl", float("nan")), 2), round(result["gpu_seconds"], 1),
                          summary.get("hypothesis", "")])
        audios += [gr.update(value=None, visible=False)] * (4 - len(audios))
        return (*audios, table)
    except ValueError as error:
        fail(error)


def batch_test(choice, sentence_set, own, voice, ref_audio, ref_text, rate_mode, cps, candidates, seed, repo, file,
               repo_type, guidance, steps, guide_mode, rescale, apg_eta, speaker_guidance, guidance_until,
               duration_scale, duration_model, oauth_token: gr.OAuthToken | None = None):
    try:
        if sentence_set == BATCH_SETS[0]:
            sentences = CUSTOM5
        elif sentence_set == BATCH_SETS[3]:
            sentences = [line.strip() for line in (own or "").splitlines() if line.strip()]
        else:
            freya = E.freya_sentences()
            sentences = freya[:20] if sentence_set == BATCH_SETS[1] else random.Random(int(seed)).sample(freya, 20)
        if not sentences:
            raise ValueError("Test edilecek cümle yok")
        sentences = sentences[: E.MAX_BATCH_SENTENCES]
        if voice == VOICES[0]:
            audio, transcript, _ = reference_inputs(ref_audio, ref_text)
        else:
            example = EXAMPLES[VOICES.index(voice) - 1]
            audio, _ = E.load_reference(str(ROOT / example["audio"]))
            transcript = speakable(example["text"])[0]
        path, label = checkpoint(choice, repo, file, repo_type, oauth_token)
        chunks = [speakable(s)[0] for s in sentences]
        job = build_job(path, audio, transcript, chunks, rate_mode, cps, duration_model, candidates, guidance, steps,
                        sampler_settings(guide_mode, rescale, apg_eta, speaker_guidance, guidance_until),
                        duration_scale, seed, True)
        started = time.time()
        result = E.run_generate(job)
        rows, table, files = [], [], []
        for i, (sentence, chunk) in enumerate(zip(sentences, result["chunks"])):
            final, _ = E.assemble({"chunks": [chunk]}, [0.0])
            wav = E.write_wav(final, f"{i:03d}")
            files.append((f"{i:03d}.wav", wav))
            chosen = chunk["candidates"][chunk["selected"]]
            quality = E.dnsmos(final) or {}
            row = {"id": i, "text": sentence, "normalized": chunk["text"], "hypothesis": chosen["hypothesis"],
                   **{k: chosen["counts"][k] for k in ("wer", "cer", "word_edits", "words", "char_edits", "chars")},
                   "similarity": chosen["similarity"], "seconds": chosen["seconds"], **quality}
            rows.append(row)
            table.append([i, sentence, chosen["hypothesis"], round(row["wer"], 3), round(row["cer"], 3),
                          round(row["similarity"] or 0, 3), round(quality.get("dnsmos_ovrl", float("nan")), 2),
                          round(row["seconds"], 1)])
        words, chars = sum(r["words"] for r in rows), sum(r["chars"] for r in rows)
        summary = {
            "model": label, "sentences": len(rows), "set": sentence_set, "voice": voice,
            "wer": sum(r["word_edits"] for r in rows) / max(words, 1),
            "cer": sum(r["char_edits"] for r in rows) / max(chars, 1),
            "similarity": float(np.mean([r["similarity"] for r in rows])),
            "dnsmos_ovrl": float(np.mean(scores)) if (scores := [r["dnsmos_ovrl"] for r in rows if "dnsmos_ovrl" in r]) else None,
            "sentences_without_error": sum(r["wer"] == 0 for r in rows),
            "settings": {k: v for k, v in job.items() if k not in {"reference", "chunks", "model"}},
            "seconds": time.time() - started,
        }
        text = (f"**{label}** · {len(rows)} cümle · WER **{summary['wer']:.3f}** · CER **{summary['cer']:.3f}** · "
                f"SIM {summary['similarity']:.3f}"
                + (f" · DNSMOS {summary['dnsmos_ovrl']:.2f}" if summary.get("dnsmos_ovrl") else "")
                + f" · hatasız {summary['sentences_without_error']}/{len(rows)} · {summary['seconds']:.0f} s")
        return text, table, E.make_zip(files, rows, summary)
    except ValueError as error:
        fail(error)


def list_files(repo, repo_type, oauth_token: gr.OAuthToken | None = None):
    try:
        kind, files = E.list_checkpoint_files(repo, repo_type, token_of(oauth_token))
    except Exception as error:
        raise gr.Error(f"Depo okunamadı: {str(error)[:200]}") from error
    if not files:
        raise gr.Error("Bu depoda .pt/.pth dosyası yok")
    return gr.update(choices=files, value=files[-1]), gr.update(value=kind)


def toggle_rate(mode):
    return gr.update(visible=mode == RATE_FIXED), gr.update(visible=mode == RATE_PROMPT)


def toggle_guidance(mode):
    return (gr.update(visible=mode == GUIDE_RESCALE), gr.update(visible=mode == GUIDE_APG),
            gr.update(visible=mode == GUIDE_SPLIT))


ABOUT = """
### Model
[VoiceHub/dacvae-tts-tr-w512](https://huggingface.co/VoiceHub/dacvae-tts-tr-w512): 66,5M parametreli akış eşleştirme (flow
matching) DiT; donmuş Meta DACVAE latent uzayında (48 kHz, 25 kare/s) sıfırdan eğitildi (`Vyvo/tr-dataset-12`, ~70 saat
Türkçe podcast). Checkpoint ve ayar seçimi yalnızca **Freya-TR-Eval**'e göre yapılır (eğitimde görülmemiş 495 cümle,
görülmemiş 24 konuşmacı, faster-whisper large-v3; iki ayrı konuşmacı çekilişi: seed 42 ve 1000).

| Ayar (aynı model, guidance 5, 32 adım) | WER % (s42) | CER % (s42) | WER % (s1000) | CER % (s1000) |
|---|---:|---:|---:|---:|
| Prompt hızı kuralı (yayımlanan sayılar) | 4,32 | 2,50 | 4,40 | 2,20 |
| Sabit 15 kar/s (eski "Sabit hız") | 5,96 | 3,39 | – | – |
| Hızlı prompt sınırı | 3,61 | 2,03 | 3,73 | 1,99 |
| **Otomatik süre** (yavaşta süre tahmincisi, hızlıda sınır) | 3,61 | 1,81 | 3,50 | 1,91 |
| **Otomatik süre + 3 aday (varsayılan)** | **1,59** | **0,72** | **1,92** | **0,72** |
| *Makale: FreyaTTS-183M / XTTS-v2* | *8,0 / 11,1* | *3,0 / –* | | |

Adayları Whisper-turbo seçer, puanı Whisper-large-v3 verir (turbo, large-v3'ten damıtıldığı için kazancın bir kısmı ortak
ASR tercihlerini yansıtabilir). Benzerlik (0,947) ve DNSMOS (2,93) varsayılan ayarlarla değişmedi.

### Ayarlar
- **Konuşma hızı · Otomatik:** 13–17 kar/s'lik prompt'larda prompt'un hızı korunur; daha yavaş prompt'larda (uzun
  duraklamalar) korpustan öğrenilmiş süre tahmincisi, daha hızlılarda ~16 kar/s'ye yavaşlatma kullanılır. *Prompt'un hızı*
  kuralı sınırsız kopyalar (+ süre ölçeği). *Sabit hız* Freya'da daha çok hata yaptı (15 kar/s: WER %6,0).
- **Aday sayısı (best-of-N):** her cümle N farklı gürültüyle tek toplu çağrıda üretilir; Whisper'ın en az hata yaptığı aday
  seçilir. Gösterilen WER bu seçimi yapan Whisper'la ölçüldüğü için iyimserdir (bağımsız ölçüm yukarıdaki tabloda).
- **Uzun metin:** cümlelere bölünür (referans + parça ≈ eğitimdeki ≤ 20–25 s), parçalar tek toplu GPU çağrısında üretilip
  kısa duraklarla birleştirilir. Sayılar, tarih/saat, para birimleri (ek uyumuyla), birimler, kısaltmalar ve semboller
  okunuşa çevrilir; modelin okuduğu metin "Modelin okuduğu metin" bölümünde görünür.
- **Guidance:** CFG 5 en düşük WER'i verir ama çıktıyı doyurur (yüksek ses, decoder tavanında kırpılma). *APG* bunu büyük
  ölçüde giderir (daha doğal seviye, biraz daha yüksek DNSMOS/benzerlik) ama Freya WER'i %4,3'ten %5,0'a çıkarır;
  *CFG + rescale* benzer (%5,2). *Ayrı metin/konuşmacı* üç dallı bağımsız guidance'tır (metin = Guidance, konuşmacı = ayrı ölçek).
- Çıktılar −16 LUFS'a normalleştirilir, kısa fade ve boşlukla 48 kHz / 16-bit WAV olarak verilir. DACVAE decoder'ı Meta'nın
  gömülü filigranını (watermark) korur.

### Kendi modelinizi test etme
*Gelişmiş ayarlar → Checkpoint → Özel checkpoint*: `kurum/ad` ya da tam Hub URL'si girin, **Dosyaları listele** ile `.pt`
seçin. Özel depolar için önce **Giriş yap** (kendi okuma izninizle indirilir; Space'te token yoktur). *A/B karşılaştırma* aynı
ses/metin/seed ile 2–4 modeli yan yana üretir; *Toplu test* bir cümle listesinde (5 örnek cümle, Freya-TR-Eval'den 20 cümle
veya kendi listeniz) WER/CER/benzerlik/DNSMOS hesaplar ve tüm sesleri zip olarak verir.

### API
```python
from gradio_client import Client, handle_file
client = Client("Vyvo/dacvae-tts-tr-demo", token="hf_...")   # token: ZeroGPU kotası hesabınızdan kullanılır
audio, info, text, metrics, transcript, reference = client.predict(
    handle_file("referans.wav"), "Referans kaydın tam transkripti.", "Söylenecek metin.",
    "Otomatik (önerilen)", 15, 3, 42, True,                       # hız modu, sabit hız, aday sayısı, seed, doğrulama
    "w512-clean 60k · yayımlanan (Freya WER %4,3)", "", "", "dataset",   # checkpoint (veya özel depo/dosya/tür)
    5.0, 32, "CFG (en düşük WER)", 0.7, 0.5, 3.0, 1.0, 1.0, "Otomatik",  # guidance, adım, guidance türü ve parametreleri
    api_name="/synthesize")
```

### Sınırlar
Eğitim verisi podcast MP3'leri (çoğunlukla 12–16 kHz bant genişliği) olduğundan DNSMOS ≈ 2,9 (codec tavanı 3,27).
Nadir yabancı özel adlarda harf hataları olabilir; kalite referans kaydına bağlıdır (temiz, tek konuşmacı, 3–15 s).
Lisans: CC-BY-NC-4.0 (ticari olmayan kullanım). Freya-TR-Eval cümleleri CC-BY-4.0 (freyavoice).
"""

with gr.Blocks(title="DACVAE-TTS Türkçe") as demo:
    gr.Markdown(
        "# DACVAE-TTS Türkçe · sıfır-atış ses klonlama\n"
        "3–15 saniyelik temiz bir Türkçe kayıt yükleyin; transkripti otomatik çıkarılır (düzeltebilirsiniz). Metniniz aynı "
        "sesle 48 kHz üretilir; uzun metinler cümlelere bölünür, sayılar ve semboller okunuşa çevrilir. "
        "Model: [VoiceHub/dacvae-tts-tr-w512](https://huggingface.co/VoiceHub/dacvae-tts-tr-w512) · 66,5M · Freya-TR-Eval WER "
        "%4,3 (tek örnek) → **%1,6 bu demonun varsayılan ayarlarıyla** (otomatik hız + 3 aday; ayrıntı: Hakkında).\n\n"
        "ℹ️ GPU kotası (ZeroGPU) Hugging Face hesabına göre verilir: siteye giriş yapmamış ziyaretçiler yalnızca birkaç istek "
        "yapabilir, giriş yaptığınızda kendi kotanız kullanılır. API'de `Client(\"Vyvo/dacvae-tts-tr-demo\", token=\"hf_...\")` kullanın."
    )
    with gr.Tabs():
        with gr.Tab("🎙️ Sentez"):
            with gr.Row():
                with gr.Column():
                    ref_audio = gr.Audio(label="Referans ses (3–15 s, tek konuşmacı)", type="filepath",
                                         sources=["upload", "microphone"])
                    ref_info = gr.Markdown()
                    with gr.Row():
                        ref_text = gr.Textbox(label="Referansın transkripti (otomatik doldurulur, düzeltilebilir)",
                                              lines=2, scale=4)
                        retranscribe = gr.Button("Yeniden yazıya dök", scale=1, size="sm")
                    text = gr.Textbox(label="Söylenecek metin (uzun metin de olur)", lines=5, value=SENTENCES[0])
                    with gr.Row():
                        rate_mode = gr.Radio([RATE_AUTO, RATE_PROMPT, RATE_FIXED], value=RATE_AUTO, label="Konuşma hızı")
                        cps = gr.Slider(10, 20, value=15, step=0.5, visible=False,
                                        label="Sabit hız (kar/s) · Freya'da prompt hızından daha çok hata (15: %6,0 WER)")
                        duration_scale = gr.Slider(0.8, 1.3, value=1.0, step=0.05, label="Süre ölçeği", visible=False)
                    with gr.Row():
                        candidates = gr.Slider(1, 4, value=E.DEFAULTS["candidates"], step=1,
                                               label="Aday sayısı (best-of-N, Whisper seçer)")
                        seed = gr.Number(value=42, precision=0, label="Seed")
                        dice = gr.Button("🎲", size="sm", scale=0, min_width=40)
                    verify = gr.Checkbox(value=True, label="Whisper ile doğrula (WER/CER + konuşmacı benzerliği)")
                    with gr.Accordion("Gelişmiş ayarlar", open=False):
                        choice = gr.Dropdown(list(E.CHECKPOINTS) + [CUSTOM], value=E.DEFAULT_MODEL, label="Checkpoint")
                        with gr.Group():
                            gr.Markdown("**Özel checkpoint:** `kurum/ad` veya tam Hub URL'si. Özel depolar için giriş yapın.")
                            with gr.Row():
                                repo = gr.Textbox(label="Depo", placeholder="VoiceHub/dacvae-tts-tr-w512-clean", scale=3)
                                repo_type = gr.Radio(["dataset", "model"], value="dataset", label="Depo türü", scale=1)
                            with gr.Row():
                                file = gr.Dropdown([], label="Dosya", allow_custom_value=True, scale=3)
                                list_button = gr.Button("Dosyaları listele", scale=1)
                            gr.LoginButton(value="Hugging Face ile giriş yap (özel depolar için)")
                        with gr.Row():
                            guidance = gr.Slider(1.0, 8.0, value=E.DEFAULTS["guidance"], step=0.5, label="Guidance (metin)")
                            steps = gr.Slider(8, 64, value=E.DEFAULTS["steps"], step=4, label="Euler adımı")
                        guide_mode = gr.Radio([GUIDE_CFG, GUIDE_RESCALE, GUIDE_APG, GUIDE_SPLIT], value=GUIDE_CFG,
                                              label="Guidance türü")
                        with gr.Row():
                            rescale = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Rescale φ", visible=False)
                            apg_eta = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="APG η (paralel bileşen)",
                                                visible=False)
                            speaker_guidance = gr.Slider(1.0, 8.0, value=3.0, step=0.5, label="Konuşmacı guidance",
                                                         visible=False)
                        with gr.Row():
                            guidance_until = gr.Slider(0.3, 1.0, value=1.0, step=0.05,
                                                       label="Guidance t < … iken (1 = her adım)")
                            duration_model = gr.Dropdown(list(DURATION_MODELS), value="Otomatik", label="Süre modeli")
                    button = gr.Button("Sentezle", variant="primary")
                with gr.Column():
                    audio_out = gr.Audio(label="Üretilen ses (48 kHz)", type="filepath", buttons=["download"])
                    info = gr.Markdown()
                    with gr.Accordion("Modelin okuduğu metin", open=False):
                        normalized = gr.Markdown()
                    with gr.Accordion("Ölçümler (JSON)", open=False):
                        metrics = gr.JSON()
            gr.Examples(
                examples=[[str(ROOT / e["audio"]), e["text"], s] for e, s in zip(EXAMPLES, SENTENCES)],
                inputs=[ref_audio, ref_text, text],
                label="Örnek referanslar (eğitimde görülmemiş konuşmacılar, normal konuşma hızı)",
            )
        with gr.Tab("⚖️ A/B karşılaştırma"):
            gr.Markdown("Referans ses, transkript, metin ve ayarlar **Sentez** sekmesinden alınır; seçilen modeller aynı "
                        "seed ile üretilir ve Whisper + WavLM + DNSMOS ile ölçülür.")
            compare_models = gr.CheckboxGroup(list(E.CHECKPOINTS) + [CUSTOM], value=list(E.CHECKPOINTS)[:2],
                                              label="Modeller (2–4)")
            compare_button = gr.Button("Karşılaştır", variant="primary")
            with gr.Row():
                compare_audio = [gr.Audio(type="filepath", visible=False, buttons=["download"]) for _ in range(4)]
            compare_table = gr.Dataframe(headers=["Model", "Süre (s)", "WER", "CER", "SIM", "DNSMOS", "GPU (s)",
                                                  "Whisper"], wrap=True)
        with gr.Tab("🧪 Toplu test"):
            gr.Markdown(f"Bir checkpoint'i cümle listesinde ölçün (en fazla {E.MAX_BATCH_SENTENCES} cümle). Ayarlar "
                        "**Sentez** sekmesinden alınır; sonuçlar ve sesler zip olarak indirilebilir.")
            with gr.Row():
                batch_model = gr.Dropdown(list(E.CHECKPOINTS) + [CUSTOM], value=E.DEFAULT_MODEL, label="Checkpoint")
                batch_voice = gr.Radio(VOICES, value=VOICES[1], label="Ses")
            batch_set = gr.Radio(BATCH_SETS, value=BATCH_SETS[0], label="Cümle seti")
            batch_own = gr.Textbox(lines=6, label="Kendi cümleleriniz (satır başına bir)")
            batch_button = gr.Button("Testi çalıştır", variant="primary")
            batch_summary = gr.Markdown()
            batch_table = gr.Dataframe(headers=["#", "Metin", "Whisper", "WER", "CER", "SIM", "DNSMOS", "Süre (s)"],
                                       wrap=True)
            batch_zip = gr.File(label="Sesler + results.jsonl + summary.json")
        with gr.Tab("🎧 Örnekler"):
            if SAMPLES:
                gr.Markdown("Varsayılan ayarlarla önceden üretilmiş örnekler (görülmemiş konuşmacılar).")
                for sample in SAMPLES:
                    gr.Audio(value=str(ROOT / sample["audio"]), label=sample["label"], type="filepath", interactive=False)
            else:
                gr.Markdown("Örnekler hazırlanıyor.")
        with gr.Tab("ℹ️ Hakkında"):
            gr.Markdown(ABOUT)
            with gr.Accordion("Ortam tanılama", open=False):
                diag_button = gr.Button("Ortamı test et")
                diag_out = gr.Textbox(label="Tanılama", lines=4)
                diag_button.click(lambda: E.run_diagnostics(), [], [diag_out], api_name="diag")

    shared = [rate_mode, cps, candidates, seed]
    advanced = [guidance, steps, guide_mode, rescale, apg_eta, speaker_guidance, guidance_until, duration_scale,
                duration_model]
    ref_audio.upload(on_reference, [ref_audio], [ref_text, ref_info], api_name="transcribe")
    ref_audio.stop_recording(on_reference, [ref_audio], [ref_text, ref_info], api_name=False)
    retranscribe.click(on_reference, [ref_audio], [ref_text, ref_info], api_name=False)
    rate_mode.change(toggle_rate, [rate_mode], [cps, duration_scale], api_name=False)
    guide_mode.change(toggle_guidance, [guide_mode], [rescale, apg_eta, speaker_guidance], api_name=False)
    dice.click(lambda: random.randint(0, 2**31 - 1), [], [seed], api_name=False)
    list_button.click(list_files, [repo, repo_type], [file, repo_type], api_name=False)
    button.click(
        synthesize,
        [ref_audio, ref_text, text, *shared, verify, choice, repo, file, repo_type, *advanced],
        [audio_out, info, normalized, metrics, ref_text, ref_info],
        api_name="synthesize",
    )
    compare_button.click(
        compare,
        [compare_models, ref_audio, ref_text, text, *shared, repo, file, repo_type, *advanced],
        [*compare_audio, compare_table],
        api_name="compare",
    )
    batch_button.click(
        batch_test,
        [batch_model, batch_set, batch_own, batch_voice, ref_audio, ref_text, *shared, repo, file, repo_type, *advanced],
        [batch_summary, batch_table, batch_zip],
        api_name="batch_test",
    )

if __name__ == "__main__":
    demo.queue(max_size=32, default_concurrency_limit=4).launch(ssr_mode=False, show_error=True, theme=gr.themes.Soft())
