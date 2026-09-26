"""DACVAE-TTS Turkish demo: zero-shot voice cloning, checkpoint comparison and Whisper verification."""

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

CUSTOM = "Custom checkpoint (repository below)"
RATE_AUTO, RATE_PROMPT, RATE_FIXED = "Automatic (recommended)", "Prompt rate", "Fixed rate"
GUIDE_CFG, GUIDE_RESCALE, GUIDE_APG, GUIDE_SPLIT = "CFG (lowest WER)", "CFG + rescale", "APG (cleaner audio)", "Separate text/speaker"
DURATION_MODELS = {"Automatic": None, "Duration predictor (refit on tr-combined)": "predictor",
                   "Prompt rate (rule)": "rule", "Rule + fast-prompt clamp": "clamp",
                   "Hybrid (predictor when slow, clamp when fast)": "auto", "Syllable rule": "syllable"}
EXAMPLES = json.loads((ROOT / "examples" / "prompts.json").read_text(encoding="utf-8"))
SAMPLES = json.loads((ROOT / "samples" / "samples.json").read_text(encoding="utf-8")) if (ROOT / "samples" / "samples.json").exists() else []
# Turkish example inputs for the model (the interface is English, the speech is Turkish).
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
BATCH_SETS = ["5 sample sentences", "Freya-TR-Eval · first 20", "Freya-TR-Eval · random 20", "My own list"]
VOICES = ["Reference from the Synthesis tab"] + [f"Example voice {i + 1}" for i in range(len(EXAMPLES))]


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
            notes.append("reference transcript converted to its spoken form")
    else:
        cut = E.cut_for_transcript(audio)
        if len(cut) < len(audio):
            notes.append(f"reference shortened to {len(cut) / E.SAMPLE_RATE:.1f} s for the automatic transcript")
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
        f"**Model:** {label} · step {result.get('step')} · {len(chunks)} chunks · "
        f"{result['meta'].get('candidates', 1)} candidates/chunk",
        f"**Audio:** {seconds:.1f} s · speaking rate {text_chars / max(seconds - 0.32, 0.1):.1f} chars/s · "
        f"generation {result['generation_seconds']:.1f} s · GPU {result.get('gpu_seconds', 0):.1f} s"
        + (f" (model loading {result['model_load_seconds']:.1f} s)" if result.get("model_load_seconds", 0) > 0.5 else ""),
    ]
    metrics = {"seconds": seconds, "chunks": len(chunks), "generation_seconds": result["generation_seconds"],
               "gpu_seconds": result.get("gpu_seconds"), "duration_mode": result["meta"].get("duration_mode"),
               "input_lufs": final_info.get("input_lufs"), "clipped_fraction": final_info.get("clipped_input_fraction")}
    if summary:
        selected = result["meta"].get("candidates", 1) > 1
        lines.append(f"**Whisper ({E.ASR_NAME.split('/')[-1]}):** WER {summary['wer']:.3f} · CER {summary['cer']:.3f}"
                     + (f" · speaker similarity {summary['similarity']:.3f}" if "similarity" in summary else "")
                     + (" · *optimistic, since the candidates were selected with this Whisper*" if selected else ""))
        lines.append(f"> {summary['hypothesis']}")
        metrics.update(wer=summary["wer"], cer=summary["cer"], similarity=summary.get("similarity"),
                       hypothesis=summary["hypothesis"])
    if quality:
        lines.append(f"**DNSMOS:** OVRL {quality['dnsmos_ovrl']:.2f} · SIG {quality['dnsmos_sig']:.2f} · BAK {quality['dnsmos_bak']:.2f}")
        metrics.update(quality)
    if any(len(c["candidates"]) > 1 for c in result["chunks"]):
        rows = ["| Chunk | Selected | Candidate CERs |", "|---:|---:|---|"]
        for i, c in enumerate(result["chunks"], 1):
            cers = ", ".join(f"{r['counts']['cer']:.3f}" for r in c["candidates"])
            rows.append(f"| {i} | {c['selected'] + 1} | {cers} |")
        lines.append("\n".join(rows))
    if result.get("reference_asr"):
        lines.append(f"**Reference transcript (Whisper):** {result['reference_asr']}")
    if changes:
        lines.append("<details><summary>Text preprocessing ({} changes)</summary>\n\n{}\n</details>".format(
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
            notes.append(f"the transcript covers the first {len(cut) / E.SAMPLE_RATE:.1f} s; synthesis uses that part too")
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
            raise ValueError("Select at least two models to compare")
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
            audios.append(gr.update(value=E.write_wav(final, "comparison"), label=result["label"], visible=True))
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
            raise ValueError("No sentences to test")
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
        text = (f"**{label}** · {len(rows)} sentences · WER **{summary['wer']:.3f}** · CER **{summary['cer']:.3f}** · "
                f"SIM {summary['similarity']:.3f}"
                + (f" · DNSMOS {summary['dnsmos_ovrl']:.2f}" if summary.get("dnsmos_ovrl") else "")
                + f" · error-free {summary['sentences_without_error']}/{len(rows)} · {summary['seconds']:.0f} s")
        return text, table, E.make_zip(files, rows, summary)
    except ValueError as error:
        fail(error)


def list_files(repo, repo_type, oauth_token: gr.OAuthToken | None = None):
    try:
        kind, files = E.list_checkpoint_files(repo, repo_type, token_of(oauth_token))
    except Exception as error:
        raise gr.Error(f"Could not read the repository: {str(error)[:200]}") from error
    if not files:
        raise gr.Error("This repository has no .pt/.pth file")
    return gr.update(choices=files, value=files[-1]), gr.update(value=kind)


def toggle_rate(mode):
    return gr.update(visible=mode == RATE_FIXED), gr.update(visible=mode == RATE_PROMPT)


def toggle_guidance(mode):
    return (gr.update(visible=mode == GUIDE_RESCALE), gr.update(visible=mode == GUIDE_APG),
            gr.update(visible=mode == GUIDE_SPLIT))


ABOUT = """
### Models
All three are 67M-parameter flow-matching DiTs trained from scratch in the frozen Meta DACVAE latent space (48 kHz,
25 frames/s) with [dacvae-tts](https://github.com/kadirnar/dacvae-tts). Checkpoints, per-run results and logs:
[VoiceHub/dacvae-tts-tr-combined](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined)
([experiment log](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/blob/main/EXPERIMENTS.md)).

- **full-v2 (default):** trained on [Codyfederer/tr-combined](https://huggingface.co/datasets/Codyfederer/tr-combined)
  (215 h after filtering) for 60k updates with cross-utterance prompts, a DNSMOS quality condition, speech-REPA
  (mHuBERT-147 alignment) and one text token per Turkish letter.
- **full-cross:** the same data with cross-utterance prompts only.
- **run C:** the previous model ([VoiceHub/dacvae-tts-tr-w512](https://huggingface.co/VoiceHub/dacvae-tts-tr-w512),
  Vyvo/tr-dataset-12, ~70 h).

Measured on Freya-TR-Eval (495 sentences) spoken by 48 Common Voice test voices never seen in training, **one sample per
sentence** (no reranking), guidance 5, 32 steps, duration predictor refit on tr-combined, sampling seeds 42 + 1000,
Whisper large-v3:

| model | WER % | CER % | SIM-o | DNSMOS | UTMOS |
|---|---:|---:|---:|---:|---:|
| run C | 5.10 | 2.93 | 0.519 | 2.860 | 2.493 |
| full-cross | 2.94 | 1.66 | 0.536 | 2.936 | 2.572 |
| **full-v2** | **0.93** | **0.36** | **0.556** | **3.128** | 2.533 |

### Settings
- **Defaults** are the measured ones: one sample per sentence and the duration predictor refit on tr-combined. The demo
  shows the model's raw output; *Number of candidates* > 1 turns on best-of-N (Whisper picks the candidate it transcribes
  best, which makes the displayed WER optimistic).
- **Speaking rate · Automatic** uses the refit duration predictor (syllables, words, punctuation and the prompt's own
  rate). *Prompt rate* copies the prompt's rate (+ duration scale); *Fixed rate* uses the characters/s you set. With the
  prompt-rate rule full-v2 measures WER 2.21 %: an over-long target is the main source of repeated words.
- **Long text:** split into sentences (reference + chunk within the lengths seen in training); the chunks are generated in
  one batched GPU call and joined with short pauses. Numbers, dates/clock times, currencies, units, abbreviations and
  symbols are converted to their spoken form; the text the model reads is shown in "Text read by the model".
- **Guidance:** CFG 5 gives the lowest WER. *APG* and *CFG + rescale* give a more natural level at a small WER cost;
  *Separate text/speaker* is three-branch guidance (text = Guidance, speaker = its own scale).
- Outputs are normalized to −16 LUFS and delivered as 48 kHz / 16-bit WAV. The DACVAE decoder keeps Meta's embedded
  watermark.

### Testing other checkpoints
*Advanced settings → Checkpoint → Custom checkpoint*: enter `org/name` or a full Hub URL and pick a `.pt` with **List
files** (e.g. any `checkpoints/step-*.pt` of VoiceHub/dacvae-tts-tr-combined). For private repositories **Log in** first.
*A/B comparison* generates 2–4 models side by side with the same audio/text/seed; *Batch test* computes
WER/CER/similarity/DNSMOS on a sentence list and returns all audio as a zip.

### API
```python
from gradio_client import Client, handle_file
client = Client("Vyvo/dacvae-tts-tr-v2-demo", token="hf_...")   # token: the ZeroGPU quota of your account is used
audio, info, text, metrics, transcript, reference = client.predict(
    handle_file("reference.wav"), "Referans kaydın tam transkripti.", "Söylenecek metin.",  # Turkish transcript, Turkish text
    "Automatic (recommended)", 15, 1, 42, True,                   # rate mode, fixed rate, candidates, seed, verification
    "full-v2 60k · tr-combined (WER 0.93 %)", "", "", "model",   # checkpoint (or custom repo/file/type)
    5.0, 32, "CFG (lowest WER)", 0.7, 0.5, 3.0, 1.0, 1.0, "Automatic",  # guidance, steps, guidance type and parameters
    api_name="/synthesize")
```

### Limits
Rare foreign proper names may get letter errors, and quality depends on the reference recording (clean, single speaker,
3–15 s). Voice similarity is the largest remaining gap (SIM-o 0.56 vs 0.69 for a perfect DACVAE reconstruction).
License: CC-BY-NC-4.0 (non-commercial use). Freya-TR-Eval sentences: CC-BY-4.0 (freyavoice).
"""

with gr.Blocks(title="DACVAE-TTS Turkish") as demo:
    gr.Markdown(
        "# DACVAE-TTS Turkish · zero-shot voice cloning\n"
        "Upload a clean 3–15 second Turkish recording; its transcript is extracted automatically (you can correct it). Your "
        "text is generated in the same voice at 48 kHz; long texts are split into sentences, numbers and symbols are converted "
        "to their spoken form. "
        "Model: **full-v2** ([VoiceHub/dacvae-tts-tr-combined](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined)) · 67M · "
        "trained on Codyfederer/tr-combined · Freya-TR-Eval WER **0.93 %** with unseen voices, one sample per sentence (the "
        "previous model: 5.10 %). The defaults show the model's raw output; details: About.\n\n"
        "ℹ️ The GPU quota (ZeroGPU) is granted per Hugging Face account: visitors who are not logged in can make only a few "
        "requests; once you log in, your own quota is used. For the API use "
        "`Client(\"Vyvo/dacvae-tts-tr-v2-demo\", token=\"hf_...\")`."
    )
    with gr.Tabs():
        with gr.Tab("🎙️ Synthesis"):
            with gr.Row():
                with gr.Column():
                    ref_audio = gr.Audio(label="Reference audio (3–15 s, single speaker)", type="filepath",
                                         sources=["upload", "microphone"])
                    ref_info = gr.Markdown()
                    with gr.Row():
                        ref_text = gr.Textbox(label="Reference transcript (filled in automatically, editable)",
                                              lines=2, scale=4)
                        retranscribe = gr.Button("Transcribe again", scale=1, size="sm")
                    text = gr.Textbox(label="Text to speak (long text is fine)", lines=5, value=SENTENCES[0])
                    with gr.Row():
                        rate_mode = gr.Radio([RATE_AUTO, RATE_PROMPT, RATE_FIXED], value=RATE_AUTO, label="Speaking rate")
                        cps = gr.Slider(10, 20, value=15, step=0.5, visible=False,
                                        label="Fixed rate (chars/s) · more errors than the prompt rate on Freya (15: 6.0 % WER)")
                        duration_scale = gr.Slider(0.8, 1.3, value=1.0, step=0.05, label="Duration scale", visible=False)
                    with gr.Row():
                        candidates = gr.Slider(1, 4, value=E.DEFAULTS["candidates"], step=1,
                                               label="Number of candidates (best-of-N, Whisper picks)")
                        seed = gr.Number(value=42, precision=0, label="Seed")
                        dice = gr.Button("🎲", size="sm", scale=0, min_width=40)
                    verify = gr.Checkbox(value=True, label="Verify with Whisper (WER/CER + speaker similarity)")
                    with gr.Accordion("Advanced settings", open=False):
                        choice = gr.Dropdown(list(E.CHECKPOINTS) + [CUSTOM], value=E.DEFAULT_MODEL, label="Checkpoint")
                        with gr.Group():
                            gr.Markdown("**Custom checkpoint:** `org/name` or a full Hub URL. Log in for private repositories.")
                            with gr.Row():
                                repo = gr.Textbox(label="Repository", placeholder="VoiceHub/dacvae-tts-tr-w512-clean", scale=3)
                                repo_type = gr.Radio(["dataset", "model"], value="dataset", label="Repository type", scale=1)
                            with gr.Row():
                                file = gr.Dropdown([], label="File", allow_custom_value=True, scale=3)
                                list_button = gr.Button("List files", scale=1)
                            gr.LoginButton(value="Log in with Hugging Face (for private repositories)")
                        with gr.Row():
                            guidance = gr.Slider(1.0, 8.0, value=E.DEFAULTS["guidance"], step=0.5, label="Guidance (text)")
                            steps = gr.Slider(8, 64, value=E.DEFAULTS["steps"], step=4, label="Euler steps")
                        guide_mode = gr.Radio([GUIDE_CFG, GUIDE_RESCALE, GUIDE_APG, GUIDE_SPLIT], value=GUIDE_CFG,
                                              label="Guidance type")
                        with gr.Row():
                            rescale = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Rescale φ", visible=False)
                            apg_eta = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="APG η (parallel component)",
                                                visible=False)
                            speaker_guidance = gr.Slider(1.0, 8.0, value=3.0, step=0.5, label="Speaker guidance",
                                                         visible=False)
                        with gr.Row():
                            guidance_until = gr.Slider(0.3, 1.0, value=1.0, step=0.05,
                                                       label="Guidance while t < … (1 = every step)")
                            duration_model = gr.Dropdown(list(DURATION_MODELS), value="Automatic", label="Duration model")
                    button = gr.Button("Synthesize", variant="primary")
                with gr.Column():
                    audio_out = gr.Audio(label="Generated audio (48 kHz)", type="filepath", buttons=["download"])
                    info = gr.Markdown()
                    with gr.Accordion("Text read by the model", open=False):
                        normalized = gr.Markdown()
                    with gr.Accordion("Measurements (JSON)", open=False):
                        metrics = gr.JSON()
            gr.Examples(
                examples=[[str(ROOT / e["audio"]), e["text"], s] for e, s in zip(EXAMPLES, SENTENCES)],
                inputs=[ref_audio, ref_text, text],
                label="Example references (speakers unseen in training, normal speaking rate)",
            )
        with gr.Tab("⚖️ A/B comparison"):
            gr.Markdown("The reference audio, transcript, text and settings are taken from the **Synthesis** tab; the "
                        "selected models generate with the same seed and are measured with Whisper + WavLM + DNSMOS.")
            compare_models = gr.CheckboxGroup(list(E.CHECKPOINTS) + [CUSTOM], value=list(E.CHECKPOINTS)[:2],
                                              label="Models (2–4)")
            compare_button = gr.Button("Compare", variant="primary")
            with gr.Row():
                compare_audio = [gr.Audio(type="filepath", visible=False, buttons=["download"]) for _ in range(4)]
            compare_table = gr.Dataframe(headers=["Model", "Duration (s)", "WER", "CER", "SIM", "DNSMOS", "GPU (s)",
                                                  "Whisper"], wrap=True)
        with gr.Tab("🧪 Batch test"):
            gr.Markdown(f"Measure a checkpoint on a sentence list (at most {E.MAX_BATCH_SENTENCES} sentences). Settings are "
                        "taken from the **Synthesis** tab; the results and audio can be downloaded as a zip.")
            with gr.Row():
                batch_model = gr.Dropdown(list(E.CHECKPOINTS) + [CUSTOM], value=E.DEFAULT_MODEL, label="Checkpoint")
                batch_voice = gr.Radio(VOICES, value=VOICES[1], label="Voice")
            batch_set = gr.Radio(BATCH_SETS, value=BATCH_SETS[0], label="Sentence set")
            batch_own = gr.Textbox(lines=6, label="Your own sentences (one per line)")
            batch_button = gr.Button("Run the test", variant="primary")
            batch_summary = gr.Markdown()
            batch_table = gr.Dataframe(headers=["#", "Text", "Whisper", "WER", "CER", "SIM", "DNSMOS", "Duration (s)"],
                                       wrap=True)
            batch_zip = gr.File(label="Audio + results.jsonl + summary.json")
        with gr.Tab("🎧 Samples"):
            if SAMPLES:
                gr.Markdown("Samples pre-generated with the default settings (unseen speakers).")
                for sample in SAMPLES:
                    gr.Audio(value=str(ROOT / sample["audio"]), label=sample["label"], type="filepath", interactive=False)
            else:
                gr.Markdown("Samples are being prepared.")
        with gr.Tab("ℹ️ About"):
            gr.Markdown(ABOUT)
            with gr.Accordion("Environment diagnostics", open=False):
                diag_button = gr.Button("Test the environment")
                diag_out = gr.Textbox(label="Diagnostics", lines=4)
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
