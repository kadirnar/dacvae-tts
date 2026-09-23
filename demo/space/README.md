---
title: DACVAE-TTS Türkçe (Vyvo)
emoji: 🗣️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Turkish zero-shot voice cloning TTS (66.5M, DACVAE latents)
hf_oauth: true
hf_oauth_scopes:
- read-repos
preload_from_hub:
- VoiceHub/dacvae-tts-tr-w512 model.pt
- facebook/dacvae-watermarked weights.pth
- openai/whisper-large-v3-turbo
- microsoft/wavlm-base-plus-sv
models:
- VoiceHub/dacvae-tts-tr-w512
- facebook/dacvae-watermarked
- openai/whisper-large-v3-turbo
datasets:
- Vyvo/tr-dataset-12
- freyavoice/freya-tr-eval
---

Turkish zero-shot voice-cloning TTS demo for [VoiceHub/dacvae-tts-tr-w512](https://huggingface.co/VoiceHub/dacvae-tts-tr-w512)
(66.5M flow-matching DiT in frozen Meta DACVAE latent space, trained with [dacvae-tts](https://github.com/kadirnar/dacvae-tts)).

- **Synthesis:** 3–15 s reference (upload or microphone; transcript filled in by Whisper and editable) + any length of text.
  Long text is split into sentences and generated in one batched GPU call; numbers, dates, clock times, currencies, units,
  abbreviations, acronyms and symbols are rewritten into spoken Turkish.
- **Speaking rate:** *Otomatik* follows the prompt but slows down prompts faster than 17 characters/s (rushed podcast
  prompts were the main source of errors on Freya-TR-Eval).
- **Best-of-N:** N candidates per sentence in one batch; Whisper picks the one it transcribes best.
- **Research tools:** any checkpoint from the Hub (log in for private repos — your own read permission is used, the Space
  holds no token), A/B comparison of 2–4 checkpoints, batch test on sentence lists (custom sentences or Freya-TR-Eval) with
  WER/CER, speaker similarity and DNSMOS, downloadable as a zip.
- ZeroGPU: models are loaded once at start-up; GPU time per request is a few seconds.
