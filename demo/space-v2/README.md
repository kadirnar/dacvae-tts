---
title: DACVAE-TTS Turkish v2 (Vyvo)
emoji: 🗣️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Turkish zero-shot voice cloning TTS, full-v2 (WER 0.93 %)
hf_oauth: true
hf_oauth_scopes:
- read-repos
preload_from_hub:
- VoiceHub/dacvae-tts-tr-combined full-v2/checkpoints/step-0060000.pt,duration/duration_trc.json
- facebook/dacvae-watermarked weights.pth
- openai/whisper-large-v3-turbo
- microsoft/wavlm-base-plus-sv
models:
- VoiceHub/dacvae-tts-tr-combined
- VoiceHub/dacvae-tts-tr-w512
- facebook/dacvae-watermarked
- openai/whisper-large-v3-turbo
datasets:
- Codyfederer/tr-combined
- freyavoice/freya-tr-eval
---

Turkish zero-shot voice-cloning TTS demo of **full-v2** from
[VoiceHub/dacvae-tts-tr-combined](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined): a 67M flow-matching DiT in the
frozen Meta DACVAE latent space, trained with [dacvae-tts](https://github.com/kadirnar/dacvae-tts) on
[Codyfederer/tr-combined](https://huggingface.co/datasets/Codyfederer/tr-combined). Freya-TR-Eval with 48 unseen voices,
one sample per sentence: WER 0.93 %, CER 0.36 % (previous model: 5.10 % / 2.93 %).

- **Synthesis:** 3–15 s reference (upload or microphone; transcript filled in by Whisper and editable) + any length of text.
  Long text is split into sentences and generated in one batched GPU call; numbers, dates, clock times, currencies, units,
  abbreviations, acronyms and symbols are rewritten into spoken Turkish.
- **Defaults = the measured setting:** one sample per sentence, duration predictor refit on tr-combined. Best-of-N is
  available as an option.
- **Models:** full-v2 (default), full-cross and the previous run C side by side (*A/B comparison*); any other checkpoint of
  the experiments repo via *Custom checkpoint*.
- ZeroGPU: models are loaded once at start-up; GPU time per request is a few seconds.
