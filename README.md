# DACVAE-TTS

Small zero-shot voice-cloning TTS trained from scratch on frozen [Meta DACVAE](https://github.com/facebookresearch/dacvae)
latents (48 kHz, 128 channels, 25 frames/s). Text in, reference audio in, 48 kHz speech out.
Non-autoregressive flow matching with a DiT-style generator; no speaker IDs, no pretrained
language or speech models inside the generator.

The recommended configuration is `configs/nano.yaml` (51M parameters). Everything below uses it.

## Install

```bash
bash scripts/setup.sh        # creates .venv with PyTorch (CUDA 12.8) and every dependency; CUDA=cu126 or CUDA=cpu to change
source .venv/bin/activate
```

Nothing else to install: the codec, Whisper/DNSMOS/speaker scoring and Weights & Biases are all
declared dependencies. `bash scripts/setup.sh --dev` adds pytest/ruff (`pytest -q`).

## 1. Prepare the data (tokenization and codec encoding happen here, once)

Rows need `audio` (bytes or path), `text` and a speaker column. Hugging Face Parquet shards are
streamed one at a time (download, encode, delete), one encoder process per GPU:

```bash
HF_TOKEN=... bash scripts/prepare_hf_8gpu.sh ORG/DATASET <number_of_shards> data/corpus \
  --speaker-column speaker --quality-column quality_score --min-quality 55 --reject-digits \
  --loudness -16 --max-seconds 20        # add --languages tr for a Turkish corpus
```

This writes latents, normalization statistics and ready token ids (UTF-8 bytes, no vocabulary to
learn) into `data/corpus/merged`. About 200x real time per GPU. For a local JSONL/Parquet file use
`dacvae-tts prepare` + `dacvae-tts merge` directly (see `docs/reference.md`).

## 2. Train

```bash
bash scripts/train_8gpu.sh configs/nano.yaml data/corpus/merged runs/nano --frame-budget 16000
python scripts/monitor.py --run runs/nano --cache data/corpus/merged --cases 48   # WER/SIM per snapshot
```

Add `--wandb-project NAME` (or `train.wandb_project` in the YAML; run `wandb login` once) to mirror the training/validation curves to Weights & Biases; the same flag
on `monitor.py` adds WER/SIM per snapshot and a few audio samples to the same run. The JSONL logs
in the run directory are always written.

`--frame-budget` is the padded latent frames per GPU and step (8000 fits 16 GB, 16000 about 20 GB).
Snapshots land in `runs/nano/step-*.pt` every `keep_every` updates; `last.pt` resumes exactly with
`--resume`, and `--init-from` warm-starts a new run from any snapshot.

## 3. Synthesize

```bash
dacvae-tts infer --checkpoint runs/nano/step-0200000.pt \
  --ref-audio prompt.wav --reference-text "What the prompt says." \
  --text "Anything you want it to say." --guidance 3.5 --output out.wav
```

Python: `Synthesizer("runs/nano/step-0200000.pt").synthesize(text, ref_audio="prompt.wav",
reference_text="...")`. Omit the transcript to let optional Whisper ASR produce it
(`--asr-language tr` for Turkish). Prompts are loudness-normalized the same way as training data.

## 4. Evaluate

```bash
python scripts/random_word_test.py --checkpoint CKPT --prompt prompt.wav --prompt-text "..." --output outputs/test
python scripts/multi_reference_test.py --checkpoint CKPT --prompts prompts.json --texts texts.json --output outputs/emotion
python scripts/eval_seedtts.py --checkpoint CKPT --meta seedtts/en_meta.lst --audio-root seedtts --output outputs/seed
```

All three report WER/CER with Whisper-large-v3 and speaker similarity; the first two also DNSMOS
when `--dnsmos sig_bak_ovr.onnx` is given.

## What is in the nano recipe

| Piece | Choice |
|---|---|
| Tokens | UTF-8 bytes, prepared once; prompt and target transcripts joined into one stream |
| Prompt | Cut from the target utterance during training (no speaker labels needed), 30% prompt dropout |
| Generator | 12 DiT blocks, width 448, single-frame tokens, RoPE + length-aware RoPE in text cross-attention, QK-norm, shared low-rank AdaLN |
| Text encoder | 4 depthwise-conv blocks + 4 self-attention blocks |
| Objective | EDM-style unit-variance target, logit-normal time sampling, frame-weighted loss, batch expansion ×2, auxiliary CTC head, skip/repeat transcript negatives |
| Optimizer | Muon for hidden matrices, AdamW for the rest; EMA 0.9999 |
| Sampling | Euler, 16–32 steps, classifier-free guidance ≈3.5 in one batched forward |
| Duration | Prompt speaking rate (frames per byte) |

Measured on a 40k-update checkpoint (901 h of podcast speech, one GPU): held-out WER 0.49
(Whisper small.en), unseen-text sentences WER 11%, random word lists 6%, DNSMOS ≈ prompt level.
The remaining errors are single-word repeats and drops; the contrastive term targets them and has
not yet been evaluated at scale. The reasoning behind every choice, block by block:
[docs/arastirma-raporu-2026-09-24.md](docs/arastirma-raporu-2026-09-24.md) (Turkish).

## Turkish model and demo

`configs/nano_tr_w512.yaml` trained on `Vyvo/tr-dataset-12` gives the published Turkish model
[VoiceHub/dacvae-tts-tr-w512](https://huggingface.co/VoiceHub/dacvae-tts-tr-w512) (66.5M, Freya-TR-Eval WER 4.3% with the
plain prompt-rate rule) and the demo [Vyvo/dacvae-tts-tr-demo](https://huggingface.co/spaces/Vyvo/dacvae-tts-tr-demo).
Serving pieces used by the demo, all usable from Python and the CLI:

- `dacvae_tts.frontend.speakable(text)`: free Turkish text to speakable words (dates, clock times, currencies, units,
  abbreviations, acronyms, symbols, emoji removal) before the `turkish-v1` normalization; `split_sentences` for long text.
- `--duration-mode rule|clamp|predictor|syllable|auto` (`Synthesizer.synthesize(..., duration_mode=...)`): target length
  from the prompt rate, with fast prompts slowed (`clamp`), a fitted predictor (`scripts/train_duration.py`) or the
  prompt-rate-dependent choice of the two (`auto`).
- Sampler options: `--guidance-from/--guidance-until` (interval CFG), `--cfg-rescale`, `--apg-eta/--apg-norm/--apg-momentum`
  and `--speaker-guidance` (independent text/speaker guidance with a prompt-free branch).
- `Synthesizer.synthesize_many(texts, voice, candidates=N)`: sentence chunks and best-of-N candidates in padded batches.
- `scripts/eval_sentences.py` takes the same options plus `--candidates N` (Whisper-ranked best-of-N).

Results and decisions (Turkish): [docs/turkce-arastirma-2026-09-22.md](docs/turkce-arastirma-2026-09-22.md).

## More

- [docs/reference.md](docs/reference.md): the detailed reference (all commands, flags, Tiny/Small
  baselines, DDP, post-training, evaluation protocols, limits).
- [docs/arastirma-raporu-2026-09-24.md](docs/arastirma-raporu-2026-09-24.md): block-by-block architecture,
  training and quality research for small Turkish models (Turkish), with the evaluation caveats (SIM metric,
  Freya comparability, confidence intervals).
- [docs/yol-haritasi.md](docs/yol-haritasi.md): the roadmap and its GitHub issues (tracking issue #17).
- `configs/tiny.yaml`, `configs/small.yaml`: the original baselines, untouched by the nano changes.

The DACVAE codec keeps its upstream license; this repository ships no trained voice model.
