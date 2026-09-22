# DACVAE-TTS

Small zero-shot voice-cloning TTS trained from scratch on frozen [Meta DACVAE](https://github.com/facebookresearch/dacvae)
latents (48 kHz, 128 channels, 25 frames/s). Text in, reference audio in, 48 kHz speech out.
Non-autoregressive flow matching with a DiT-style generator; no speaker IDs, no pretrained
language or speech models inside the generator.

The recommended configuration is `configs/nano.yaml` (51M parameters). Everything below uses it.

## Install

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e '.[codec,data,dev]'      # + '.[eval]' for WER/SIM scoring, '.[asr]' for audio-only prompts
pytest -q
```

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

Add `--wandb-project NAME` (or `train.wandb_project` in the YAML; `pip install wandb` and
`wandb login` first) to mirror the training/validation curves to Weights & Biases; the same flag
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
not yet been evaluated at scale. Full numbers and the reasoning behind every choice:
[docs/iyilestirme-yol-haritasi.md](docs/iyilestirme-yol-haritasi.md) (Turkish).

## More

- [docs/reference.md](docs/reference.md): the detailed reference (all commands, flags, Tiny/Small
  baselines, DDP, post-training, evaluation protocols, limits).
- [docs/kapsamli-tts-arastirma-raporu.md](docs/kapsamli-tts-arastirma-raporu.md) and
  [docs/flow-dit-mimarileri-aciklamasi.md](docs/flow-dit-mimarileri-aciklamasi.md): research report
  and architecture guide (Turkish).
- `configs/tiny.yaml`, `configs/small.yaml`: the original baselines, untouched by the nano changes.

The DACVAE codec keeps its upstream license; this repository ships no trained voice model.
