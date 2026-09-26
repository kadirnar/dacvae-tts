# tr-combined: GPU evidence for the roadmap issues and a new Turkish model (25 September 2026)

Everything below ran on one RTX 5090 (32 GB). Branch: `trc/tr-combined-experiments`. Tooling: `scripts/trc/`.
Checkpoints, per-checkpoint scores and listening sets: private `VoiceHub/dacvae-tts-trc-*` model repos.

**Headline.** With identical inference (single sample, no reranking), the new 60k model trained on
[Codyfederer/tr-combined](https://huggingface.co/datasets/Codyfederer/tr-combined) with cross-utterance prompts
beats run C (`VoiceHub/dacvae-tts-tr-w512`) on every metric:

| 60k updates, duration predictor refit on tr-combined, seeds 42+1000 | WER % | CER % | SIM-o | DNSMOS | UTMOS |
|---|---:|---:|---:|---:|---:|
| run C (Vyvo/tr-dataset-12, 70 h) | 5.10 | 2.93 | 0.519 | 2.860 | 2.493 |
| **new: tr-combined + cross prompts** (`VoiceHub/dacvae-tts-trc-full-cross`) | **2.94** | **1.66** | **0.536** | **2.936** | **2.572** |
| paired difference [95 % CI] | -2.16 [-3.27, -1.05] | -1.26 [-2.18, -0.34] | +0.017 [0.004, 0.029] | +0.076 [0.042, 0.110] | +0.078 [0.023, 0.133] |

With the plain prompt-rate rule: WER 7.99 → 6.46, CER 5.30 → 4.40 (ties), SIM-o/DNSMOS/UTMOS wins. Side by side:
`VoiceHub/dacvae-tts-trc-comparison` (prompt / run C / new, 40 sentences).

## 1. Data

| step | result |
|---|---|
| corpus | 221,531 clips, 277 h, 2,158 labels in 894 source recordings; 44.1 kHz WAV; median 3.6 s |
| encoding | DACVAE fast backend, -16 LUFS, 1–20 s, turkish-v2; streamed shard by shard (`scripts/data/stream_encode_score.sh`) |
| rejected | 4,985 rows: transcripts with `[gülme]`-style tags and other unsupported characters (1.9 %), > 20 s clips |
| scores | Whisper large-v3 CER vs the transcript (turkish-v2 metric) and DNSMOS P.835 for every clip (GPU DNSMOS port) |
| filter | CER ≤ 0.10, DNSMOS OVRL ≥ 2.3, ≥ 2 words: keeps 77 % (the old OVRL ≥ 2.8 rule would drop 45 %) |
| split | by source recording (`--split-map`): train 161,904 clips / 215 h / 1,967 labels; val 3,331; test 1,191 |

Filter yields per threshold: `scripts/trc/filter_report.py`. One shard (18) was much noisier than the rest (median
CER 7.5 %, OVRL 2.33); the filter handles it.

## 2. Evaluation protocol

Freya-TR-Eval (495 sentences) × **48 leak-free Common Voice 17 test voices** (`make_prompt_set.py`, 24 f / 24 m,
best-DNSMOS clip each, Freya-overlapping CV sentences excluded), guidance 5, 32 Euler steps, protocol v2
(faster-whisper large-v3 deterministic, SIM-o WavLM-large ECAPA + SpeechBrain ECAPA, UTMOS22, DNSMOS, clipping,
LUFS), turkish-v2 metric. Sampling seeds 42 and 1000 pooled. Paired speaker-clustered jackknife-t intervals.

- **Run C re-measured:** WER 7.54 / CER 4.89 / SIM-o 0.535 (published 4.3 / 2.5 used podcast prompts whose voices
  occur in training).
- **Ceilings** (`scripts/trc/cv_ceiling.py`, 168 other CV recordings of the prompt speakers): real speech SIM-o 0.708,
  DNSMOS 3.12; DACVAE round trip SIM-o 0.694, DNSMOS 3.07.
- **Noise floors.** Sampling seed: WER ±0.9 (a tie). **Training seed: ~5 WER points at 20k updates** — the evaluation
  interval does not cover it, so an option counts only when it beats the base's two training seeds.

## 3. Training A/Bs (20k of the 60k schedule, frame budget 6000, compiled execution)

| arm | WER % | CER % | SIM-o | DNSMOS | verdict |
|---|---:|---:|---:|---:|---|
| run C recipe, training seeds 42 / 43 | 30.1 / 24.9 | 22.1 / 15.0 | 0.516 / 0.544 | 2.93 / 2.88 | base |
| run C recipe, run C's eager execution | 20.1 | 13.1 | 0.553 | 2.97 | 4.8 below the better compiled seed: the size of the seed spread itself, unresolved with one eager seed; equal-frames check shows parity (#7) |
| **+ cross-utterance prompts (#11)**, seeds 42 / 43 | **12.3 / 16.9** | **8.3 / 9.7** | 0.540 / 0.534 | 2.96 / 2.94 | **adopted** |
| run C at 20k (old data, reference) | 15.2 | 10.4 | 0.541 | 2.95 | |

On top of cross prompts:

| arm | WER % | CER % | SIM-o | DNSMOS | verdict |
|---|---:|---:|---:|---:|---|
| latent negatives (#8, λ 0.2/0.2, no cap) | 61.9 | 38.8 | 0.136 | 1.93 | **rejected: collapse** |
| tail silence + short targets + quiet cut (#11) | 13.0 | 8.9 | 0.544 | 2.97 | tie |
| character CTC targets (#11) | 13.6 | 8.6 | 0.543 | 2.94 | tie |
| **character units** (one token per Turkish letter) | **11.1 / 11.2** (seeds 42 / 1000) | **7.4 / 7.4** | 0.552 | 2.96 | best arm; training seed 43 pending |
| **quality condition** (new, below) | 13.6 | 8.9 | 0.531 | 3.02 | **adopted** (quality control, WER neutral) |
| **speech-REPA (#10)**, block 10 → mHuBERT-147 L12 (seeds 42+1000) | **5.6** | **3.8** | **0.585** | 2.88 | **adopted**: WER -6.7 [-8.0, -5.5], alignment ~4× earlier; DNSMOS -0.08, UTMOS -0.24 |

**Why cross prompts matter so much here:** tr-combined clips are short, so within-utterance prompts are 0.4–2.2 s
while inference prompts are 3.5–12 s. **Why latent negatives collapse:** the per-frame optimum of
|F - F+|² - 0.2|F - F_rand|² - 0.2|F - F_aug|² is F+ + ⅓(F+ - F_rand) + ⅓(F+ - F_aug), a regression target pushed away
from real speech (SIM-o 0.14).

**Quality condition** (`model.quality_condition`, new): the target clip's DNSMOS [SIG, BAK, OVRL] enters the voice
condition through a zero-init linear map (QA-MDT arXiv 2405.15863; Lyth & King arXiv 2402.01912). Same checkpoint,
requested quality only (paired, no training variance):

| requested SIG/BAK/OVRL | WER % | CER % | SIM-o | DNSMOS | UTMOS |
|---|---:|---:|---:|---:|---:|
| unknown 3/3/3 | 13.07 | 8.59 | 0.534 | 2.992 | 2.441 |
| p90 3.6/4.1/3.3 (default) | 13.60 | 8.86 | 0.531 | 3.023 | 2.491 |
| high 4.0/4.5/3.8 | 12.84 | 8.78 | 0.533 | **3.128 (+0.135)** | **2.629 (+0.188)** |

## 4. The full-length model and post-training

`full-cross` = run C recipe + cross prompts, 60k updates (2.6 h). Quick-set trajectory (96 sentences, WER): 10k 27.9,
20k 12.0, 30k 9.9, 40k 7.2, 50k 8.3. Final numbers: headline table above.

| post-training from full-cross 60k (#14, #16) | sampling | WER % | CER % | SIM-o | DNSMOS | verdict |
|---|---|---:|---:|---:|---:|---|
| base 60k | CFG 5 | 6.46 | 4.40 | 0.548 | 2.970 | |
| control fine-tune, 8k, w = 0 | CFG 5 | 6.58 | 4.42 | 0.547 | 2.964 | tie |
| model-guidance fine-tune, 8k, w = 0.7 | no CFG | 9.01 | 5.82 | 0.526 | 3.022 | rejected (half NFE, clipping ÷5) |
| Flow-GRPO, 600 updates, composite reward (seed 42) | CFG 5 | 6.83 | 4.65 | 0.553 | 2.968 | no effect at this budget |

## 5. Inference options (#12, #13; run C unless noted; single sample)

| option | WER % | CER % | ΔWER [95 % CI] | SIM-o | DNSMOS | verdict |
|---|---:|---:|---|---:|---:|---|
| prompt-rate rule (base) | 7.54 | 4.89 | – | 0.535 | 2.903 | |
| auto / predictor duration | 5.50 / 5.55 | 3.12 / 3.17 | -2.05 / -1.99 | 0.532 | 2.93 | win |
| articulation rule | 6.98 | 4.76 | -0.56 [-2.74, 1.61] | 0.506 | 2.770 | loss |
| clamp | 7.49 | 4.86 | tie | 0.535 | 2.903 | tie |
| CFG only t < 0.5 (g 5 / 6 / 7) | 7.57 / 7.82 / 7.34 | | ties | 0.51 | 2.92–2.97 | SIM-o ↔ quality trade-off |
| g 5 early, g 2 late | 7.47 | 4.98 | tie | 0.521 | 2.959 | gentlest trade-off |
| APG η 0.5 late | 7.57 | 4.92 | tie | 0.535 | 2.905 | no effect |
| pre-tanh gain auto, level-matched | 7.57 | 4.89 | tie | 0.535 | +0.006 | UTMOS -0.059: rejected |
| moment matching | 7.39 | 4.95 | tie | 0.533 | 2.829 | loss |
| speaker guidance 3 | 8.16 | 5.22 | tie | 0.486 | 2.905 | loss |
| **duration predictor refit on tr-combined (new model)** | **2.94** | **1.66** | -1.27 vs auto | 0.536 | 2.936 | **win** |

The remaining errors of the new model with the prompt-rate rule are mid-sentence repetitions ("ince ince ince"), not
end fillers: an over-long target gets filled with repeats, which the refit predictor removes (log-MAE 0.200 → 0.137;
slow prompts 0.341 → 0.168).

## 6. Speed (#7)

RTX 5090, frame budget 6000: run C's execution 0.273 s/update; compiled blocks + selective checkpointing + symbolic
lengths 0.14 s/update (1.8× frames/s, ~8 GB). Pad lengths to multiples of 8 (not 64: with short clips 64 wasted 20 %
of every batch; unpadded text crashes the compiled attention).

## 7. Bugs found and fixed on this branch

- Tail-silence batch costs used the maximum for every row (-14 % batch size); text negatives corrupted prompt words on
  cross-prompt rows; dropout models fell back to eager at update 1 under compiled blocks; `compile_blocks` recompile
  limit 64 → 256.
- Evaluation rows that failed in scoring (CUDA OOM) were averaged away; `--rescore` kept them. Now rescored, and the
  arm runner fails loudly.
- `Synthesizer.generate` dropped the quality condition; a stale test stub; protobuf 7 broke wandb; onnx2torch replaced
  the CUDA torch build (setup now installs it with `--no-deps`).

## 8. Open

#9 (DiT block options), #10 (TLA-SA and REPA+TLA), character units (training seed 43), #14 WSD/regularization, #15
(other corpora), and the second full-length model (training): cross prompts + quality condition + speech-REPA + character units. Raw logs: `outputs/trc/RESULTS.md` on the GPU machine.
