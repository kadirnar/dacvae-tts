# tr-combined: GPU evidence for the roadmap issues and a new Turkish model (25 September 2026)

Everything below ran on one RTX 5090 (32 GB). Branch: `trc/tr-combined-experiments`. Tooling: `scripts/trc/`.
Checkpoints, per-checkpoint scores and listening sets: one folder per run in `VoiceHub/dacvae-tts-tr-combined`.

**Headline.** With identical inference (single sample, no reranking, duration predictor refit on tr-combined), both
new 60k models trained on [Codyfederer/tr-combined](https://huggingface.co/datasets/Codyfederer/tr-combined) beat
run C (`VoiceHub/dacvae-tts-tr-w512`); the second one cuts WER to less than a fifth:

| 60k updates, seeds 42+1000, [95 % CI of the difference to run C] | WER % | CER % | SIM-o | DNSMOS | UTMOS |
|---|---:|---:|---:|---:|---:|
| run C (Vyvo/tr-dataset-12, 70 h) | 5.10 | 2.93 | 0.519 | 2.860 | 2.493 |
| `full-cross`: tr-combined + cross prompts | 2.94 [-3.27, -1.05] | 1.66 [-2.18, -0.34] | 0.536 | 2.936 | 2.572 |
| **`full-v2`**: + quality condition + speech-REPA + character units | **0.93** [-5.35, -2.99] | **0.36** [-3.52, -1.62] | **0.556** | **3.128** | 2.533 (tie) |

`full-v2` vs `full-cross`: WER -2.01 [-2.55, -1.46], CER -1.31 [-1.66, -0.95], SIM-o +0.021, DNSMOS +0.192, UTMOS
-0.039 (tie). Seed 42: 468 of 495 sentences without a word error. With the plain prompt-rate rule instead of the predictor: run C 7.99 / 5.30, `full-cross` 6.46 / 4.40, `full-v2` 2.21 / 1.17 (SIM-o 0.577, DNSMOS 3.164; vs `full-cross` WER -4.24 [-5.50, -2.99]); the rule keeps the prompt's tempo, so SIM-o is higher than with the predictor (0.556). Every run, checkpoint, evaluation and listening set:
[`VoiceHub/dacvae-tts-tr-combined`](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined) (one folder per run).

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
| character units (one token per Turkish letter), training seeds 42 / 43 | 11.2 / 16.1 | 7.4 / 9.7 | 0.551 / 0.546 | 2.97 / 2.93 | same-seed ΔWER -1.1 / -0.9 (ties), SIM-o +0.011 / +0.013 (wins), RTF -30 %: small consistent gain, **adopted** |
| **quality condition** (new, below) | 13.6 | 8.9 | 0.531 | 3.02 | **adopted** (quality control, WER neutral) |
| no text-negative contrastive loss (#8 ablation) | 13.9 | 9.9 | 0.540 | 2.97 | same-seed +1.6 [0.2, 3.1], inside the seed spread: not resolved; the negatives stay |
| SwiGLU FFN (#9) | 17.5 | 9.9 | 0.524 | 2.90 | no gain: level with the seed-43 base on every metric (retest on v2) |
| attention output gate per head (#9) | 16.7 | 10.3 | 0.529 | 2.92 | no gain: level with the seed-43 base although it starts as the base function (retest on v2) |
| TLA-SA (#10), all layers, ECAPA targets, weight 0.5 | 24.7 | 14.1 | 0.455 | 2.77 | **rejected**: +12.4 WER, slower alignment at every snapshot |
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

## 8. Architecture audit and round 3

An audit of every optional architecture path (on small CPU models) found no bugs; see #9 for the list of checks.
The same audit showed why single-seed 20k A/Bs on base + cross were unresolvable: a zero-init option that starts as
exactly the base function, with the same data order, moved the 20k WER by ~5 points. That WER records when alignment
happened. Four risks were fixed: `train.weight_decay_scope: matrices`, TLA-SA masks prompt-free rows, the speaker
condition embeds the same audio at inference as its store, and a single-prompt speaker context is empty. Round 3 runs
the remaining options on the `full-v2` recipe (REPA aligns within ~5k updates) against `y-base`, full-v2's 20k
snapshot, and its seed twin `y-s43`. Every run with its result and verdict: `docs/experiments-log.md` /
[EXPERIMENTS.md](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/blob/main/EXPERIMENTS.md).

Round 3 results (20k, seeds 42+1000; base seeds y-base / y-s43 = WER 4.32 / 3.64, CER 2.98 / 2.30, SIM-o 0.587 / 0.571,
DNSMOS 3.111 / 3.112, UTMOS 2.486 / 2.506; an option counts when it beats both):

| option (on the v2 recipe) | WER % | CER % | SIM-o | DNSMOS | UTMOS | verdict |
|---|---:|---:|---:|---:|---:|---|
| value residual | 3.94 | 2.29 | 0.582 | **3.138** | **2.533** | **adopted**: DNSMOS/UTMOS above both seeds, 24 parameters |
| long skip | 3.99 | 2.76 | 0.589 | 3.110 | 2.512 | no gain |
| final adaLN | 4.04 | 2.27 | 0.588 | 3.128 | 2.495 | no gain (DNSMOS +0.016, CIs touch 0) |
| cond text pool | 4.53 | 3.07 | 0.590 | 3.112 | 2.499 | no gain |
| SwiGLU | 4.13 | 2.78 | 0.569 | 3.092 | 2.509 | no gain |
| weight decay on matrices only | 3.77 | 2.11 | 0.581 | 3.122 | 2.480 | neutral at 0.01 |
| ffn depthwise conv | 4.04 | 2.74 | 0.598 | 3.090 | 2.415 | rejected: SIM-o up, DNSMOS/UTMOS down, +22 % time |
| ECAPA speaker condition | 3.66 | 2.40 | 0.599 | 3.039 | 2.295 | rejected: SIM-o up, UTMOS -0.2 |
| dropout 0.1 + weight decay 0.05 | 4.58 | 3.05 | 0.572 | 3.043 | 2.397 | rejected: no overfitting to fix (full-v2 validation flow falls to 60k) |

The attention-gate retest and `full-v3` (v2 + value residual at width 640, 104M parameters, defined in
`scripts/trc/arms.py`) were stopped by the owner before training.

## 9. Demo and a limitation it exposed

Demo: https://huggingface.co/spaces/Vyvo/dacvae-tts-tr-v2-demo (full-v2 by default; full-cross and run C for A/B).
Testing it showed that the tr-combined models skip the words just before a sentence end (`.`, `;`, `:`) *inside* one
chunk. The same two clauses joined by a comma read perfectly. On 3 voices x 2 seeds:

| chunk | WER % |
|---|---:|
| "... on dört otuzda başlayacak**;** lütfen zamanında gelin." | 27.8 |
| same with **.** | 27.8 |
| same with **,** | 0.0 |

tr-combined clips are mostly single sentences (median 3.6 s), so the model rarely saw a sentence end mid-target. The
Freya-TR-Eval numbers are single sentences and are unaffected. The demo therefore generates one sentence per chunk and
reads in-sentence `;`/`: ` as a comma: 0 errors in 198 words on the same inputs. A training-side fix would be
multi-sentence targets (consecutive clips of one recording joined into one target).

Demo defaults for the highest audio quality (inference only, full-v2). A sweep on the quick set (96 sentences) found
two quality levers:
- **The quality condition.** Asking for DNSMOS 4.6/4.9/4.4 gave UTMOS +0.21 and DNSMOS +0.17. Asking for 5.0/5.0/5.0
  leaves the trained range: SIM-o falls to 0.36 and WER rises.
- **APG guidance** (η 0.5). UTMOS +0.06, DNSMOS +0.07, and clipping is halved.

Guidance 3, a lower late guidance, CFG rescale (UTMOS −0.09), 64 steps and the prompt-rate rule did not help quality.
The two levers add up. Confirmed on the full protocol (495 sentences, seeds 42 + 1000):

| full-v2 | WER % | CER % | SIM-o | DNSMOS | UTMOS | clipped % |
|---|---:|---:|---:|---:|---:|---:|
| trained quality target + CFG | 0.93 | 0.36 | 0.556 | 3.128 | 2.533 | 0.031 |
| **quality 4.6/4.9/4.4 + APG (demo default)** | 1.02 (tie) | 0.31 | 0.520 | **3.332** | **2.761** | **0.003** |

UTMOS +0.23 [0.19, 0.27] closes more than half of the gap to the codec ceiling (2.93). The cost is SIM-o −0.036: the
output is cleaner than the Common Voice reference recordings, so it matches them less.

## 10. Open

#10: REPA + TLA-SA (deferred). #14: WSD, uniform-t cooldown and dual EMA (untested). `full-v3`: v2 + value residual at
width 640, stopped before training. #15 (other corpora) is out of scope: the dataset owner will grow tr-combined. Raw
logs: `logs/notebook.md` in the experiments repo.
