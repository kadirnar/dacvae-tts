# tr-combined experiment log

Every run of the tr-combined study (DACVAE-TTS, Turkish zero-shot TTS): the question, the evidence it came from, the measured result against its baseline and the decision. Protocol: Freya-TR-Eval, 495 sentences x 48 leak-free Common Voice voices, one sample per sentence (no reranking), guidance 5, 32 steps, prompt-rate duration rule, Whisper large-v3, turkish-v2 metric; paired speaker-clustered jackknife-t 95 % intervals. A/B arms stop at 20k of the 60k schedule, where the training-seed spread is ~5 WER points: an option counts only when it beats both base seeds. Code: https://github.com/kadirnar/dacvae-tts (branch `trc/tr-combined-experiments`, registry `scripts/trc/experiments.yaml`).

| verdict | runs |
|---|---|
| reference | `base-s42`, `base-s43`, `x-s43`, `run-c-reference`, `y-base` |
| tie | `base-eager`, `x-no-negatives`, `x-pairs-tail`, `x-pairs-char-ctc`, `x-swiglu`, `x-attn-gate`, `x-long-skip`, `ft-w0`, `grpo` |
| rejected | `base-s42-pad64`, `x-latent-negatives`, `x-tla`, `ft-mg-w07` |
| adopted | `pairs-cross`, `x-char-units`, `x-char-units-s43`, `x-quality-cond`, `x-repa`, `full-cross`, `full-v2` |
| not run | `x-value-residual`, `x-ffn-conv`, `x-final-adaln`, `x-cond-text-pool`, `x-regularized`, `x-speaker-context`, `x-repa-tla`, `x-speaker-condition` |
| pending | `y-s43`, `y-long-skip`, `y-value-residual`, `y-ffn-conv`, `y-final-adaln`, `y-cond-text-pool`, `y-swiglu`, `y-attn-gate`, `y-speaker-condition`, `y-decay-matrices`, `y-regularized` |

## Models side by side (refit duration predictor, seeds 42 + 1000 pooled)

| model | WER % | CER % | SIM-o | DNSMOS | UTMOS |
|---|---:|---:|---:|---:|---:|
| run C | 5.10 | 2.93 | 0.519 | 2.860 | 2.493 |
| full-cross | 2.94 | 1.66 | 0.536 | 2.936 | 2.572 |
| full-v2 | 0.93 | 0.36 | 0.556 | 3.128 | 2.533 |

## `base-s42` (#7): reference


**Question.** Baseline. Run C's recipe with the #7 speed options (compiled blocks, selective checkpointing, pad 8) on tr-combined.

**Baseline.** none (reference run)

Final evaluation, sampling seeds 42 + 1000 pooled (base-s42/step-0020000, base-s42/step-0020000-s1000):

| WER % | CER % | SIM-o | DNSMOS | UTMOS |
|---:|---:|---:|---:|---:|
| 30.09 | 22.11 | 0.516 | 2.925 | 2.312 |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 76.4 / 46.8*, 10k: 36.4 / 21.4*, 15k: 32.2 / 24.5*, 20k: 30.3 / 22.1

**Training.** final validation flow 0.6679 (step 20000); 0.170 s/update; 16,860 target frames/update

**Verdict: reference.** WER 30.1 at 20k. Its seed-43 twin reached 24.9, so the training-seed spread at 20k is ~5 WER points; an option counts only when it beats both seeds.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/base-s42) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/base-s42/logs)

## `base-s43` (#4): reference


**Question.** Noise floor. The same recipe with training seed 43.

**Change.** `train.seed=43`

**Baseline.** `base-s42`

Final evaluation, sampling seeds 42 + 1000 pooled (base-s43/step-0020000, base-s43/step-0020000-s1000):

| metric | `base-s42` | `base-s43` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 30.09 | 24.90 | -5.19 [-9.18, -1.20] | win |
| CER % | 22.11 | 15.02 | -7.09 [-10.05, -4.14] | win |
| SIM-o | 0.516 | 0.543 | +0.027 [0.012, 0.043] | win |
| DNSMOS | 2.925 | 2.884 | -0.041 [-0.067, -0.014] | loss |
| UTMOS | 2.312 | 2.329 | +0.017 [-0.022, 0.056] | tie |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 10k: 46.0 / 27.3*, 15k: 31.5 / 18.4*, 20k: 24.5 / 15.0

**Training.** final validation flow 0.6700 (step 20000); 0.179 s/update; 16,854 target frames/update

**Verdict: reference.** 5 WER points better than seed 42 with nothing changed. This spread is the bar for every 20k A/B.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/base-s43) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/base-s43/logs)

## `base-eager` (#7): tie


**Question.** Does the compiled execution change the result? Run C's exact eager execution.

**Baseline.** `base-s42`

Final evaluation, sampling seeds 42 + 1000 pooled (base-eager/step-0020000, base-eager/step-0020000-s1000):

| metric | `base-s42` | `base-eager` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 30.09 | 20.12 | -9.97 [-13.12, -6.83] | win |
| CER % | 22.11 | 13.12 | -8.99 [-11.66, -6.33] | win |
| SIM-o | 0.516 | 0.553 | +0.037 [0.021, 0.054] | win |
| DNSMOS | 2.925 | 2.972 | +0.048 [0.023, 0.072] | win |
| UTMOS | 2.312 | 2.430 | +0.118 [0.076, 0.160] | win |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 73.3 / 51.6*, 10k: 40.0 / 25.9*, 15k: 28.7 / 19.7*, 20k: 19.8 / 12.7

**Training.** final validation flow 0.6676 (step 20000); 0.333 s/update; 17,278 target frames/update

**Verdict: tie.** 4.8 WER below the better compiled seed, the size of the seed spread itself; at equal frames seen the runs match (parity).

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/base-eager) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/base-eager/logs)

## `base-s42-pad64` (#7): rejected


**Question.** First baseline, with lengths padded to multiples of 64.

**Baseline.** `base-s42`

Final evaluation, sampling seeds 42 + 1000 pooled (base-s42-pad64/step-0020000, base-s42-pad64/step-0020000-s1000):

| metric | `base-s42` | `base-s42-pad64` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 30.09 | 27.87 | -2.22 [-5.20, 0.76] | tie |
| CER % | 22.11 | 17.76 | -4.35 [-6.80, -1.91] | win |
| SIM-o | 0.516 | 0.534 | +0.018 [0.003, 0.032] | win |
| DNSMOS | 2.925 | 2.886 | -0.039 [-0.064, -0.015] | loss |
| UTMOS | 2.312 | 2.303 | -0.009 [-0.045, 0.028] | tie |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 83.6 / 48.8*, 10k: 52.7 / 32.7*, 15k: 40.5 / 25.2*, 20k: 27.6 / 17.1

**Training.** final validation flow 0.6705 (step 20000); 0.295 s/update; 13,790 target frames/update

**Verdict: rejected.** 20 % of every batch was padding with tr-combined's short clips (13.8k vs 17.3k target frames per update). Replaced by pad 8.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/base-s42-pad64) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/base-s42-pad64/logs)

## `pairs-cross` (#11): adopted


**Question.** Cross-utterance prompts. The prompt comes from other clips of the same speaker (40 %, up to 3 clips / 12 s) instead of the start of the target clip.

**Change.** `train.cross_prompt_prob=0.4`, `train.cross_prompt_max_utterances=3`, `train.cross_prompt_max_seconds=12.0`

**Evidence.** VoiceStar (arXiv 2505.19462). tr-combined's within-utterance prompts are 0.4-2.2 s while inference prompts are 3.5-12 s.

**Baseline.** `base-s42`

Final evaluation, sampling seeds 42 + 1000 pooled (pairs-cross/step-0020000, pairs-cross/step-0020000-s1000):

| metric | `base-s42` | `pairs-cross` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 30.09 | 12.30 | -17.80 [-21.41, -14.18] | win |
| CER % | 22.11 | 8.25 | -13.87 [-16.96, -10.78] | win |
| SIM-o | 0.516 | 0.540 | +0.024 [0.005, 0.044] | win |
| DNSMOS | 2.925 | 2.961 | +0.036 [0.008, 0.065] | win |
| UTMOS | 2.312 | 2.445 | +0.133 [0.082, 0.184] | win |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 97.6 / 58.5*, 10k: 29.6 / 17.6*, 15k: 17.3 / 11.9*, 20k: 12.0 / 8.2

**Training.** final validation flow 0.6716 (step 20000); 0.176 s/update; 13,518 target frames/update

**Verdict: adopted.** WER 30.1 -> 12.3 (seed 43: 24.9 -> 16.9), better on every metric. The base of all later arms.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/pairs-cross) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/pairs-cross/logs)

## `x-s43` (#4): reference


**Question.** Noise floor of the new base (base + cross prompts, training seed 43).

**Change.** `train.seed=43`

**Baseline.** `pairs-cross`

Final evaluation, sampling seeds 42 + 1000 pooled (x-s43/step-0020000, x-s43/step-0020000-s1000):

| metric | `pairs-cross` | `x-s43` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 12.30 | 16.94 | +4.64 [3.22, 6.06] | loss |
| CER % | 8.25 | 9.66 | +1.41 [0.56, 2.26] | loss |
| SIM-o | 0.540 | 0.533 | -0.007 [-0.013, -0.001] | loss |
| DNSMOS | 2.961 | 2.936 | -0.025 [-0.042, -0.009] | loss |
| UTMOS | 2.445 | 2.394 | -0.051 [-0.072, -0.031] | loss |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 101.7 / 58.1*, 10k: 36.7 / 19.0*, 15k: 29.3 / 16.1*, 20k: 17.2 / 9.6

**Training.** final validation flow 0.6735 (step 20000); 0.153 s/update; 13,481 target frames/update

**Verdict: reference.** 16.9 vs 12.3 WER; the seed spread stays ~5 points.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-s43) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-s43/logs)

## `x-latent-negatives` (#8): rejected


**Question.** Latent negatives (contrastive flow matching against shifted/corrupted latents, lambda 0.2/0.2).

**Change.** `train.contrastive_mode=latent_delta`, `train.contrastive_random_weight=0.2`, `train.contrastive_aug_weight=0.2`, `train.contrastive_span_min=3`, `train.contrastive_span_max=125`, `train.contrastive_repeat_coverage=[0.2, 0.4]`, `train.contrastive_skip_coverage=[0.4, 0.8]`, `train.contrastive_negative_cap=0.0`

**Evidence.** RobustSpeechFlow (arXiv 2605.22083); Delta-FM (arXiv 2506.05350).

**Baseline.** `pairs-cross`

Final evaluation, sampling seeds 42 + 1000 pooled (x-latent-negatives/step-0020000, x-latent-negatives/step-0020000-s1000):

| metric | `pairs-cross` | `x-latent-negatives` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 12.30 | 62.15 | +49.85 [44.25, 55.45] | loss |
| CER % | 8.25 | 38.60 | +30.35 [25.56, 35.14] | loss |
| SIM-o | 0.540 | 0.139 | -0.402 [-0.424, -0.379] | loss |
| DNSMOS | 2.961 | 1.919 | -1.043 [-1.110, -0.975] | loss |
| UTMOS | 2.445 | 1.258 | -1.188 [-1.276, -1.099] | loss |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 104.2 / 86.4*, 10k: 95.0 / 64.7*, 15k: 75.1 / 48.2*, 20k: 61.9 / 38.8

**Training.** final validation flow 0.7433 (step 20000); 0.209 s/update; 13,518 target frames/update

**Verdict: rejected.** Collapse (WER 62, SIM-o 0.14). Without a cap the per-frame optimum is F+ + 1/3(F+ - F_rand) + 1/3(F+ - F_aug), a target pushed away from real speech.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-latent-negatives) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-latent-negatives/logs)

## `x-no-negatives` (#8): tie


**Question.** Are run C's existing text negatives (contrastive loss) useful? Turned off.

**Change.** `train.contrastive_mode=none`

**Baseline.** `pairs-cross`

Final evaluation, sampling seeds 42 + 1000 pooled (x-no-negatives/step-0020000, x-no-negatives/step-0020000-s1000):

| metric | `pairs-cross` | `x-no-negatives` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 12.30 | 13.92 | +1.62 [0.16, 3.09] | loss |
| CER % | 8.25 | 9.90 | +1.65 [0.53, 2.77] | loss |
| SIM-o | 0.540 | 0.540 | -0.001 [-0.008, 0.006] | tie |
| DNSMOS | 2.961 | 2.965 | +0.003 [-0.016, 0.023] | tie |
| UTMOS | 2.445 | 2.430 | -0.015 [-0.038, 0.009] | tie |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 101.9 / 59.3*, 10k: 34.5 / 19.5*, 15k: 17.9 / 12.3*, 20k: 14.4 / 10.1

**Training.** final validation flow 0.6713 (step 20000); 0.119 s/update; 13,518 target frames/update

**Verdict: tie.** WER +1.6 and CER +1.7 against the same seed, inside the ~5-point training-seed spread: not resolved. The text negatives stay (no evidence to remove them); removing them would save ~20 % update time.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-no-negatives) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-no-negatives/logs)

## `x-pairs-tail` (#11): tie


**Question.** Tail silence, long prompts / short targets and quiet prompt cuts.

**Change.** `train.tail_silence_prob=0.3`, `train.tail_silence_max_seconds=0.8`, `train.long_prompt_prob=0.25`, `train.prompt_fraction_long_max=0.85`, `train.prompt_cut=quiet`

**Evidence.** Robustness to end-of-utterance fillers and to short targets.

**Baseline.** `pairs-cross`

Final evaluation, sampling seeds 42 + 1000 pooled (x-pairs-tail/step-0020000, x-pairs-tail/step-0020000-s1000):

| metric | `pairs-cross` | `x-pairs-tail` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 12.30 | 13.26 | +0.96 [-0.46, 2.38] | tie |
| CER % | 8.25 | 8.89 | +0.64 [-0.45, 1.73] | tie |
| SIM-o | 0.540 | 0.543 | +0.002 [-0.003, 0.008] | tie |
| DNSMOS | 2.961 | 2.972 | +0.011 [-0.007, 0.028] | tie |
| UTMOS | 2.445 | 2.438 | -0.007 [-0.032, 0.018] | tie |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 95.4 / 59.3*, 10k: 28.0 / 16.4*, 15k: 17.5 / 11.8*, 20k: 13.0 / 8.9

**Training.** final validation flow 0.6728 (step 20000); 0.168 s/update; 12,472 target frames/update

**Verdict: tie.** WER 13.3 vs 12.3, inside the seed spread; no measurable effect.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-pairs-tail) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-pairs-tail/logs)

## `x-pairs-char-ctc` (#11): tie


**Question.** Character targets for the alignment CTC head.

**Change.** `model.ctc_targets=chars`

**Baseline.** `pairs-cross`

Final evaluation, sampling seeds 42 + 1000 pooled (x-pairs-char-ctc/step-0020000, x-pairs-char-ctc/step-0020000-s1000):

| metric | `pairs-cross` | `x-pairs-char-ctc` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 12.30 | 13.64 | +1.34 [0.15, 2.53] | loss |
| CER % | 8.25 | 8.73 | +0.48 [-0.40, 1.37] | tie |
| SIM-o | 0.540 | 0.540 | -0.000 [-0.007, 0.006] | tie |
| DNSMOS | 2.961 | 2.939 | -0.023 [-0.044, -0.001] | loss |
| UTMOS | 2.445 | 2.429 | -0.016 [-0.036, 0.004] | tie |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 98.7 / 66.6*, 10k: 26.7 / 16.4*, 15k: 19.4 / 12.3*, 20k: 13.6 / 8.6

**Training.** final validation flow 0.6705 (step 20000); 0.267 s/update; 13,518 target frames/update

**Verdict: tie.** WER 13.6 vs 12.3, inside the seed spread.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-pairs-char-ctc) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-pairs-char-ctc/logs)

## `x-char-units` (round 2): adopted


**Question.** Character units. One text token per Turkish letter instead of the BPE-like units.

**Change.** `model.text_units=chars`

**Evidence.** Turkish orthography is close to phonemic; a character vocabulary is the future-work item of the roadmap.

**Baseline.** `pairs-cross`

Final evaluation, sampling seeds 42 + 1000 pooled (x-char-units/step-0020000, x-char-units/step-0020000-s1000):

| metric | `pairs-cross` | `x-char-units` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 12.30 | 11.17 | -1.13 [-2.27, 0.02] | tie |
| CER % | 8.25 | 7.41 | -0.84 [-1.65, -0.02] | win |
| SIM-o | 0.540 | 0.551 | +0.011 [0.004, 0.017] | win |
| DNSMOS | 2.961 | 2.973 | +0.011 [-0.005, 0.028] | tie |
| UTMOS | 2.445 | 2.439 | -0.006 [-0.029, 0.017] | tie |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 93.8 / 55.8*, 10k: 30.6 / 18.1*, 15k: 16.3 / 9.9*, 20k: 11.1 / 7.4

**Training.** final validation flow 0.6712 (step 20000); 0.150 s/update; 13,518 target frames/update

**Verdict: adopted.** WER -1.1 / CER -0.8 and SIM-o +0.011 against the same seed; the seed-43 repeat agrees (x-char-units-s43). Small but consistent, and 30 % faster inference.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-char-units) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-char-units/logs)

## `x-char-units-s43` (round 2): adopted


**Question.** Character units, training seed 43 (the strict test of the rule from #4).

**Change.** `model.text_units=chars`, `train.seed=43`

**Baseline.** `x-s43`

Final evaluation, sampling seeds 42 + 1000 pooled (x-char-units-s43/step-0020000, x-char-units-s43/step-0020000-s1000):

| metric | `x-s43` | `x-char-units-s43` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 16.94 | 16.08 | -0.86 [-2.24, 0.53] | tie |
| CER % | 9.66 | 9.71 | +0.06 [-0.91, 1.02] | tie |
| SIM-o | 0.533 | 0.546 | +0.013 [0.005, 0.020] | win |
| DNSMOS | 2.936 | 2.926 | -0.009 [-0.026, 0.007] | tie |
| UTMOS | 2.394 | 2.341 | -0.053 [-0.073, -0.034] | loss |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 85.6 / 48.7*, 10k: 33.6 / 19.2*, 15k: 21.0 / 13.3*, 20k: 16.2 / 9.8

**Training.** final validation flow 0.6729 (step 20000); 0.150 s/update; 13,481 target frames/update

**Verdict: adopted.** WER -0.9 (tie), SIM-o +0.013 (win), UTMOS -0.05 against the seed-43 base. Same direction as seed 42.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-char-units-s43) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-char-units-s43/logs)

## `x-quality-cond` (new): adopted


**Question.** Quality condition. The target clip's DNSMOS [SIG, BAK, OVRL] enters the voice condition (zero-init); inference requests high quality.

**Change.** `model.quality_condition=true`, `train.quality_scores=quality/dnsmos.json`

**Evidence.** QA-MDT (arXiv 2405.15863); Lyth & King (arXiv 2402.01912).

**Baseline.** `pairs-cross`

Final evaluation, sampling seeds 42 + 1000 pooled (x-quality-cond/step-0020000, x-quality-cond/step-0020000-s1000):

| metric | `pairs-cross` | `x-quality-cond` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 12.30 | 13.22 | +0.92 [-0.44, 2.28] | tie |
| CER % | 8.25 | 8.67 | +0.43 [-0.51, 1.37] | tie |
| SIM-o | 0.540 | 0.533 | -0.007 [-0.012, -0.002] | loss |
| DNSMOS | 2.961 | 3.020 | +0.059 [0.043, 0.075] | win |
| UTMOS | 2.445 | 2.491 | +0.046 [0.027, 0.064] | win |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 97.8 / 67.9*, 10k: 29.3 / 17.3*, 15k: 18.5 / 11.6*, 20k: 13.6 / 8.9, 20k: 12.8 / 8.8, 20k: 13.1 / 8.6

**Training.** final validation flow 0.6715 (step 20000); 0.153 s/update; 13,518 target frames/update

**Verdict: adopted.** WER neutral. Requesting 4.0/4.5/3.8 raises DNSMOS by +0.135 and UTMOS by +0.19 on the same checkpoint. A quality control, not a WER option.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-quality-cond) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-quality-cond/logs)

## `x-repa` (#10): adopted


**Question.** Speech-REPA. Align block 10 with mHuBERT-147 layer 12 (PCA 256), weight 1.0.

**Change.** `model.repa_layer=10`, `model.repa_dim=256`, `train.teacher_features=teacher/mhubert147-l12-pca256`, `train.repa_weight=1.0`, `train.repa_stop_step=0`, `train.repa_frames=all`

**Evidence.** A-DMA (arXiv 2505.19595): ~2x faster alignment; BareWave (arXiv 2606.09048): WavLM REPA WER 3.32 -> 2.86.

**Baseline.** `pairs-cross`

Final evaluation, sampling seeds 42 + 1000 pooled (x-repa/step-0020000, x-repa/step-0020000-s1000):

| metric | `pairs-cross` | `x-repa` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 12.30 | 5.56 | -6.74 [-8.00, -5.47] | win |
| CER % | 8.25 | 3.76 | -4.48 [-5.39, -3.58] | win |
| SIM-o | 0.540 | 0.585 | +0.045 [0.035, 0.054] | win |
| DNSMOS | 2.961 | 2.882 | -0.079 [-0.101, -0.058] | loss |
| UTMOS | 2.445 | 2.207 | -0.239 [-0.273, -0.204] | loss |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 13.6 / 8.9*, 10k: 10.5 / 7.9*, 15k: 7.2 / 5.0*, 20k: 6.1 / 3.9

**Training.** final validation flow 0.6811 (step 20000); 0.155 s/update; 13,518 target frames/update

**Verdict: adopted.** The largest single win. WER 12.3 -> 5.6, alignment ~4x earlier (quick WER 13.6 at 5k vs ~98), SIM-o +0.045; DNSMOS -0.08 and UTMOS -0.24 (recovered in full-v2 by the quality condition).

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-repa) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-repa/logs)

## `x-tla` (#10): rejected


**Question.** TLA-SA. Per-layer speaker alignment of token features to frozen ECAPA embeddings, weight 0.5 (trained before the audit fix that masks prompt-free rows).

**Change.** `model.tla_layers=all`, `model.tla_dim=192`, `model.tla_hidden=256`, `train.speaker_embeddings=teacher/ecapa-speechbrain`, `train.tla_weight=0.5`, `train.tla_entropy=0.01`

**Evidence.** TLA-SA (arXiv 2511.09995).

**Baseline.** `pairs-cross`

Final evaluation, sampling seeds 42 + 1000 pooled (x-tla/step-0020000, x-tla/step-0020000-s1000):

| metric | `pairs-cross` | `x-tla` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 12.30 | 24.67 | +12.38 [10.64, 14.12] | loss |
| CER % | 8.25 | 14.05 | +5.81 [4.77, 6.84] | loss |
| SIM-o | 0.540 | 0.455 | -0.085 [-0.094, -0.077] | loss |
| DNSMOS | 2.961 | 2.771 | -0.190 [-0.213, -0.168] | loss |
| UTMOS | 2.445 | 2.167 | -0.278 [-0.314, -0.242] | loss |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 112.3 / 83.0*, 10k: 43.5 / 25.2*, 15k: 29.6 / 17.7*, 20k: 24.2 / 13.4

**Training.** final validation flow 0.6860 (step 20000); 0.156 s/update; 13,518 target frames/update

**Verdict: rejected.** WER 24.7 vs 12.3, SIM-o 0.455 vs 0.540, slower alignment at every snapshot. Outside the seed spread.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-tla) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-tla/logs)

## `x-swiglu` (#9): tie


**Question.** SwiGLU feed-forward (equal parameters).

**Change.** `model.ffn_activation=swiglu`

**Evidence.** Positive in T5 / LightningDiT, neutral in the 140M SR-DiT.

**Baseline.** `pairs-cross`

Final evaluation, sampling seeds 42 + 1000 pooled (x-swiglu/step-0020000, x-swiglu/step-0020000-s1000):

| metric | `pairs-cross` | `x-swiglu` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 12.30 | 17.45 | +5.15 [3.73, 6.57] | loss |
| CER % | 8.25 | 9.91 | +1.66 [0.76, 2.57] | loss |
| SIM-o | 0.540 | 0.524 | -0.016 [-0.024, -0.009] | loss |
| DNSMOS | 2.961 | 2.903 | -0.058 [-0.074, -0.043] | loss |
| UTMOS | 2.445 | 2.366 | -0.079 [-0.102, -0.056] | loss |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 98.6 / 58.1*, 10k: 38.7 / 22.2*, 15k: 24.7 / 19.3*, 20k: 17.8 / 10.0

**Training.** final validation flow 0.6732 (step 20000); 0.153 s/update; 13,518 target frames/update

**Verdict: tie.** No gain. WER +5.2 against the same seed, but level with the seed-43 base (17.5 vs 16.9) on every metric; the audit found the code correct (Shazeer form, equal parameters). Not adopted; retested on the v2 recipe (y-swiglu).

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-swiglu) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-swiglu/logs)

## `x-attn-gate` (#9): tie


**Question.** Head-wise sigmoid gate on the attention output.

**Change.** `model.attn_gate=head`

**Evidence.** Gated attention (arXiv 2505.06708), LLMs only; no TTS ablation.

**Baseline.** `pairs-cross`

Final evaluation, sampling seeds 42 + 1000 pooled (x-attn-gate/step-0020000, x-attn-gate/step-0020000-s1000):

| metric | `pairs-cross` | `x-attn-gate` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 12.30 | 16.67 | +4.37 [2.78, 5.96] | loss |
| CER % | 8.25 | 10.25 | +2.01 [0.92, 3.10] | loss |
| SIM-o | 0.540 | 0.529 | -0.011 [-0.017, -0.005] | loss |
| DNSMOS | 2.961 | 2.917 | -0.044 [-0.064, -0.025] | loss |
| UTMOS | 2.445 | 2.375 | -0.070 [-0.096, -0.045] | loss |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 99.1 / 56.3*, 10k: 29.3 / 16.9*, 15k: 18.8 / 12.0*, 20k: 16.8 / 10.3

**Training.** final validation flow 0.6723 (step 20000); 0.279 s/update; 13,518 target frames/update

**Verdict: tie.** No gain. WER +4.4 against the same seed, but level with the seed-43 base (16.7 vs 16.9) on every metric, although it starts as exactly the base function (zero-init gate): a zero-init change alone moves the base+cross 20k WER by ~5 points. Not adopted; retested on the v2 recipe (y-attn-gate).

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-attn-gate) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-attn-gate/logs)

## `x-long-skip` (#9): tie


**Question.** Input -> output long skip (concat + LayerNorm + Linear).

**Change.** `model.long_skip=true`

**Evidence.** DiTTo-TTS: WER 3.30 -> 2.93; counter-evidence F5 4.17 -> 5.17.

**Baseline.** `pairs-cross`

Final evaluation, sampling seeds 42 + 1000 pooled (x-long-skip/step-0020000, x-long-skip/step-0020000-s1000):

| metric | `pairs-cross` | `x-long-skip` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 12.30 | 13.01 | +0.72 [-0.55, 1.98] | tie |
| CER % | 8.25 | 8.21 | -0.03 [-0.97, 0.90] | tie |
| SIM-o | 0.540 | 0.535 | -0.005 [-0.011, 0.001] | tie |
| DNSMOS | 2.961 | 2.953 | -0.009 [-0.025, 0.008] | tie |
| UTMOS | 2.445 | 2.407 | -0.039 [-0.060, -0.017] | loss |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 90.5 / 56.4*, 10k: 30.6 / 18.3*, 15k: 17.4 / 10.9*, 20k: 12.9 / 8.1

**Training.** final validation flow 0.6713 (step 20000); 0.156 s/update; 13,518 target frames/update

**Verdict: tie.** WER +0.7 [-0.6, 2.0], CER, SIM-o and DNSMOS ties; UTMOS -0.04. No gain on base+cross; retested on the v2 recipe (y-long-skip).

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-long-skip) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-long-skip/logs)

## `x-value-residual` (#9): not run


**Question.** Value residual (v_l <- l1 v_l + l2 v_1).

**Change.** `model.value_residual=true`

**Evidence.** ResFormer (arXiv 2410.17897); SR-DiT FID 4.02 -> 3.64.

**Baseline.** `pairs-cross`

No final evaluation yet.


**Verdict: not run.** Not run on base+cross: after the audit the option moved to the v2 recipe as y-value-residual.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-value-residual) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-value-residual/logs)

## `x-ffn-conv` (#9): not run


**Question.** Depthwise convolution (k=5) inside the feed-forward.

**Change.** `model.ffn_conv_kernel=5`

**Evidence.** ZipVoice (removing its conv modules takes WER 1.69 -> 9.79), U-DiT, SANA.

**Baseline.** `pairs-cross`

No final evaluation yet.


**Verdict: not run.** Not run on base+cross: after the audit the option moved to the v2 recipe as y-ffn-conv.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-ffn-conv) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-ffn-conv/logs)

## `x-final-adaln` (#9): not run


**Question.** Final adaLN before the output projection.

**Change.** `model.final_adaln=true`

**Evidence.** Standard in DiT and F5-TTS.

**Baseline.** `pairs-cross`

No final evaluation yet.


**Verdict: not run.** Not run on base+cross: after the audit the option moved to the v2 recipe as y-final-adaln.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-final-adaln) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-final-adaln/logs)

## `x-cond-text-pool` (#9): not run


**Question.** Pooled target text added to the condition.

**Change.** `model.cond_text_pool=true`

**Evidence.** DiTTo-TTS: pooled text WER 3.00 -> 2.93.

**Baseline.** `pairs-cross`

No final evaluation yet.


**Verdict: not run.** Not run on base+cross: after the audit the option moved to the v2 recipe as y-cond-text-pool.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-cond-text-pool) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-cond-text-pool/logs)

## `x-regularized` (#14): not run


**Question.** Dropout 0.1 and weight decay 0.05.

**Change.** `model.dropout=0.1`, `train.weight_decay=0.05`

**Baseline.** `pairs-cross`

No final evaluation yet.


**Verdict: not run.** Not run on base+cross: after the audit the option moved to the v2 recipe as y-regularized (with weight decay on matrices only).

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-regularized) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-regularized/logs)

## `x-speaker-context` (round 2): not run


**Question.** Multi-clip speaker context vector (50 % of updates).

**Change.** `model.speaker_context=vector`, `train.speaker_context_prob=0.5`

**Baseline.** `pairs-cross`

No final evaluation yet.


**Verdict: not run.** Not run: deferred after the audit (the single-prompt evaluation has no extra clips, so it cannot show a context gain).

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-speaker-context) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-speaker-context/logs)

## `x-repa-tla` (#10): not run


**Question.** Speech-REPA and TLA-SA together. Does TLA-SA hurt less once REPA supplies the alignment?

**Change.** `model.repa_layer=10`, `model.repa_dim=256`, `train.teacher_features=teacher/mhubert147-l12-pca256`, `train.repa_weight=1.0`, `train.repa_stop_step=0`, `train.repa_frames=all`, `model.tla_layers=all`, `model.tla_dim=192`, `model.tla_hidden=256`, `train.speaker_embeddings=teacher/ecapa-speechbrain`, `train.tla_weight=0.5`, `train.tla_entropy=0.01`

**Baseline.** `x-repa`

No final evaluation yet.


**Verdict: not run.** Not run: deferred after the audit (TLA-SA alone lost clearly; the v2 recipe already has REPA).

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-repa-tla) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-repa-tla/logs)

## `x-speaker-condition` (round 2): not run


**Question.** Frozen ECAPA speaker embedding -> adaLN.

**Change.** `model.speaker_condition_dim=192`, `train.speaker_condition=teacher/ecapa-speechbrain`

**Baseline.** `pairs-cross`

No final evaluation yet.


**Verdict: not run.** Not run on base+cross: after the audit the option moved to the v2 recipe as y-speaker-condition.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-speaker-condition) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/x-speaker-condition/logs)

## `full-cross` (model): adopted


**Question.** First full-length model. Run C's recipe + cross prompts, 60k updates.

**Baseline.** `run-c-reference`

Final evaluation, sampling seeds 42 + 1000 pooled (full-cross/step-0060000, full-cross/step-0060000-s1000):

| metric | `run-c-reference` | `full-cross` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 7.99 | 6.46 | -1.53 [-3.77, 0.70] | tie |
| CER % | 5.30 | 4.40 | -0.90 [-2.60, 0.79] | tie |
| SIM-o | 0.536 | 0.548 | +0.013 [0.000, 0.025] | win |
| DNSMOS | 2.898 | 2.970 | +0.072 [0.042, 0.103] | win |
| UTMOS | 2.420 | 2.519 | +0.099 [0.044, 0.155] | win |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 92.9 / 53.8*, 10k: 27.9 / 16.5*, 15k: 17.9 / 11.5*, 20k: 12.0 / 8.5*, 25k: 11.4 / 7.2*, 30k: 9.9 / 6.5*, 35k: 9.2 / 6.2*, 40k: 7.2 / 4.8*, 45k: 8.1 / 4.9*, 50k: 8.3 / 5.9*, 55k: 7.9 / 6.2*, 60k: 6.7 / 4.7

**Training.** final validation flow 0.6583 (step 60000); 0.152 s/update; 13,574 target frames/update

**Verdict: adopted.** Beats run C on every metric (refit duration predictor, seeds pooled; WER 5.10 -> 2.94).

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/full-cross) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/full-cross/logs)

## `full-v2` (model): adopted


**Question.** Second full-length model. Cross prompts + quality condition + speech-REPA + character units, 60k updates.

**Change.** `model.quality_condition=true`, `train.quality_scores=quality/dnsmos.json`, `model.quality_target=[4.0, 4.5, 3.8]`, `model.repa_layer=10`, `model.repa_dim=256`, `train.teacher_features=teacher/mhubert147-l12-pca256`, `train.repa_weight=1.0`, `train.repa_stop_step=0`, `train.repa_frames=all`, `model.text_units=chars`

**Baseline.** `full-cross`

Final evaluation, sampling seeds 42 + 1000 pooled (full-v2/step-0060000, full-v2/step-0060000-s1000):

| metric | `full-cross` | `full-v2` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 6.46 | 2.21 | -4.24 [-5.50, -2.99] | win |
| CER % | 4.40 | 1.17 | -3.23 [-4.18, -2.27] | win |
| SIM-o | 0.548 | 0.577 | +0.029 [0.018, 0.040] | win |
| DNSMOS | 2.970 | 3.164 | +0.194 [0.167, 0.221] | win |
| UTMOS | 2.519 | 2.515 | -0.004 [-0.047, 0.038] | tie |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 11.3 / 6.7*, 10k: 6.8 / 4.9*, 15k: 5.8 / 4.5*, 20k: 4.3 / 3.7*, 25k: 5.1 / 3.3*, 30k: 3.4 / 2.4*, 35k: 2.4 / 1.6*, 40k: 2.6 / 1.8*, 45k: 2.3 / 1.7*, 50k: 2.2 / 1.3*, 55k: 1.8 / 1.1*, 60k: 2.4 / 1.4

**Training.** final validation flow 0.6700 (step 60000); 0.153 s/update; 13,605 target frames/update

**Verdict: adopted.** The current best model. With the refit duration predictor, WER 0.93 / CER 0.36 vs run C 5.10 / 2.93 and full-cross 2.94 / 1.66; DNSMOS 3.13, SIM-o 0.556, UTMOS tie.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/full-v2) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/full-v2/logs)

## `ft-w0` (#14): tie


**Question.** Control fine-tune. 8k more updates of full-cross without model guidance, sampled with CFG 5.

**Baseline.** `full-cross`

Final evaluation, sampling seeds 42 + 1000 pooled (ft-w0/step-0008000, ft-w0/step-0008000-s1000):

| metric | `full-cross` | `ft-w0` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 6.46 | 6.58 | +0.13 [-0.58, 0.84] | tie |
| CER % | 4.40 | 4.42 | +0.02 [-0.51, 0.55] | tie |
| SIM-o | 0.548 | 0.546 | -0.002 [-0.006, 0.002] | tie |
| DNSMOS | 2.970 | 2.964 | -0.006 [-0.017, 0.005] | tie |
| UTMOS | 2.519 | 2.486 | -0.033 [-0.051, -0.016] | loss |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 2k: 6.5 / 4.7*, 4k: 8.8 / 6.1*, 6k: 6.0 / 4.6*, 8k: 6.6 / 4.7

**Training.** final validation flow 0.6577 (step 8000); 0.209 s/update; 13,876 target frames/update

**Verdict: tie.** WER 6.58 vs 6.46, a tie; extra updates alone change nothing.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/ft-w0) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/ft-w0/logs)

## `ft-mg-w07` (#14): rejected


**Question.** Model-guidance fine-tune (w = 0.7), sampled without CFG (half the network evaluations).

**Evidence.** Model guidance (arXiv 2504.20334).

**Baseline.** `full-cross`

Final evaluation, sampling seeds 42 + 1000 pooled (ft-mg-w07/step-0008000, ft-mg-w07/step-0008000-s1000):

| metric | `full-cross` | `ft-mg-w07` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 6.46 | 9.01 | +2.56 [1.64, 3.47] | loss |
| CER % | 4.40 | 5.82 | +1.42 [0.80, 2.04] | loss |
| SIM-o | 0.548 | 0.526 | -0.023 [-0.029, -0.016] | loss |
| DNSMOS | 2.970 | 3.022 | +0.052 [0.032, 0.071] | win |
| UTMOS | 2.519 | 2.546 | +0.026 [-0.004, 0.056] | tie |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 2k: 13.8 / 8.2*, 4k: 11.5 / 6.7*, 6k: 11.3 / 6.8*, 8k: 9.4 / 5.9

**Training.** final validation flow 0.7678 (step 8000); 0.248 s/update; 13,876 target frames/update

**Verdict: rejected.** WER 9.0 vs 6.5 and SIM-o -0.02 (clipping / 5, DNSMOS +0.05). Not worth half the NFE at this size.

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/ft-mg-w07) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/ft-mg-w07/logs)

## `grpo` (#16): tie


**Question.** Flow-GRPO, 600 updates, composite reward (CER 1.0 + SIM 0.5 + DNSMOS 0.4 + UTMOS 0.4) from full-cross 60k.

**Evidence.** FlowTTS-GRPO (arXiv 2606.23190).

**Baseline.** `full-cross`

Final evaluation, sampling seeds 42 + 1000 pooled (grpo/step-0000600, grpo/step-0000600-s1000):

| metric | `full-cross` | `grpo` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 6.46 | 6.70 | +0.24 [-0.21, 0.69] | tie |
| CER % | 4.40 | 4.51 | +0.11 [-0.22, 0.44] | tie |
| SIM-o | 0.548 | 0.551 | +0.003 [0.000, 0.005] | win |
| DNSMOS | 2.970 | 2.976 | +0.006 [-0.004, 0.015] | tie |
| UTMOS | 2.519 | 2.536 | +0.016 [0.002, 0.030] | win |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 600: 6.8 / 4.7

**Verdict: tie.** No measurable effect at this budget (WER 6.83 vs 6.46, SIM-o +0.005).

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/grpo) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/grpo/logs)

## `run-c-reference` (#3): reference


**Question.** Run C (VoiceHub/dacvae-tts-tr-w512, trained on Vyvo/tr-dataset-12) re-measured under the leak-free protocol.

**Baseline.** none (reference run)

Final evaluation, sampling seeds 42 + 1000 pooled (inference-runc/base, inference-runc/seed1000):

| WER % | CER % | SIM-o | DNSMOS | UTMOS |
|---:|---:|---:|---:|---:|
| 7.99 | 5.30 | 0.536 | 2.898 | 2.420 |


**Verdict: reference.** WER 7.54 / CER 4.89 / SIM-o 0.535 with the prompt-rate rule (the published 4.3 used podcast prompts whose voices occur in training).

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/run-c-reference) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/run-c-reference/logs)

## `y-base` (round 3): reference


**Question.** The full-v2 recipe (cross prompts + quality condition + speech-REPA + character units) at 20k updates: full-v2's own 20k snapshot, evaluated in full.

**Baseline.** `pairs-cross`

Final evaluation, sampling seeds 42 + 1000 pooled (y-base/step-0020000, y-base/step-0020000-s1000):

| metric | `pairs-cross` | `y-base` | difference [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER % | 12.30 | 4.32 | -7.98 [-9.32, -6.63] | win |
| CER % | 8.25 | 2.98 | -5.26 [-6.22, -4.31] | win |
| SIM-o | 0.540 | 0.587 | +0.046 [0.036, 0.057] | win |
| DNSMOS | 2.961 | 3.111 | +0.150 [0.128, 0.171] | win |
| UTMOS | 2.445 | 2.486 | +0.041 [0.007, 0.075] | win |


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 11.3 / 6.7*, 10k: 6.8 / 4.9*, 15k: 5.8 / 4.5*, 20k: 4.3 / 2.9

**Training.** final validation flow 0.6810 (step 20000); 0.153 s/update; 13,518 target frames/update

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-base) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-base/logs)

## `y-s43` (round 3): pending


**Question.** Noise floor of the v2 recipe: training seed 43. Is the 20k seed spread smaller once REPA aligns early?

**Change.** `train.seed=43`

**Baseline.** `y-base`

No final evaluation yet.


**Trajectory** (WER / CER %; * = quick check, first 96 sentences): 5k: 11.4 / 6.3*

**Training.** final validation flow 0.7015 (step 6000); 0.272 s/update; 13,483 target frames/update

Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-s43) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-s43/logs)

## `y-long-skip` (#9): pending


**Question.** Input -> output long skip (concat + LayerNorm + Linear), on the v2 recipe.

**Change.** `model.long_skip=true`

**Evidence.** DiTTo-TTS: WER 3.30 -> 2.93; counter-evidence F5 4.17 -> 5.17.

**Baseline.** `y-base`

No final evaluation yet.


Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-long-skip) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-long-skip/logs)

## `y-value-residual` (#9): pending


**Question.** Value residual (v_l <- l1 v_l + l2 v_1), on the v2 recipe.

**Change.** `model.value_residual=true`

**Evidence.** ResFormer (arXiv 2410.17897); SR-DiT FID 4.02 -> 3.64.

**Baseline.** `y-base`

No final evaluation yet.


Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-value-residual) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-value-residual/logs)

## `y-ffn-conv` (#9): pending


**Question.** Depthwise convolution (k=5) inside the feed-forward, on the v2 recipe.

**Change.** `model.ffn_conv_kernel=5`

**Evidence.** ZipVoice (removing its conv modules takes WER 1.69 -> 9.79), U-DiT, SANA.

**Baseline.** `y-base`

No final evaluation yet.


Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-ffn-conv) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-ffn-conv/logs)

## `y-final-adaln` (#9): pending


**Question.** Final adaLN before the output projection, on the v2 recipe.

**Change.** `model.final_adaln=true`

**Evidence.** Standard in DiT and F5-TTS.

**Baseline.** `y-base`

No final evaluation yet.


Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-final-adaln) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-final-adaln/logs)

## `y-cond-text-pool` (#9): pending


**Question.** Pooled transcript added to the condition, on the v2 recipe (in the joined layout the pool covers prompt and target text).

**Change.** `model.cond_text_pool=true`

**Evidence.** DiTTo-TTS: pooled text WER 3.00 -> 2.93.

**Baseline.** `y-base`

No final evaluation yet.


Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-cond-text-pool) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-cond-text-pool/logs)

## `y-swiglu` (#9): pending


**Question.** SwiGLU feed-forward (equal parameters), retested on the v2 recipe.

**Change.** `model.ffn_activation=swiglu`

**Evidence.** Positive in T5 / LightningDiT, neutral in the 140M SR-DiT.

**Baseline.** `y-base`

No final evaluation yet.


Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-swiglu) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-swiglu/logs)

## `y-attn-gate` (#9): pending


**Question.** Head-wise sigmoid gate on the attention output, retested on the v2 recipe.

**Change.** `model.attn_gate=head`

**Evidence.** Gated attention (arXiv 2505.06708), LLMs only.

**Baseline.** `y-base`

No final evaluation yet.


Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-attn-gate) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-attn-gate/logs)

## `y-speaker-condition` (round 2): pending


**Question.** Frozen ECAPA speaker embedding -> adaLN, on the v2 recipe (after the audit fix: inference embeds the same audio as the store).

**Change.** `model.speaker_condition_dim=192`, `train.speaker_condition=teacher/ecapa-speechbrain`

**Baseline.** `y-base`

No final evaluation yet.


Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-speaker-condition) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-speaker-condition/logs)

## `y-decay-matrices` (audit): pending


**Question.** Weight decay on matrices only (norm gains, biases, gates and embeddings undecayed), on the v2 recipe.

**Change.** `train.weight_decay_scope=matrices`

**Evidence.** Standard practice (AdamW exclusions in GPT/DiT training); the audit found decay on every parameter.

**Baseline.** `y-base`

No final evaluation yet.


Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-decay-matrices) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-decay-matrices/logs)

## `y-regularized` (#14): pending


**Question.** Dropout 0.1 and weight decay 0.05 on matrices, on the v2 recipe. Caveat: the text-hinge negative pass draws its own dropout masks.

**Change.** `model.dropout=0.1`, `train.weight_decay=0.05`, `train.weight_decay_scope=matrices`

**Baseline.** `y-base`

No final evaluation yet.


Files: [checkpoints, evaluations, audio](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-regularized) · [logs](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/y-regularized/logs)

