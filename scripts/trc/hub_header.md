## Current result: new model vs run C (single sample, no reranking)

Same inference for both: duration predictor refit on tr-combined, guidance 5, 32 steps, sampling seeds 42 + 1000
pooled, paired speaker-clustered jackknife-t 95 % intervals.

| model | WER % | CER % | SIM-o | DNSMOS | UTMOS |
|---|---:|---:|---:|---:|---:|
| run C (`VoiceHub/dacvae-tts-tr-w512`, old data, 70 h) | 5.10 | 2.93 | 0.519 | 2.860 | 2.493 |
| [`full-cross`](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/full-cross) (tr-combined + cross-utterance prompts, 60k) | **2.94** | **1.66** | **0.536** | **2.936** | **2.572** |
| difference [95 % CI] | -2.16 [-3.27, -1.05] | -1.26 [-2.18, -0.34] | +0.017 [0.004, 0.029] | +0.076 [0.042, 0.110] | +0.078 [0.023, 0.133] |

[`full-v2`](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/full-v2) (cross prompts + quality condition + speech-REPA + character units, 60k) is being evaluated; its
quick-set WER at 50k is 2.2 against 8.3 for `full-cross` at 50k.

What was adopted, from the 20k A/B arms below: cross-utterance prompts (#11), speech-REPA (#10: WER halved, alignment
~4x earlier), the quality condition (DNSMOS control), character units (small, consistent over two training seeds).
Rejected: latent negatives (#8, collapse), TLA-SA (#10), model-guidance fine-tune (#14), Flow-GRPO at 600 updates
(#16, no effect). Side-by-side audio: [`comparison`](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/comparison). Inference comparisons: [`systems`](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/systems).
