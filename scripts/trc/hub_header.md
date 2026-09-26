## Current result: new models vs run C (single sample, no reranking)

Same inference for all three: duration predictor refit on tr-combined, guidance 5, 32 steps, 495 Freya-TR-Eval
sentences × 48 leak-free Common Voice voices, sampling seeds 42 + 1000 pooled. Brackets: paired speaker-clustered
jackknife-t 95 % interval of the difference to run C.

| model | WER % | CER % | SIM-o | DNSMOS | UTMOS |
|---|---:|---:|---:|---:|---:|
| run C (`VoiceHub/dacvae-tts-tr-w512`, old data, 70 h) | 5.10 | 2.93 | 0.519 | 2.860 | 2.493 |
| [`full-cross`](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/full-cross): tr-combined + cross-utterance prompts, 60k | 2.94 [-3.27, -1.05] | 1.66 [-2.18, -0.34] | 0.536 | 2.936 | 2.572 |
| [**`full-v2`**](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/full-v2): + quality condition + speech-REPA + character units, 60k | **0.93** [-5.35, -2.99] | **0.36** [-3.52, -1.62] | **0.556** | **3.128** | 2.533 |

`full-v2` against `full-cross`: WER -2.01 [-2.55, -1.46], CER -1.31 [-1.66, -0.95], SIM-o +0.021 [0.010, 0.031],
DNSMOS +0.192 [0.163, 0.221], UTMOS -0.039 [-0.084, 0.006] (tie). Seed 42: 468 of 495 sentences have no word error; the rest
are near-homophones ("Vurmak da" → "Vurmakta"), foreign names and one repeated phrase.

What was adopted, from the 20k A/B arms below: cross-utterance prompts (#11), speech-REPA (#10: WER halved, alignment
~4x earlier), the quality condition (DNSMOS control), character units (small, consistent over two training seeds).
Rejected: latent negatives (#8, collapse), TLA-SA (#10), model-guidance fine-tune (#14), Flow-GRPO at 600 updates
(#16, no effect). Side-by-side audio: [`comparison`](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/comparison).
Inference comparisons: [`systems`](https://huggingface.co/VoiceHub/dacvae-tts-tr-combined/tree/main/systems).
