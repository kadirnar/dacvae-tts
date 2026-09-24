# DACVAE-TTS Turkish roadmap (24 September 2026)

Rationale and evidence: [research-report-2026-09-24.md](research-report-2026-09-24.md). Tracking issue:
[#17](https://github.com/kadirnar/dacvae-tts/issues/17). Each work item is a GitHub issue and
a branch of the same name; code changes are **enabled through configuration and off by default** (existing checkpoints and
configs run unchanged). Issues stay open until experimental evidence arrives: the code is ready on the branch, the A/B runs on a GPU.

**Priority principle:** measurement first (decisions made with the wrong metric cannot be undone), then speed gains
that do not change quality (they make every later experiment cheaper), then the A/Bs, and last data scale-up and post-training.

## Phase 0 — Measurement and cleanup

| # | Work item | Branch | Target metric |
|---|---|---|---|
| [#2](https://github.com/kadirnar/dacvae-tts/issues/2) | Documentation cleanup, research report, roadmap | `docs/temizlik-arastirma-raporu` | — |
| [#3](https://github.com/kadirnar/dacvae-tts/issues/3) | Evaluation protocol v2: SIM-o, UTMOS, deterministic Whisper, clipping/bandwidth | `eval/protokol-v2` | SIM, quality, WER reliability |
| [#4](https://github.com/kadirnar/dacvae-tts/issues/4) | Speaker-clustered bootstrap and paired comparison | `eval/istatistik` | Decision reliability |
| [#5](https://github.com/kadirnar/dacvae-tts/issues/5) | Turkish normalization v2 (consonant softening, abbreviations, metric bugs) | `text/turkish-v2` | WER/CER |
| [#6](https://github.com/kadirnar/dacvae-tts/issues/6) | Speaker leakage and program-level split | `data/konusmaci-sizinti` | SIM/WER reliability |

## Phase 1 — Speed and cheap wins

| # | Work item | Branch | Target metric |
|---|---|---|---|
| [#7](https://github.com/kadirnar/dacvae-tts/issues/7) | Training speed: synchronizations, per-block compile, padding, selective checkpointing | `perf/egitim-hizi` | Training time ×1.8–2.5 |
| [#8](https://github.com/kadirnar/dacvae-tts/issues/8) | Latent repeat/skip negatives (no extra forward pass) | `objective/latent-negatifler` | Step time −20 %, WER/CER |
| [#12](https://github.com/kadirnar/dacvae-tts/issues/12) | Duration: articulation rule, duration-diverse reranking | `duration/artikulasyon-kurali` | WER/CER (short sentences) |
| [#13](https://github.com/kadirnar/dacvae-tts/issues/13) | Sampler/output quality: per-window guidance, pre-tanh gain, composite selector | `inference/ornekleyici-kalite` | DNSMOS/UTMOS, clipping |

## Phase 2 — Architecture and objective A/Bs

| # | Work item | Branch | Target metric |
|---|---|---|---|
| [#9](https://github.com/kadirnar/dacvae-tts/issues/9) | DiT block options: long skip, value residual, conv-FFN, attention gate, SwiGLU, final adaLN | `arch/dit-blok-secenekleri` | WER/CER, quality |
| [#10](https://github.com/kadirnar/dacvae-tts/issues/10) | Speech-REPA + TLA-SA auxiliary losses | `objective/ogretmen-hizalama` | WER, SIM, convergence speed |
| [#11](https://github.com/kadirnar/dacvae-tts/issues/11) | Training pairs: different-sentence prompt, short target, trailing silence, character CTC | `data/egitim-ciftleri` | WER/CER, SIM, duration robustness |
| [#14](https://github.com/kadirnar/dacvae-tts/issues/14) | Schedule/regularization: WSD, uniform-t cooldown, dropout, dual EMA, model-guidance | `train/cizelge-duzenlilestirme` | Overfitting, quality, CFG-free inference |

## Phase 3 — Data and post-training

| # | Work item | Branch | Target metric |
|---|---|---|---|
| [#15](https://github.com/kadirnar/dacvae-tts/issues/15) | Turkish data expansion pipeline (YODAS tr000, Common Voice tr, ISSAI TSC) | `data/turkce-veri-hatti` | SIM, generalization |
| [#16](https://github.com/kadirnar/dacvae-tts/issues/16) | Flow-GRPO with a composite reward | `posttrain/grpo` | DNSMOS/UTMOS, WER, SIM |

## Merge order

The branches were opened separately from `main` and can be reviewed independently of each other. Several of them make **additions**
to shared files such as `config.py`, `model.py`, `training.py` and `data.py`; the suggested merge order keeps the small conflicts
to a minimum: #2 → #5 → #3 → #4 → #6 → #7 → #8 → #13 → #12 → #9 → #11 → #10 → #14 → #15 → #16. After each merge the remaining
branches are rebased onto `main`; the conflicts are mostly config fields added side by side.

## A/B protocol (all experiments)

- Baseline: `configs/nano_tr_w512.yaml`, clean cache, 20k updates (~3.2 hours on one 4090; ~1.5–2 hours after #7).
- **The baseline is run with two seeds** (noise floor); a single variable per arm.
- Evaluation: protocol v2 (#3) — Freya-TR-Eval (register/length buckets + an 8 kHz column), MiniMax-Turkish, internal podcast
  set; CER primary, WER, S/D/I, SIM-o + a second SV model, DNSMOS/UTMOSv2, clipping, step time.
- Decision: speaker-clustered paired bootstrap (#4); if the CI contains 0, "tie". Changes whose CER gain is larger than the seed
  spread are combined and validated in a full 60k run.

## Targets

| Metric | Now (C-60k) | Phase 1–2 target |
|---|---|---|
| Freya CER / WER (single sample, rule duration) | 2.5 % / 4.3 % | ≤ 1.8 % / ≤ 3.3 % |
| SIM-o (WavLM-large ECAPA) | to be measured (#3) | ≥ 90 % of the real-audio ceiling |
| DNSMOS OVRL / UTMOSv2 | 2.89 / to be measured | ≥ 3.05 / baseline + 0.2 |
| Time for 60k updates (single 4090) | 9.5 hours | ≤ 5 hours |
| Parameters (inference) | 66.5M | ≤ 68M |

## Second round (24 September 2026): review, training-free evidence, new options

Details: [review-and-architecture-2026-09-24.md](review-and-architecture-2026-09-24.md). Summary:

- Code review: ~37 verified bugs fixed. Among them are distill's EDM target, the DDP compile fallback, an inference crash on
  MPS, the post-training setup and the GRPO advantage baseline.
- #4: confidence intervals switched to speaker jackknife-t by default. The old method reported a win/loss 10–13 % of the time
  when the true difference was 0. In the re-analysis clamp comes out as a tie, while best-of-3 is a win.
- #5: the bugs in turkish-v2 were fixed and validated on 42,591 transcripts and 9,400 scored sentences.
- #6: 6/10 of the Freya prompt speakers are acoustically present in train. `--prompt-set` and
  `make_prompt_set.py` were added for leak-free measurement.
- Four of the "future work" items below were implemented (off by default, with A/B configs):
  - character units,
  - prompt tempo perturbation,
  - frozen speaker embedding → adaLN,
  - multi-clip speaker context.

## Future work (no issue opened)

4-step inference via CFG-fused MeanFlow/IntMeanFlow distillation; a frozen CAM++ embedding → adaLN and a separate reference encoder
for long (≥20 s) multi-clip references; tempo perturbation of the prompt only (requires re-encoding); full-band data with Sidon
restoration; a 16 × 448 depth/width A/B; text input with a character vocabulary; an SV loss with a latent-space speaker
verifier.
