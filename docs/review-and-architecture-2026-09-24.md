# Code review, training-free evidence and new architecture/data-pipeline options (24 September 2026, second round)

This document describes the `iyilestirme/inceleme-mimari-veri-hatti` branch. The branch adds 42 commits on top of
`main` (c1aeaf7). The content falls under four headings:

1. A code review of the whole repository and fixes for the confirmed bugs.
2. Evidence obtained without a GPU or training: statistics, normalization, speaker leakage, re-analysis of the
   published results.
3. Four new architecture/data-pipeline options from the "future work" list of the roadmap.
4. A/B runs to be done on the GPU.

Rules:

- No model training, synthesis/scoring run or dataset preparation was done locally.
- Every change was verified with unit tests that involve no training (last run: 520 tests passed, `ruff` clean).
- Every new option is off by default. When it is off:
  - the data flow stays bit-identical (recorded digest tests),
  - the model stays the same (fingerprint tests),
  - old checkpoints load unchanged.

Previous documents: [research-report-2026-09-24.md](research-report-2026-09-24.md) and
[roadmap.md](roadmap.md).

---

## 1. Code review: bugs found and fixed

Five parallel reviewers looked at these areas: model/objective, training/data, inference/duration, text/evaluation,
GRPO/data pipeline. Every finding was confirmed with the code or a small script. Every fix has its own commit and a
regression test.

| Area | Bug | Impact |
|---|---|---|
| Inference | `quality.py` computed float64 on the device | **Every synthesis** crashed on Apple Silicon (MPS) |
| Training | The `compile` fallback switched off gradient synchronization under DDP | In a multi-GPU `compile: blocks` run the ranks trained separate models |
| Training | EMA warm-up made the additional EMA tracks identical to each other; `--init-from` wiped a warm EMA in ~10 steps | The second EMA was meaningless in model-guidance fine-tuning; `train.ema_warmup: false` added |
| Training | The text-hinge RNG was not written to the checkpoint | Resume was not bit-exact reproducible |
| Training | The split of the WSD cooldown cache was not checked | Val/test audio could enter training during the cooldown |
| Training | The TLA-SA logit heads were in Muon | Zero-init heads received full-size updates from the first step |
| Post-training | `distill` regressed the raw output onto the velocity in EDM models | **Distill taught the wrong function in every config** |
| Post-training | The dataset did not take `layout` | Joined models (all Turkish configs) were post-trained with a SEP/segment layout they had not seen in training |
| GRPO | The standardized advantage amplified below-floor noise to unit variance | Judge noise received a full-size policy gradient |
| Data | `make_drop_list.py` kept rows with missing scores | A row with CER 0.6 could stay in training because of a DNSMOS failure |
| Data | License/source information was lost in `prepare` | A per-source license summary was added to the metadata |
| Sampler | `speaker_guidance` silently switched off at text scale 1 | Speaker guidance did not work at the lowest value of the demo slider |
| Sampler | APG/rescale options were ignored together with `speaker_guidance` | Now raises an error |
| Inference | CFG was re-applied on model-guidance checkpoints (`recommended_guidance` was not read) | A guidance of 5.0 effectively became ~16.7 |
| Inference | `synthesize_many` ignored the head of duration-head models; ASR crashed on a tuple reference; ASR ran in English on a Turkish checkpoint | All three fixed |
| Text | turkish-v2: `Prof.Dr.` stayed glued together; `2'incisi` ("the 2nd one") became `ikiincisi` (doubled "i"); clock times/dates/minus were missing; `5'de` ("at 5") became `beşde` (should be `beşte`); `9.-10.` ("9th–10th") became `dokuz.eksi onuncu` ("nine.minus tenth") | All fixed in v2 (v1 unchanged) |
| Metric | turkish-v2 merged hyphenated words (`yazlık-kışlık`, "summer-and-winter") | It charged 2 errors each to 16 correct transcripts (§2.2) |
| Evaluation | The scorer identity was not written to the rows; there was no `--metric-normalization`; determinism was incomplete; prompt WAVs relied on the file name; there was a seed-tts punctuation difference; there was no Freya metric | All fixed; `--freya-metric` added |

The full list and the rationale for each item are in the commit messages: `git log c1aeaf7..HEAD`.

---

## 2. Training-free evidence

### 2.1 Statistics (#4): the old confidence intervals were too narrow

`scripts/simulate_interval_coverage.py` runs a simulation against a known ground truth. The model was calibrated to
the published Freya results: the real word counts of the 495 sentences, 10 speakers, speaker/sentence/generation noise
effects. Every scenario was repeated 500 times.

| Criterion (target 95 % or 5 %) | Sentence bootstrap | Cluster percentile (old default) | **Jackknife-t (new default)** |
|---|---|---|---|
| Single-system WER coverage | 64–86 % | 87–89 % | **91–93 %** |
| Paired-difference coverage | 85–94 % | 87–90 % | **93–96 %** |
| False win/loss when the true difference is 0 | 6–7 % (16 % with interaction) | **10–13 %** | **4–6 %** |

Decision: `compare_evaluations`, `compare` and `scripts/compare_evals.py` now use a per-speaker delete-one jackknife
and t(G−1) by default (MacKinnon, Nielsen & Webb, arXiv:2301.04527). `--interval percentile` reproduces the old
intervals bit-exactly.

The published GPU results (run C, Freya s42 and s1000) were re-analysed with jackknife-t:

| Setting | WER difference [95 % CI] | Verdict |
|---|---|---|
| best-of-3 | −2.25 [−3.18, −1.32] | win |
| auto + best-of-3 (both sets pooled, 990 sentences) | −2.61 [−3.85, −1.36] | win |
| clamp (both sets pooled) | −0.69 [−1.46, +0.08] | **tie** (the old method called it a "win" on s1000) |
| predictor | −0.29 [−1.43, +0.84] | tie; DNSMOS +0.07 win |
| APG η 0.5 | +0.64 [+0.19, +1.09] | loss |
| CFG-rescale 0.7 | +0.84 [+0.26, +1.42] | loss |
| speaker guidance 7 | +0.82 [+0.27, +1.36] | loss |
| duration ×1.3 | +5.22 [+1.40, +9.03] | loss |

Caveat: in the best-of-N gain the selector (Whisper-turbo) and the judge (large-v3) come from the same family. Until it
is confirmed with an independent judge (e.g. MMS-1b), this number should be read as an optimistic upper bound.

### 2.2 Turkish normalization (#5)

**Training text.** turkish-v1 and v2 were applied to the 42,591 transcripts of `Vyvo/tr-dataset-12`. Of the 3,189 rows
that contain digits, 138 change; the effect on training is small. Every change is a correction:

- `yetmiş dörtü` → `dördü` ("seventy-four" + accusative: the stem must soften)
- `altmışdan` → `altmıştan` ("from sixty": consonant assimilation)
- `yetmişde` → `yetmişte` ("at seventy")
- `sekiz nokta otuzda` → `sekiz otuzda` ("at eight point thirty" → "at eight thirty")
- dates
- `vs.` → `vesaire` ("etc.")
- `A.Ş.` → `anonim şirketi` ("joint-stock company")

The single exception, the `9.-10.` bug, was fixed as well.

**Metric.** The Whisper hypotheses of the 19 published runs (~9,400 sentences) were rescored with v1 and v2:

- v2 **counts fewer false errors in 24 sentences, and there is no sentence where it is worse than v1.**
- Corpus WER 4.464 % → 4.431 %, CER 2.607 % → 2.588 %.

### 2.3 Speaker leakage (#6): the "unseen" voices are largely seen

**Metadata analysis.** Of the 22 held-out labels, 21 have their show and 13 their episode in train.

**Acoustic analysis.** Speaker centroids were computed with the SIM-o model (WavLM-large ECAPA). **6 of Freya's 10
prompt speakers have a counterpart from the same show in train with a cosine of 0.90–0.97.** Reference values:

- two halves of the same label: median 0.936, 5th percentile 0.876,
- labels from other shows: at most 0.60.

One speaker is borderline (0.81), only 3 are clean. So the published "unseen speaker" SIM/WER numbers were partly
measured on seen voices.

**Tooling.**

- `scripts/data/make_prompt_set.py` builds a leak-free prompt set from Common Voice test speakers. These speakers do not
  appear in the podcast data at all.
- `eval_sentences.py --prompt-set` uses this set. The original recording for SIM-o is the prompt file itself.
- Using, for example, 48 independent voices instead of 10 also narrows the confidence intervals.

### 2.4 Other measurements

**SIM scale (#3).** On run C's 16 Freya outputs the old metric `wavlm-base-plus-sv` saturates between 0.87 and 0.98,
while SIM-o discriminates between 0.37 and 0.83.

**Local reproduction.** Freya's 24 prompts were rebuilt one-to-one from the raw data (correlation 1.0000). The results
of the local baseline arm agree with the published ones:

| Metric | Local | Published |
|---|---|---|
| Old SIM | 0.9466 | 0.9461 |
| DNSMOS | 2.896 | 2.889 |
| Clipped files | 96 % | 96 % |

**Duration rules (#12).** Error in predicting the true duration (log-MAE) on 1,722 real sentence pairs from 22 held-out
speakers:

| Rule | log-MAE | Note |
|---|---|---|
| predictor | 0.126 | |
| auto | 0.132 | |
| clamp | 0.145 | |
| byte rule | 0.154 | |
| syllable | 0.159 | |
| **articulation** | 0.170 | systematically 7 % short |

The articulation rule is the worst predictor of the true duration. The WER measurement with synthesis was left to the
GPU.

**Pre-tanh gain (#13).** From the same latents, paired comparison:

| Metric | Difference |
|---|---|
| Clipped-file ratio | 96 % → 41 % |
| DNSMOS | +0.055 [+0.034, +0.077] |
| WER/CER | tie |
| UTMOS | −0.16 |

Because the gain also lowers the level (−13.7 → −15.6 LUFS), a level-normalized comparison was needed. That step could
not be done because the local runs were stopped at the user's request. The level-normalized repeat should be done on the
GPU/demo side.

---

## 3. New architecture and data-pipeline options

All four are switched on through the config and are off by default. Each has a single-variable A/B config. The designs
rest on a separate literature and code survey for each option.

| Option | Config | Parameters | Basis |
|---|---|---|---|
| **Character units** `model.text_units: chars` | `experiments/tr_w512_char_units.yaml` | 0 | Turkish two-byte letters (12 %) take two shares of the LARoPE diagonal, two CTC labels and extra duration in the byte rule. FreyaTTS uses a 92-symbol character vocabulary. There is no ablation comparing bytes with characters on Latin script. |
| **Prompt tempo perturbation** `train.tempo_prompt_prob` | `experiments/tr_w512_tempo_prompts.yaml` | 0 | VoiceStar: WER 6.42 → 5.66 (±25 %, WSOLA). The model sees the rate break between prompt and target during training. |
| **Frozen speaker embedding** `model.speaker_condition_dim` | `experiments/tr_w512_speaker_condition.yaml` | +98k | Koel-TTS: SV vector 0.619, in-context 0.637. MiniMax: encoder + prompt 0.746, prompt only 0.726; a frozen SV raised WER. WER should be treated as a gate. |
| **Multi-clip speaker context** `model.speaker_context: vector` | `experiments/tr_w512_speaker_context.yaml` | +2.24M (3.4 %) | Irodori v4: single clip 0.661 → 30 s 0.752 → 120 s 0.775. Evidence only from large models. Because it is zero-init, it can be warm-started from run C. |

**Character units.**

- The IDs are ISO-8859-9 (Latin-5) + 4. ASCII IDs stay equal to the bytes; the vocabulary (260), the embedding shape and
  the space/punctuation tokens do not change.
- Each of the 82 symbols that turkish-v1/v2 let through is a single, distinct unit.
- No ID falls into the UTF-8 continuation range (0x80–0xBF). As a result, the model raises an error if a character
  model is given a byte row.
- The cache does not change; the conversion happens at load time.
- At inference the prompt-rate rule keeps frames per character.

**Tempo.**

- `dacvae_tts.tempo`: WSOLA (40 ms window, 15 ms search, float64, pitch preserved) and a side-store
  writer/reader/merger.
- `scripts/build_tempo_variants.py`: produces row × tempo (×0.8/0.9/1.0/1.111/1.25) latents. ×1.0 is the re-encoded
  control.
- Dataset: the cut stays the same; the prompt is `variant[:round(cut/t)]`, the target does not change.
- The static and per-epoch cost limits preserve the frame budget.
- Stretched prompt frames are excluded from REPA.

**Speaker embedding.**

- A zero-init, bias-free Linear adds the L2-normalized embedding to `voice`. The CFG null branch, dropout and the
  prompt-free branch drop the embedding automatically.
- For within-sentence items the embedding is taken from **another** sentence of the same label, so the target does not
  leak.
- The checkpoint stores the embedder; at inference the same embedder is applied to the codec reconstruction of the
  prompt.
- It cannot share its embedding store with TLA-SA.

**Speaker context.**

- A 3 × 256 position-free set transformer, attention pooling and a zero-init output. The order of the clips does not
  change the result.
- Half of the items get 3–30 s of context from other sentences of the same label.
- At inference the context consists of the prompt and the `--context-audio` clips.

Other work done without training:

- `eval_sentences.py --prompt-set` and `scripts/data/make_prompt_set.py` (§2.3),
- jackknife-t intervals and the simulation script (§2.1),
- the turkish-v2 fixes (§2.2).

---

## 4. To do on the GPU (in order of priority)

1. **Fix the measurement.** Re-measure Freya with a leak-free prompt set:
   - 48 speakers from the CV-tr test split with `make_prompt_set.py`,
   - `eval_sentences.py --prompt-set ... --protocol-v2 --sim-o --metric-normalization turkish-v2`,
   - decisions with jackknife-t (`compare_evals.py`).
2. **Confirm best-of-N independently.** Pick the selector from a different family than the judge (e.g. MMS-1b CER) or
   add a second judge. The demo default (auto + best-of-3) depends on this.
3. **Inference options.** Re-measure the pre-tanh gain level-normalized. Measure the articulation duration rule and
   duration-diverse best-of-N (#12, #13).
4. **New architecture A/Bs.** At 20k updates, against a two-seed baseline run:
   - character units,
   - tempo prompts (generate the store first),
   - speaker embedding (first `speakers --splits train,val`),
   - speaker context (from scratch or `--init-from` run C).
5. **Older issues.** The A/Bs of #7–#11 and #14–#16, with the protocol in [roadmap.md](roadmap.md).
