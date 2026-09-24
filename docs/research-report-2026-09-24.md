# DACVAE-TTS Turkish: block-by-block architecture, training and quality research (24 September 2026)

**Scope.** This report synthesizes a read-through of all the code in the repository (model, training, data, codec,
inference, optimizer, duration, metrics, demo) and six parallel literature surveys (DiT backbone; text/alignment/duration;
training objective and speed; speaker similarity; audio quality on DACVAE; Turkish evaluation and data). Goal: to jointly
improve **audio quality, WER, CER and SIM** in a **small-parameter, fast-to-train** Turkish zero-shot TTS. Meta's
**DACVAE** stays fixed as the codec; every proposal was evaluated for this latent space.

Log of earlier results: [turkish-experiments-2026-09-22.md](turkish-experiments-2026-09-22.md).
Implementation plan and issues: [roadmap.md](roadmap.md).

**Evidence tags:** **[S]** strong (multiple TTS papers/ablations or a local measurement), **[M]** medium (a single TTS
paper or consistent image-DiT evidence), **[W]** weak (LLM/image only, a mechanism-based guess, or conflicting evidence).
Unless stated otherwise, numbers are the respective paper's own reported figures; those marked "local" were measured in
this repository.

---

## 0. Executive summary

1. **Measurement must be fixed first.** The current SIM metric (`microsoft/wavlm-base-plus-sv`) gives same-speaker pairs
   0.89–0.96 and even different speakers 0.60–0.84; our 0.94–0.95 values sit at the metric's ceiling and do not separate
   systems. The literature standard **SIM-o** (WavLM-large + ECAPA-TDNN, seed-tts-eval) gives ~0.6–0.7 for the same
   speaker and ~0 for different ones. Also: the "WER 4.3 vs FreyaTTS 8.0" comparison on Freya-TR-Eval is not
   like-for-like (Freya scores after downsampling to 8 kHz); on the 495-sentence set, differences smaller than ~±0.7 WER
   points cannot be distinguished statistically; the "24 unseen prompts" come from only 10 speakers, and because of
   within-episode diarization labels the same person may be in train under a different label; best-of-3 results are
   inflated because the selector (Whisper-turbo) and the judge (large-v3) come from the same family.
2. **The architecture is largely right.** Separate self-attention + cross-attention with **LARoPE**, shared low-rank
   adaLN, QK-norm, patch 1, EDM preconditioning, logit-normal *t*, batch expansion Ke=4, the CTC auxiliary loss and Muon
   are supported by the evidence. Joint/MM-DiT attention, patching/downsampling, layer sharing, phoneme/BPE input and a
   large pretrained text encoder are **not recommended**.
3. **The biggest levers** (ordered by cost):
   - **Training speed ×1.8–2.5** (with quality unchanged): removing ~10–15 host–device synchronizations per step,
     per-block `torch.compile` + length padding, reducing activation checkpointing, removing the extra forward pass for
     text negatives (ΔFM-style latent negatives). 60k updates 9.5 hours → ~4–5.5 hours. [M]
   - **Duration and sentence endings:** a silence-aware, syllable-based duration rule; duration-diverse reranking;
     tail-silence padding in training. Targets the "filler word" insertions at the end of short sentences. [M]
   - **SIM:** correct metric + guidance sweep (inference), the **TLA-SA** speaker alignment loss (+0.03–0.06 SIM-o, <2 %
     extra compute), a mix of same-/different-sentence prompts, more Turkish speakers. [M–S]
   - **WER/CER:** the A-DMA speech term / **speech-REPA** (alignment to precomputed multilingual SSL features; WER
     −10–15 % relative, ~2× faster convergence), Turkish text normalization bugs (e.g. "4'e" ("to 4") → "dörte"),
     converting CTC targets to lowercase characters. [M]
   - **Audio quality:** most of the DNSMOS gap (2.89 vs the codec ceiling of 3.27) comes not from clipping but from
     **generation errors**; the strongest training-side lever is **GRPO** with a composite reward (DNSMOS +0.25 on F5,
     with WER going down). Training-free: applying CFG only in the noisy half, gain control before the tanh, a composite
     best-of-N selector. On the data side, Sidon restoration (DNSMOS 3.07 → 3.45 on Turkish FLEURS, CER unchanged). [M]
4. **Small-model principle:** keeping 66.5M and spending the gained speed on more updates is better than reducing
   parameters (448 → 512 width gave the largest gain). The total parameter cost of the new proposals is <2M. If
   parameters must be cut, the first candidate is the text encoder (22 % of the parameters).

---

## 1. Current state

### 1.1 Architecture (configs/nano_tr_w512.yaml)

| Component | Choice |
|---|---|
| Codec | Frozen `facebook/dacvae-watermarked`: 48 kHz, hop 1920 → 25 frames/s, 128-channel continuous VAE posterior mean; −16 LUFS; per-channel standardization |
| Text | `turkish-v1` normalization (numbers to their spoken form, İ/ı), UTF-8 bytes (vocabulary 260), prompt and target transcript in a single stream (joined) |
| Text encoder | byte embedding + sinusoidal position, 4 × (depthwise conv k7 + MLP), 4 × pre-LN transformer (RoPE, GELU) |
| Generator | 12 DiT blocks, width 512, 8 heads, patch 1. Block: shared adaLN + rank-64 per-block correction (zero-init gates) → self-attention (RoPE, QK-RMSNorm) → cross-attention (LARoPE, γ=10) → GELU FFN (×3). LayerNorm without affine parameters |
| Conditioning | time sinusoid → MLP + the mean of an MLP over the prompt latents (global "voice" vector) |
| Prompt | first 10–60 % of the same utterance (within-utterance, F5/E2 infilling), 30 % prompt dropout, 20 % joint CFG dropout |
| Objective | EDM-style unit-variance velocity target, stratified logit-normal *t*, frame-weighted MSE, Ke=4 batch expansion, CTC at block 8 (λ 0.1), skip/repeat text negatives (hinge, λ 0.2) |
| Optimizer | Muon (hidden matrices, per-part orthogonalization) + AdamW; LR 8e-4, 2k warmup, cosine → 10 %, wd 0.01, EMA 0.9999, bf16, grad-clip 1 |
| Duration | the prompt's frames-per-byte ratio (F5 rule); at inference `clamp`/`predictor`/`auto` |
| Sampling | Euler 32 steps, sway −1, CFG 5 (both branches in one batch) |

### 1.2 Parameter breakdown (local measurement, `FlowTTS(ModelConfig)`)

| Module | 66.5M (w512) | Share |
|---|---:|---:|
| Generator FFNs | 18.9M | 28.4 % |
| Generator self-attention | 12.6M | 19.0 % |
| Generator cross-attention | 12.6M | 19.0 % |
| Text encoder transformer blocks | 10.5M | 15.8 % |
| Text encoder MLPs | 4.2M | 6.3 % |
| adaLN (shared + low-rank) | 6.4M | 9.6 % |
| Other (time, reference, CTC, input/output, embedding) | 1.3M | 1.9 % |

### 1.3 Results (summary)

Best model C (66.5M, 60k updates, 9.5 hours on a single 4090, clean data): Freya-TR-Eval WER 4.3 % / CER 2.5 %,
SIM (base-plus-sv) 0.946, DNSMOS OVRL 2.89 (post-codec ceiling of real speech: 3.27). `auto` duration mode 3.61/1.81;
`auto` + best-of-3 1.59/0.72 (see the selector-bias caveat in §2.2). Remaining error types: filler words at the end of short
sentences, single-word repetition/dropping, letter errors in rare/foreign proper names, WER varying between 5–23 %
depending on the prompt, the decoder hitting the tanh ceiling at high CFG.

---

## 2. Measurement reliability (before anything else)

### 2.1 The SIM metric is on the wrong scale [S]

- According to its model card, the `wavlm-base-plus-sv` we use has a same/different-speaker threshold of ~0.86; it gives
  0.89–0.96 for the same speaker and 0.60–0.84 for different speakers. On the same VoxCeleb clips, seed-tts-eval's model
  gives 0.60–0.69 for the same speaker and −0.17–0.18 for different ones.
- **Standard (SIM-o):** UniSpeech `ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large')`, s3prl WavLM-Large
  backbone + ECAPA head, `wavlm_large_finetune.pth`. Audio: channel 0, resampled to 16 kHz, no trimming/normalization,
  full clip; cosine between the generated audio (excluding the prompt frames) ↔ the **original** prompt wav. The
  seed-tts-eval and F5-TTS code do exactly this. The original download link has expired; HF mirror:
  `bezzam/wavlm_large_finetune_seed_tts_eval` (sha256 `51f07e3b…f7f94b`). License CC BY-SA 3.0 (fine for evaluation).
- `monitor.py` and `eval_sentences.py` measure against the codec-passed prompt: this is **SIM-r**, not SIM-o.
- Typical SIM-o values: Seed-TTS real audio 0.73–0.76; F5-TTS 336M 0.66–0.76; ZipVoice 123M 0.67–0.75; SupertonicTTS
  60M 0.60; Turkish MiniMax set: MiniMax-Speech 0.779, ElevenLabs 0.596, OmniVoice 0.851, VoxCPM2 0.871.
- Conclusion: earlier SIM-based decisions such as "independent speaker guidance did not help" must be **re-measured**. A
  second, independent metric that cannot be gamed (CAM++ or SpeechBrain ECAPA) should also be reported — especially if
  WavLM-SV is later used for training/reranking.

### 2.2 WER comparability and statistics [S]

- **FreyaTTS protocol:** all system outputs are downsampled to 8 kHz and upsampled back to 16 kHz, faster-whisper
  large-v3 (beam 5), corpus-level WER/CER with jiwer (CER counts spaces), apostrophes are converted to spaces; the
  cloning baselines are given a single fixed reference; FreyaTTS itself is single-voice. Real FLEURS-tr speech gets
  9.7 % WER under this protocol. So our full-band 4.3 % with 24 podcast prompts is not like-for-like with Freya's 8.0;
  we should also report our own number **with Freya's 8 kHz script**.
- **Confidence intervals (local, from published results):** for corpus WER 4.32, the 95 % CI is [3.59, 5.09] with a
  sentence bootstrap and [3.07, 5.62] speaker-clustered; CER 2.50 → [2.03, 2.99] / [1.68, 3.38]. Per-speaker WER
  1.2–9.8. Paired comparisons: duration predictor ΔWER −0.18 [−1.26, +0.64] (tie), `guidance_until 0.7` +0.10 (tie),
  `clamp` −0.72 [−1.49, 0.00] (borderline). **Differences smaller than ~0.7 points cannot be resolved on this set.** A
  5-sentence set is ±6.8 WER; an 8-clip codec ceiling is uninterpretable.
- **Selector bias:** in best-of-N, having Whisper-turbo select the candidates and then scoring with large-v3 carries a
  same-family bias (arXiv 2607.08256: a selector/judge from the same family recovers 2–3× the "oracle" gap). The selector
  should come from a different family (e.g. Omnilingual ASR) or be verified with a second judge.
- **Determinism:** `Evaluator.score` leaves faster-whisper's temperature fallback on (0 → 1.0); on bad clips decoding can
  drift into sampling → `temperature=0`, `without_timestamps=True`.
- **ASR bias:** the training data was filtered by large-v3 CER and is scored with large-v3; the model may learn
  "Whisper-friendly" speech. A second judge from a different family (omniASR_CTC_3B_v2; Turkish FLEURS WER 9.2 / CER
  2.0) is recommended.
- **CER should be primary:** in Turkish, large-v3 WER is ~3.5× the CER (FLEURS-tr 5.66 / 1.62); a single wrong suffix
  counts the whole long word as wrong.

### 2.3 Speaker leakage [M]

`speaker_split` splits by a hash of the speaker label; the labels are within-episode diarization of the form
`<episode>_speaker_k`. Recurring podcast hosts may be in train under another episode's label; this inflates both the
"unseen speaker" WER and SIM. Fix: splitting at the episode/show level, and checking how close the test speakers are to
the train clusters (e.g. cosine < 0.6) by clustering WavLM-ECAPA centroids across episodes.

### 2.4 Metric text normalization bugs [M]

`metric_text_turkish`: (1) the Mn-removal line is dead code — the alnum filter first turns U+0307 into a space
("i̇stanbul" → "i stanbul"; this breaks with a second ASR that uses Python `lower()`); (2) â/î/û are not folded ("kâr"
≠ "kar", "profit" vs "snow"); (3) Roman numerals ("II." → "ıı"); (4) ordinals without a space ("21.yüzyıl", "21st
century"); (5) clock-time readings ("3.30'a" → "üç nokta otuza", "three point thirty" instead of a time); (6) spelling
variants ("avro"/"euro"); (7) units/abbreviations; (8) hyphenated words ("e-posta" ("e-mail") as two words); (9)
abbreviation letter fixes ("COVID" → "covıd"). It only moves Freya from 4.32 → ~4.37, but it matters for the podcast
monitor and the CER filter in training. Reference implementation: **trnorm** (Apache-2.0).

The training normalization has a bug as well: **missing consonant softening** — "4'e" → "dörte" (correct: "dörde", "to
four"), "4'ü" → "dörtü", "2024'e" → "iki bin yirmi dörte". This wrong text leaked into training. In addition, "T.C.",
"A.Ş.", "Ltd. Şti." (abbreviations for the Turkish Republic, joint-stock company and limited company) are not expanded.
The fix should be made as a new normalization version (`turkish-v2`) so that existing caches and checkpoints do not
break.

### 2.5 Proposed evaluation protocol v2

1. Primary judge faster-whisper large-v3 (fp16, `language="tr"`, beam 5, `temperature=0`, `without_timestamps=True`);
   secondary judge omniASR_CTC_3B_v2. A gain is claimed only if both judges move in the same direction.
2. `turkish-v2` metric normalization (applied identically to both sides), unit-tested against trnorm outputs.
3. Metrics: space-free CER (primary), WER (corpus + sentence mean), S/D/I breakdown, share of sentences with WER>50 and
   of error-free sentences; SIM-o (against the original prompt) + real-audio ceiling; DNSMOS SIG/BAK/OVRL + UTMOSv2
   (+ Distill-MOS/NISQA-48k optional); output bandwidth, clipping rate, LUFS, RTF; the prompt's own DNSMOS.
4. Sets: Freya-TR-Eval (by register and length buckets + an 8 kHz column), MiniMax-MLS-Turkish (100 sentences, 2 fixed
   prompts; direct comparison with commercial and large open systems), an internal podcast set of ≥200 cases (≥40 truly
   unseen speakers, with real-audio references), a hard set of ≥100 sentences (numbers, dates, abbreviations, foreign
   names, questions).
5. Ceilings: real audio and real audio passed through DACVAE, ≥200 clips.
6. 3 noise seeds + a second prompt draw; speaker-clustered bootstrap (B ≥ 5000); in a paired comparison, if the CI
   contains 0 the result is a "tie".

---

## 3. Block-by-block evaluation

Each row: **what we do → evidence → decision**.

### 3.1 Codec and latent space

| Topic | Evidence | Decision |
|---|---|---|
| DACVAE structure | A KL-regularized VAE instead of DAC's RVQ; 48 kHz, 25 Hz, C=128; decoder head `tanh(conv(snake(x))) + α·watermark` → the 0.999 "clipping" is actually tanh saturation. The official `encode()` returns a sample; we use the mean | Keep |
| Posterior mean vs sample | Local: posterior std 0.0026–0.0046, sampling changes the log-mel by only 0.021 | The mean is correct [S] |
| Per-channel standardization | Channel stds 0.61–1.00 (1.6× spread); Echo's "it hurt" observation concerns PCA latents with very different variances | Keep; a global-scale A/B is not worth it [W] |
| PCA/dimensionality reduction | Local: 96/64/32 components log-mel 0.67/1.12/1.70 (codec floor 0.58); 120 components are needed for 99 % of the variance | **Reject** [S] |
| Patch 2 | Local: adjacent-frame correlation 0.054; P=1 at 4k steps beat P=2 at 16k steps | Keep patch 1 [S] |
| Latent noise augmentation | Posterior noise is ~0.5 % of σ; the decoder is not robust to noise (0.2σ isotropic error = as much degradation as the codec's own error) | Reject [M] |
| Loudness via latent scaling | Local: −18 dB gain changes the latent norm by only 4 % but changes its direction (cosine 0.86) | Never scale latents [S] |
| Watermark branch | LAION and Irodori turn the watermark off; since this is a cloning system, it should stay on in the released model | A/B for diagnostic purposes only |
| Lower-dimensional/fine-tuned codec | LongCat: at fixed capacity 64 > 128 > 256 dims for TTS; Irodori's 32-d Semantic-DACVAE UTMOSv2 2.28 → 2.40 | Counts as a codec change; out of scope (future work) |

**Conclusion:** The latent space is hard (full rank, temporally white, decoder-sensitive), but the best use achievable
without changing DACVAE is already very close to what we have. The quality gap comes from generator error → the fix
lies in the training objective, guidance and post-training.

### 3.2 Text input and text encoder

- **Unit:** UTF-8 bytes. In Turkish, two-byte letters (ı ü ş ç ğ ö) are ~11.8 % of letters → ~10 % longer sequences;
  the cost is negligible. There is no published work measuring a byte/character penalty for Latin scripts. Turkish
  orthography is 95.4 % transparent (English 36 %); espeak-ng merges Turkish double consonants ("elli" → "eli", "fifty"
  → "hand"); in ZMM-TTS raw IPA is not better than characters; in MaskGCT BPE is worse than G2P (WER 4.04 vs 2.47).
  **Decision: keep bytes [M]; reject phonemes/BPE [M].** Side effects: in LARoPE a two-byte letter takes twice its share
  of the diagonal prior; the byte rule gives too much duration to sentences with ı/ş/ü; CTC is borderline for fast
  speakers (16–19 bytes/s vs 25 fps) and `zero_infinity` silently zeroes those examples. Cheap fix: **make the CTC
  targets Turkish lowercase characters + space (no punctuation)** [W, zero cost]. If retraining, A/B a ~95-symbol
  character vocabulary or positioning the LARoPE keys by character index.
- **Encoder:** in F5, removing the ConvNeXt text blocks takes WER 4.17 → ~5.5; in ZipVoice without a text encoder
  1.69 → 2.04; SupertonicTTS uses 6 ConvNeXt + 4 self-attention. In DiTTo, a 415M ByT5 (text-only) gets WER 6.22 vs 3.07
  for an 85M SpeechT5 trained jointly with speech — **being aligned matters more than size.** In Irodori v4,
  ModernBERT-ja worsened the standard CER. **Decision: keep the current encoder [M]; do not add BERTurk/ModernBERT.** If
  parameters must be cut, the first candidate: 4 → 2 attention blocks (~5M saving) — with an A/B.

### 3.3 Generator DiT block

| Topic | Evidence | Decision |
|---|---|---|
| Attention layout | Cross-attention works at very small scale (SupertonicTTS 19M WER 2.41; DiTTo-S 42M 3.07). The F5/E2 filler-token approach fails to align within 60k steps (20.19 WER in ZipVoice; F5-small 29.5 at 100k). F5's 151M MMDiT "learned fast and collapsed fast"; in DiT-Air a single stream is worse than MMDiT at small scale. Echo-style joint attention has no ablation at ≤100M and removes LARoPE | **Keep** [M] |
| LARoPE | In the same 19M model WER 2.41 → 2.25; on long sentences 4.98 → 2.16; at 200k CER 2.00 → 1.23; robustness to duration scaling. Local: switches on text usage ~10× earlier | **Keep** [S] |
| Normalization | The RMSNorm/LayerNorm difference is negligible at ≤140M (SR-DiT 4.58 → 4.56); the QK-norm effect is small at small scale but it provides stability for high LR/Muon | Keep; RMSNorm optional, for speed only |
| adaLN | Shared + low-rank (EzAudio SOLA / Echo); in DiTTo global adaLN WER 3.38 → 2.93 with 30 % fewer parameters; pure adaLN-single collapses in EzAudio | **Keep** [S] |
| FFN activation | GLU is good at equal parameter count in T5/LightningDiT, neutral/bad in the 140M SR-DiT | SwiGLU/GEGLU only as an A/B [W] |
| Depthwise conv inside the FFN | In ZipVoice, removing the conv modules takes WER 1.69 → 9.79; gains in U-DiT/SANA; FastSpeech CMOS −0.11 (without conv). Counter-evidence: F5 "+Conv2Audio" 4.17 → 5.78 | A/B: k=5 depthwise conv, +0.1M, ~2 % compute [M–W] |
| Value residual | In a 140M DiT (SR-DiT) FID 4.02 → 3.64, the largest single architectural gain in the ablation; ~0 parameters | A/B [W–M] |
| Attention gate | Qwen gated attention (LLM): stability and a higher LR; Echo/Irodori/Darya use it; no TTS ablation | Head-wise gate A/B (+0.1M) [W] |
| Long skip | DiTTo (cross-attention DiT, like ours): input→output skip WER 3.30 → 2.93, SIM-r 0.573 → 0.588. Counter-evidence: F5 (in-context) 4.17 → 5.17 | A/B: concat(h0, h12) → LN → Linear, +0.5M [M] |
| Downsampling / U-Net | Harmful at 25 Hz (DiTTo U-Net 3.70 vs 2.93; local P=2) | Reject [M] |
| Position | RoPE > absolute (SR-DiT, LightningDiT); half-head RoPE is marginal | Keep |
| Depth/width | Most of DiTTo's S→B jump is in SIM; in MobileLLM deep-and-narrow is better; layer sharing lowers quality at equal compute | Optional 16×448 vs 12×512 A/B; no sharing |
| Output head | JiT: x-prediction is needed only when token size ≈ width (128/512 for us); DiT/F5 use adaLN in the final norm | Keep EDM; final adaLN A/B [W] |
| Pooled text in the conditioning | DiTTo: 3.00 → 2.93 | A/B together with final adaLN [W] |

### 3.4 Reference / speaker conditioning

- **Keep in-context infilling.** In Koel-TTS, on unseen speakers in-context (0.637) > frozen SV embedding (0.619) >
  separate encoder + cross-attention (0.601); the separate encoder overfits to seen speakers. YourTTS 87M (global
  embedding) 0.462 vs F5 158M in-context 0.584. [M]
- **Keep the global voice vector;** in MiniMax, embedding + prompt together give +0.016/+0.046. Optional A/B: instead of
  the learned MLP, a frozen **CAM++** 192-d embedding → Linear → adaLN (a prior trained on ~200k speakers; ~0.1M). A
  separate reference encoder only if we move to ≥20 s multi-clip references (Irodori: single clip 0.661 → 30 s 0.752 →
  120 s 0.775).
- **Training pairs:** F5/E2/Voicebox reach a cross-sentence SIM-o of 0.66–0.67 even with within-utterance training only;
  but in VoiceStar, mixing in prompts from a different sentence of the same speaker (CPM) lowers WER 8.49 → 6.42 (SIM
  ~flat) and makes prompt repetition safe. Within-utterance training rewards prosody copying. **Proposal:** 50–60 %
  within-utterance + 40–50 % a different sentence with the same `<episode>_speaker_k` label (concatenating 1–3 clips,
  2–15 s), cutting the prompt at a silence, a 3–12 s prompt distribution defined in seconds. [M]
- **Structural observation (specific to our model):** with the joined layout + LARoPE, frame *i* maps to token *i·S/L*;
  the prior at the prompt/target boundary is correct only when the prompt's frames-per-byte ratio equals the target's.
  The byte rule enforces exactly this; any other duration creates a "break" at the boundary, and within-utterance
  training never shows this break to the model. This is a plausible explanation for why all duration scalings hurt and
  why WER swings between 5–23 % with the prompt. CPM + tempo perturbation of the prompt only (VoiceStar: ±25 %, WER
  6.42 → 5.66) makes the model robust to this break. [M]

### 3.5 Training objective

| Topic | Evidence | Decision |
|---|---|---|
| EDM preconditioning | Local: better than v-prediction at every *t*; x-prediction degrades at the clean end; in SR-DiT x0-prediction is clearly worse in latent space | Keep [S] |
| Logit-normal *t* | SD3: best average rank across 61 formulations; BareWave (TTS): faster early convergence than uniform; switching to uniform in the last phase of training SIM 0.522 → 0.543, UTMOS 3.70 → 3.82 | Keep + A/B uniform in the last 15–20 % [W–M] |
| Batch expansion Ke=4 | SupertonicTTS: Ke 1→4 +64 % time vs 4× B +253 %; local: Ke4 beat Ke2 by 0.02–0.04 WER and preserved SIM | Keep [S] |
| CTC (A-DMA) | F5-small 585 h: text-only CTC WER 7.47 → 2.48; block 8 is good | Keep; convert the targets to characters |
| Text negatives (hinge) | Extra text encoding + generator forward/backward per step → ~20–25 % extra compute. The paper-faithful version of RobustSpeechFlow uses **latent** negatives (length-preserving repeat/skip + silence padding) and the **correct** text; loss L_pos − 0.2·L_rand − 0.2·L_aug, no extra forward pass; Seed WER 1.44 → 1.38 | Replace with ΔFM/latent negatives or remove; A/B at equal wall-clock time [M] |
| **Speech-REPA / A-DMA speech term** | A-DMA: text CTC at 8 + HuBERT alignment at 12 WER 2.35/SIM 0.609 (CTC only 2.60/0.586); main table 2.68 → 1.97, ~2× convergence. BareWave (983M): WavLM REPA WER 3.32 → 2.86, SIM 0.471 → 0.522. AG-REPA: WER 5.82 → 3.45 (weaker evidence). Putting CTC and REPA on the same layer is bad; an early layer is bad | **Add:** precomputed multilingual SSL (mHuBERT-147 or w2v-BERT 2.0) pooled to 25 fps, blocks 10–11, Conv1d(k3) projector, negative cosine, λ 0.5–1; ~<2 % extra compute; storage ~10 GB (~3 GB with PCA-256) [M–S] |
| **TLA-SA speaker alignment** | Intermediate block features are averaged over the target frames and mapped by a per-layer MLP to a cosine against the WavLM-SV embedding; layer weights come from a softmax over the time embedding. LibriTTS F5-like: Sim-WavLM 0.398 → 0.458 and with a **different** SV model (ERes2Net) 0.500 → 0.571 (not metric hacking); CosyVoice 2 100k h: 0.606 → 0.644 (en), 2.9× faster SIM convergence; WER ±0.1 | **Add:** precomputed 256-d WavLM-ECAPA (or CAM++) utterance embedding; ~1.5–3M training-only parameters; <2 % compute [M–S] |
| Model-guidance training | v + w·Δv (w=0.7) in the target → CFG-free sampling; on F5 MOS 4.026 → 4.159, WER 2.28 → 1.96; w ≥ 1 collapses | A/B as a fine-tune: removes CFG saturation and the 2× inference cost [M] |
| Latent SV loss / GRPO | DMOSpeech (distillation context) SIM 0.53 → 0.70; F5R GRPO SIM 0.698 → 0.730, WER 2.10 → 1.48 | Moved to the post-training phase (§3.10) |

### 3.6 Duration

- **Rule vs predictor:** DMOSpeech 2 (Seed-en): ground-truth duration WER 1.821; F5 speed rule 2.028; **supervised
  predictor 3.750 (worse than the rule)**; GRPO-optimized predictor 1.752 (better even than the ground-truth duration).
  So a naive predictor is worse than the rule, and a metric-optimized predictor is best. Irodori v4.1: with the backbone
  frozen and only the duration predictor retrained (to fix overlong predictions), CER 5.35 → 4.69. [S]
- **Short sentences / sentence endings:** F5 slows the speed down for short texts (≤10 bytes), trims the silence at the
  reference edges and adds 50 ms, and ends the reference text with ". "; FreyaTTS applies SFT on short segments and a
  duration floor for 1–2-word inputs; Cross-Lingual F5-TTS 2, with a syllable-based speed predictor trained by adding
  30–70 % silence to prompts, cuts the relative error on silence-padded prompts from 70+ % → 13–19 %. **Whisper side:**
  large-v3 produces hallucinations on 40.3 % of speechless inputs; in Turkish these are fixed strings such as
  "Altyazı M.K." ("Subtitles M.K.") — some of our "filler" errors may be ASR artifacts. The podcast data contains
  untranscribed "yani"/"şey" fillers ("I mean"/"um"); the model may have learned to produce them when it gets too many
  frames.
- **Proposals:** (i) in evaluation, trim trailing silence + Silero VAD + a hallucination-string filter; (ii) a
  silence-aware, syllable-based **articulation rate** rule (prompt edges trimmed, rate computed excluding pauses
  >200 ms, a pause budget from the target's punctuation); (iii) reranking across **duration factors {0.9, 1.0, 1.1}**
  instead of seeds; (iv) the DMOSpeech 2 idea without RL: corpus pairs are synthesized at 5 duration factors and scored
  with CER+SIM, and the predictor is fitted to the best factor; (v) in training, append 0–0.8 s of encoded-silence latent
  to 30–50 % of the targets and compute the loss on those frames too; at inference, duration ×~1.05 + trimming; (vi) in
  ~25 % of the examples `prompt_fraction_max` 0.85 → short-target coverage. Duration changes interact with the LARoPE
  break → they should be handled together with CPM/tempo training.

### 3.7 Optimizer, schedule, EMA, regularization

- **Keep Muon** [M]: in Moonlight ~2× the compute efficiency of AdamW; 1.3–1.4× at 0.1B in fairly tuned comparisons.
  In DiT, vanilla Muon ≈ AdamW, while CMuon, which orthogonalizes adaLN/QKV piece by piece, gives more than 2× — our
  `FUSED_ROWS` implementation already does this. Next: a small LR sweep (0.5×/1.5×), NorMuon (+11 % efficiency, 1.1B).
- **WSD schedule** [M]: WSD with a 1-sqrt cooldown matches cosine and allows branching from any point; running the
  cooldown phase on the clean subset (MiniCPM) and with uniform *t* folds the current two-stage "warm-start fine-tune"
  into a single run.
- **Overfitting** [W–M]: in the 51M model the val flow loss plateaus after 30k; F5-TTS uses 0.1 dropout in the DiT. A/B:
  wd 0.05–0.1 and attention/FFN dropout 0.1.
- **EMA** [M, image]: according to EDM2 the best EMA length is very sensitive to the configuration and to CFG; two EMA
  traces (0.999 and 0.9999) or post-hoc power-EMA snapshots. For fine-tunes of <15k steps, 0.9999 is too slow.

### 3.8 Efficiency (single RTX 4090)

Local analysis: ~44M generator × ~24k frames × (6 + 2 recompute) + hinge ≈ 9 TFLOP/step; at 0.60 s that is
~15 TFLOP/s, i.e. **~9 %** of the 4090's ~165 TFLOPS bf16 capacity — the model is bound by overhead/memory, not by
compute. Code facts: the `.any()`/`isfinite` checks in `mask_values`/`per_example_mse`/`flow_loss`, the `.cpu()` in
`negatives()`, the CTC length list and the loss/gradient checks cause ~10–15 synchronizations per step. Every boolean
mask passed to SDPA disables FlashAttention (it falls back to the memory-efficient kernel).

Recommended order: (1) put the value checks behind a debug flag, produce the negative/CTC lengths in the dataloader
(3–8 %); (2) remove the hinge's extra forward pass (~20 %); (3) compile each `Block` separately (regional compile), pad
the audio length to a multiple of 64 and the text to a multiple of 32, mark only the batch dimension as dynamic;
PyTorch ≥ 2.9 (1.3–1.6×); (4) turn off checkpointing using compile's memory savings, or use selective checkpointing
(1.2–1.3×). Realistic total **1.8–2.5×**. FP8 is slower at these matrix sizes (torchao table); CUDA graphs and
FlexAttention packing once the shapes become static. At inference the text K/V is recomputed at every step → it can be
cached.

### 3.9 Sampling and guidance

- Local: `guidance_until 0.5` saves 15 % compute without changing WER; `noise_scale 0.9` and `duration_scale 0.9` are
  harmful; 16 steps ≈ 32 steps (for WER); CFG 5 is the operating point; APG (η 0.5, β −0.3) and CFG-rescale φ 0.7
  reduced clipping and increased WER. Clipping lowers DNSMOS by only ~0.1 (g=3 → 5: 3.11 → 3.01); APG largely removed
  clipping and gained only +0.07 → most of the gap is not clipping.
- Literature: guidance interval (harmful at high noise, unnecessary at low noise; FID 1.81 → 1.40 on images); image CFG
  fixes mostly fail in TTS (CFG-Zero* and zero-init are worse than baseline CFG on F5); in F5 Euler ≈ midpoint (32 NFE,
  UTMOS 3.90 vs 3.89) but midpoint is 1.7× slower; for SIM, MegaTTS 3 multi-condition CFG (speaker 3.5 / text 2.5)
  SIM-O 0.68 → 0.71; Selective CFG (normal early, speaker-emphasized late) SIM +0.007–0.011 on F5, no gain on zh.
- **Proposals (training-free):** (i) CFG only at t<0.5, with a higher scale in that window (≈6); (ii) APG/rescale in the
  late phase (t≥0.5), plain CFG in the early phase — a small code change for a separate η/φ per window; (iii) **pre-tanh
  gain:** a factor g ≤ 1 on the decoder's Conv(→1) output before the tanh, based on the peak/RMS measured in a first
  pass (controls loudness before the saturating nonlinearity and does not change the trajectory → no WER cost); (iv)
  also score the {32, 64} steps × sway {−1, −0.5} sweep with DNSMOS/UTMOS; (v) compare the per-channel std/kurtosis of
  generated latents with real latents and, if inflated, apply moment matching after sampling; (vi) add DNSMOS/UTMOS and
  SIM terms to the best-of-N selector (for SIM, a speaker model **different** from the evaluation model).

### 3.10 Post-training

- **GRPO with a composite reward** [S, on F5]: FlowTTS-GRPO with a WER + SIM + DNSMOS P.835 reward: DNSMOS 3.15 → 3.41
  (test-en), WER 1.88 → 1.73, SIM2 0.753 → 0.790; 1289 steps, 8 GPUs, group 10, 16 steps, SDE σ=0.5 in a 2-step window.
  Reward-hacking risk: with the SCOREQ reward alone, held-out UTMOS collapsed from 4.51 → 1.23; an equal-weight ensemble
  of WER + Distill-MOS + UTMOSv2 held up. At 66M the cost is dominated by Whisper/scoring, ~1–2 GPU-days. **First, a free
  oracle test:** best-of-8 with DNSMOS+UTMOS → the upper bound that RL could reach. The current `posttrain.reward` gives
  DNSMOS a very low weight.
- **Few-step inference** [M]: the main training should remain plain flow matching; afterwards, CFG-fused
  MeanFlow/IntMeanFlow distillation reduces 32 forward passes to 4 (IntMeanFlow F5: 3 NFE WER 1.60 vs teacher 1.87).
  Training MeanFlow from scratch slows training down.

---

## 4. Path per metric

| Target | Levers (in priority order) |
|---|---|
| **Measurement itself** | SIM-o + CAM++, second ASR judge, deterministic Whisper, `turkish-v2` metric normalization, speaker-clustered CI, speaker leakage check, Freya 8 kHz column, MiniMax-Turkish set |
| **WER / CER** | Evaluation hygiene (VAD, hallucination filter) → articulation rate rule + duration-diverse reranking → Turkish normalization fixes → CTC character targets → tail-silence padding + short-target coverage → speech-REPA → CPM + tempo perturbation → latent negatives → GRPO |
| **SIM** | Correct metric → guidance sweep (joint g 2–5; nested text 2.5 / speaker 3.5; late/early window) → TLA-SA → CPM → more speakers (YODAS/CV/TSC) → best-of-N with a different-model SIM term → frozen CAM++ → adaLN |
| **Audio quality** | UTMOSv2/Distill-MOS/NISQA-48k + prompt DNSMOS → CFG only at t<0.5 → pre-tanh gain → composite best-of-N → late-phase APG → model-guidance fine-tune → GRPO → Sidon-restored data |
| **Training speed** | Removing synchronizations → latent negatives → regional compile + padding → turning off checkpointing → WSD (staged training in a single run) → speech-REPA's convergence speedup |

---

## 5. Data

- **The current filter is at the right level** [S]: in Raon-OpenTTS, dropping the worst 15 % gives WER 2.19 → 2.00 /
  SIM 0.661 → 0.672, while dropping 50 % is harmful (2.32). Our filter drops ~25 % of the rows; do not go beyond 30 %.
  Emilia's DNSMOS ≥3.0-filtered mean is 3.26; our median is 3.28 → data quality is not the bottleneck. Cheap additions:
  a Silero speech-ratio < 0.8 filter, a 1.5×IQR outlier cut on characters per second.
- **Open Turkish data** (Emilia, MLS, VoxPopuli and Granary contain no Turkish):

| Source | Hours / speakers | License | Note |
|---|---|---|---|
| YODAS tr000 (manual subtitles) | 588.7 h | CC-BY-3.0 | Re-transcription + filtering; an estimated ~150–300 h usable |
| YODAS tr100 (automatic subtitles) | 4,067.7 h | CC-BY-3.0 | Noisier; second stage |
| Common Voice tr (v26–27) | ~130 h validated, ~1,840 speakers | CC0 | Best speaker diversity; ~50–80 h after DNSMOS. **Freya's 495 sentences must be removed first** (its short-native section comes from CV17/CoVoST2 texts) |
| ISSAI TSC | 218 h | MIT / CC-BY-4.0 | Human transcripts; a filtered 94 h version |
| KIRAAT | 3,106 h, 90 audiobook speakers | **No license** | Only with legal approval; weights must not be released |
| Evaluation only | FLEURS-tr, MediaSpeech-tr, Antalia | CC-BY | Do not use in training |

  Phase 1 (CV + YODAS tr000 + TSC-94h): ~70 h → ~350–450 h, a few hundred → 2000+ speakers; the expected main gain is in
  SIM and generalization to unseen speakers (ZipVoice 123M: LibriTTS 0.610 → Emilia 0.668). First measure the filter
  yield on a 5 % sample of each source.
- **Restoration** [M]: Sidon (MIT, 48 kHz output, 104 languages): F5 training on TED-LIUM, MOS original 3.25 / Demucs
  3.27 / VoiceFixer 3.77 / **Sidon 4.25**; on Turkish FLEURS DNSMOS 3.07 → 3.45, CER 0.040 → 0.041. Demucs/UVR
  separation gives no gain — dropping clips with music is cheaper.
- **Bandwidth** [S]: a latent model reproduces the bandwidth of its training audio; our MP3s are 12–16 kHz → the output is
  limited to ~16 kHz. Since DNSMOS/UTMOS run at 16 kHz, they cannot see anything above 8 kHz. Steps: a per-clip
  `bandwidth_hz` and an output-bandwidth metric → full-band targets with Sidon → (optional) a bandwidth-bucket embedding.

---

## 6. Small-model proposal

- **Budget:** 12 × 512 (66.5M) should be kept. Evidence: 448 → 512 (+30 %) + clean data brought Freya WER from 9.1 →
  4.3; the DiTTo S (42M) → B (152M) jump is mainly in SIM. The proposed new architecture options total <1.5M (long skip
  0.5M, conv-FFN 0.1M, gate 0.1M, final adaLN 0.04–0.5M); the auxiliary-loss projectors (~3–5M) exist **only during
  training** and are dropped at inference.
- **If it has to be smaller, the order is:** (1) text encoder attention 4 → 2 (~−5M); (2) FFN multiplier 3 → 2.5 +
  conv-FFN; (3) 12 × 448. 12 × 384 (~37M) only as a fast experiment/ablation model.
- **Speed budget:** quality-neutral efficiency work brings 60k updates down to ~4–5.5 hours; 100k+ updates fit into the
  same time. Together with speech-REPA's convergence speedup, the target is to reach the current 60k quality at 30–40k
  updates.

---

## 7. Prioritized roadmap

Detailed work list, issue links and acceptance criteria: [roadmap.md](roadmap.md). Summary:

- **Phase 0 — Measurement and cleanup:** evaluation protocol v2; speaker-clustered statistics; `turkish-v2` text
  normalization; speaker leakage check; documentation cleanup (this report).
- **Phase 1 — Speed and cheap wins:** training speed (synchronization, regional compile, checkpointing); latent
  negatives; duration model and sentence endings; sampler/output quality.
- **Phase 2 — Architecture and objective A/Bs:** DiT block options; speech-REPA + TLA-SA; training pairs (CPM, short
  target, tail silence, CTC character targets); schedule/regularization (WSD, uniform-*t* cooldown, dropout, dual EMA,
  model-guidance).
- **Phase 3 — Data and post-training:** Turkish data expansion pipeline; GRPO with a composite reward.

Every A/B: w512 baseline, 20k updates (~3.2 hours), **baseline with 2 seeds** (to measure the noise floor), a single
variable, evaluated with protocol v2; promote if the CER gain is larger than the seed spread.

---

## 8. Not recommended

Joint/MM-DiT attention (removes LARoPE, evidence of collapse at small scale), F5-style filler-token in-context text
(fails to align at 60k), patching/downsampling at 25 Hz, layer sharing, differential attention, phoneme/espeak and BPE
input, a BERTurk/ModernBERT text encoder, PCA/whitening/latent noise augmentation, loudness control via latent scaling,
x-prediction, MeanFlow/shortcut training from scratch, FP8, CFG-Zero* and initial-noise truncation, music separation
with Demucs, a released model trained on unlicensed data (KIRAAT).

---

## 9. Local measurement archive (evidence preserved from deleted documents)

The measurements below were carried over from the documents of 18–23 September (`iyilestirme-yol-haritasi.md`,
`research-2026-09-21/`); the raw files remain in the git history.

**DACVAE latent probe** (LibriSpeech test-clean, 40 speakers, 234 sentences, 43,682 frames; source 16 kHz → 48 kHz):

| Measurement | Value |
|---|---:|
| Channel std (min / median / max) | 0.61 / 0.69 / 1.00 |
| Posterior std (median) | 0.0034 |
| Channels with SNR < 25 | 0 / 128 |
| KL | ~5.5 nats/channel, 702 nats/frame |
| PCA 90 % / 95 % / 99 % variance | 89 / 102 / 120 components |
| Effective rank (participation ratio) | 77–83 / 128 |
| Lag-1 temporal autocorrelation (median) | 0.054 |

Decoder sensitivity (log-mel L1; original ↔ codec = 0.577): posterior sampling 0.021; isotropic error in normalized
space β=0.05/0.1/0.2/0.3/0.5 → 0.16/0.31/0.57/0.81/1.27; PCA k=96/64/32/16 → 0.67/1.12/1.70/2.26; 1σ noise on a single
channel 0.17–0.42.

**Input level:** −6/−12/−18 dB gain → latent cosine 0.963/0.916/0.858, norm ratio ~0.96–1.04, codec error
1.06×/1.20×/1.42× → level is encoded in the latent as direction, not scale; −16 LUFS normalization is essential.

**Small-scale A/Bs (LibriSpeech, 8 hours):** EDM preconditioning is better than v-prediction at every *t* (−5.4 % at
P=2, −1.5 % at P=1); the P=1 gap widens with training (−10.3 % at 4k → −11.0 % at 16k); LARoPE raises the text gain to
0.0056 at 2k steps (baseline ~0.001 up to 14k steps) but worsens the total flow loss by ~1.2 % → model selection should
not be based on the flow loss; 2 self-attention layers in the text encoder raised the text gain at 16k from 0.0130 →
0.0183. Running both CFG branches in a single batch cut the step time by ~2× (output difference 0).

**Sampler padding:** costing by the speaker's longest reference produced 35.6 % padding waste; with the real epoch cost
+ a frame budget it is 0.5 % (23.0 % → 0.33 % on real data).

---

## 10. Main references

F5-TTS 2410.06885 · E2-TTS 2406.18009 · SupertonicTTS 2503.23108 · LARoPE 2509.11084 · ZipVoice 2506.13053 ·
DiTTo-TTS 2406.11427 · A-DMA 2505.19595 · RobustSpeechFlow 2605.22083 · FreyaTTS 2607.09530 · MegaTTS 3 2502.18924 ·
DMOSpeech 2410.11097 · DMOSpeech 2 2507.14988 · F5R-TTS 2504.02407 · FlowTTS-GRPO 2606.23190 · Flow-GRPO 2505.05470 ·
TLA-SA 2511.09995 · BareWave 2606.09048 · REPA 2410.06940 · HASTE 2505.16792 · ΔFM 2506.05350 · MeanFlow 2505.13447 ·
IntMeanFlow 2510.07979 · Model-guidance 2504.20334 · Selective CFG 2509.19668 · Guidance interval 2404.07724 ·
APG 2410.02416 · VoiceStar 2505.19462 · Koel-TTS 2502.05236 · Voicebox 2306.15687 · MiniMax-Speech 2505.07916 ·
OmniVoice 2604.00688 · XTTS 2406.04904 · Moonlight 2502.16982 · CMuon 2608.02502 · NorMuon 2510.05491 · WSD 2405.18392 ·
EDM2 2312.02696 · SD3 2403.03206 · DiT-Air 2503.10618 · SR-DiT 2512.12386 · LightningDiT 2501.01423 · Gated attention
2505.06708 · Value residual 2410.17897 · LongCat-AudioDiT 2603.29339 · Raon-OpenTTS 2605.20830 · Emilia 2407.05361 ·
Sidon 2509.17052 · Omnilingual ASR 2511.09690 · Whisper hallucinations 2501.11378 · Selector/judge bias 2607.08256 ·
Echo-TTS (jordandarefsky.com/blog/2025/echo) · Irodori-TTS (github.com/Aratako/Irodori-TTS) ·
seed-tts-eval (github.com/BytedanceSpeech/seed-tts-eval) · trnorm (github.com/ysdede/trnorm).
