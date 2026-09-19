# Reproducible experiments and promotion gates

Current status: correctness tooling and optional ablations are implemented. Corpus
quality experiments have **not run**. Use only the user's English corpus for TTS
pretraining, preference candidates, replay and teacher/student data. Synthetic unit
fixtures must not enter a production cache. Frozen external ASR/speaker/DNSMOS models
are optional frontend/evaluation dependencies, not TTS initialization weights.

## Inference: reference audio, no speaker ID

```bash
uv pip install -e '.[codec,asr]'
dacvae-tts infer --checkpoint runs/tiny/last.pt \
  --ref-audio reference.wav --text "This is my new sentence." \
  --steps 16 --guidance 1.5 --seed 42 --profile --output outputs/example.wav
```

```python
from dacvae_tts.inference import Synthesizer

tts = Synthesizer("runs/tiny/last.pt", device="cuda", profile=True)
result = tts.synthesize(
    text="This is my new sentence.",
    ref_audio="reference.wav",
    output="outputs/example.wav",
)

# Reuse the reference without repeating DACVAE encoding or ASR:
voice = tts.prepare_reference("reference.wav")
tts.synthesize("Another sentence.", reference=voice, output="outputs/another.wav")
```

`examples/voice_clone.py` is runnable. A **trained TTS checkpoint is required**;
the preserved initialization snapshot is not a voice model. `small.en` ASR loads
lazily on CPU by default and may download weights on first use. `--asr-model` and
`--asr-device` are configurable. ASR errors can hurt conditioning. Provide an exact
`reference_text=` / `--reference-text` to bypass ASR when available. The architecture
still consumes a transcript internally; a truly transcript-free model has not been
implemented or tested. Automatic ASR dispatch is unit-tested with a mock; real ASR
quality on the user's reference recordings has not been evaluated.

## Gate 1: data integrity and codec-only diagnostics

```bash
dacvae-tts audit-cache --cache /cache/english/merged --scan-latents \
  --output outputs/cache-audit.json
dacvae-tts codec-reconstruct --manifest /dataset/clean-diagnostic.jsonl \
  --cache /cache/english/merged --limit 32 --output outputs/codec
dacvae-tts evaluate --manifest outputs/codec/original.jsonl \
  --dnsmos-model /models/DNSMOS/sig_bak_ovr.onnx --output outputs/original-scores.jsonl
dacvae-tts evaluate --manifest outputs/codec/reconstructed.jsonl \
  --dnsmos-model /models/DNSMOS/sig_bak_ovr.onnx --output outputs/codec-scores.jsonl
```

Input diagnostic JSONL needs accessible `audio` and exact `text`; include `id` and
`speaker_id` for later grouping. Reconstruction uses production posterior means,
default cache float16 rounding, training stats, denormalization, decoder and trimming
to original samples. Compare intelligibility, consonants, boundaries, naturalness and
identity by listening, WER/CER and original-reference speaker similarity. Codec
reconstruction is a diagnostic reference, not a mathematical upper bound on scores.
The generic `compare` command expects synthesis case keys; codec manifests instead
provide paired original/reconstruction rows in the same order and separate summaries.

Inspect rejected JSONL files, retained speakers/hours, dominant-speaker fractions,
duplicate-label conflicts, supplied source intervals, and cross-split speakers. Exact
hashes and supplied intervals do not establish near-duplicate freedom or correct
transcripts. Review ASR disagreements against original speech and listen. No aggressive
silence trimming is applied. If adding trimming, version it and regenerate caches.

Optional source columns are `spoken_text`, `session_id`, `source_recording`,
`start_seconds`, `end_seconds`. Preparation preserves original/normalized text in
SQLite provenance. Default `unicode-v1` retains baseline behavior. Optional
`--text-normalization english-explicit-v2` rejects common numeric/date/time/currency
and selected abbreviation patterns unless already supplied in spoken form; it does
not claim a complete English pronunciation normalizer. Resolve ambiguous expressions
in the corpus, rather than guessing dates or units. Special token IDs do not change.

## Gate 2: demonstrate learnability on clean speech

Create a small separately versioned manifest with a few reliable speakers, several
distinct complete utterances each, and deliberate speaker-disjoint train/val/test
assignments. The training subset should be small enough to memorize. Then:

```bash
dacvae-tts prepare --manifest /dataset/clean-small.jsonl --output /cache/clean-small/part
dacvae-tts merge --inputs /cache/clean-small/part --output /cache/clean-small/merged
dacvae-tts train --config configs/experiments/tiny_learnability.yaml \
  --cache /cache/clean-small/merged --output runs/learnability
dacvae-tts make-cases --cache /cache/clean-small/merged --split train --limit 16 \
  --output data/learnability-cases.jsonl
```

Use `configs/evaluation.yaml` with `checkpoint: runs/learnability/last.pt`,
`cases: data/learnability-cases.jsonl`, a new output directory and
`sampling.duration: [ground_truth]`. Run it at saved checkpoints and listen:

```bash
dacvae-tts run-eval --config configs/learnability-evaluation.local.yaml
```

The `.local.yaml` is your edited copy of the supplied evaluation config. Reference and
target must be different utterances; the train-split check tests memorization only.
Generation uses EMA weights, so account for EMA lag in short overfit runs. Five hundred
updates are an initial bounded experiment, not a guaranteed learning budget. Require
recognizable target content, no reference-text leakage and usable reference conditioning
before scaling. Investigate failures in preprocessing, packing/masks, alignment and
objectives first. Lower latent loss alone does not pass this gate.

## Gate 3: freeze evaluation and run one change at a time

```bash
dacvae-tts make-cases --cache /cache/english/merged --split val --limit 1000 \
  --seed 42 --output data/evaluation-cases.jsonl
# Optional separate protocol if session metadata exists:
dacvae-tts make-cases --cache /cache/english/merged --split val --limit 1000 \
  --seed 42 --cross-session --output data/cross-session-cases.jsonl
dacvae-tts run-eval --config configs/evaluation.yaml
```

Edit checkpoint/cases/output paths first. Export rejects known overlapping reference/
target intervals, reused recordings, equal waveform hashes and cross-split speaker
leakage; it cannot verify unknown source intervals or approximate duplicates. Embedded
audio needs accessible original-file exports before `make-cases`. Cases and index
digests are saved. Add curated `tags` for names, numbers, abbreviations, repeated
phrases, long text, punctuation, short/noisy references and style mismatch; automatic
linguistic curation or per-tag summary reports are not implemented. Keep changed-text
cases separate from duration-oracle cases: original target duration is valid only for
its original target text.

The config defaults to 32 cases and predicted/ground-truth duration at identical
sampler settings. Increase limits deliberately after gate 2. Add [0.9,1.0,1.1] duration
scales in a separate duration perturbation experiment. Each variant saves WAVs, JSONL,
duration absolute/relative error, profiles, branch evaluations and configuration.
It generates one sample/request/seed and performs **no best-of-N selection**. Optional
`judges` enables scoring; otherwise `quality_metrics` explicitly says not scored.
Missing judges do not produce synthetic scores. Errors abort a run rather than quietly
dropping failures; investigate and report failure counts before comparisons.

Use several seeds to quantify variability. `compare` requires matching cases and
evaluator identities and provides paired speaker-clustered bootstrap intervals:

```bash
dacvae-tts compare --before outputs/baseline/n16-g1.5-s-1.0-dpredicted-x1.0-seed42/manifest.jsonl \
  --after outputs/candidate/n16-g1.5-s-1.0-dpredicted-x1.0-seed42/manifest.jsonl \
  --output outputs/comparison.json
```

Use the actual emitted variant directory names (YAML integer/float formatting can
change the name). Both manifests must contain evaluator scores. The strict built-in
metric gate requires WER/CER upper confidence bounds below zero, DNSMOS lower bound
above zero and speaker-similarity lower bound at least zero for candidate-minus-baseline.
It does not replace listening, subgroup review or performance criteria. Record runtime
tolerance **before** running an experiment in `experiment-results.template.json`;
an unset tolerance means a promotion decision is incomplete. A distillation experiment
may preregister a different explicit quality/latency trade-off, rather than claim a
strict quality improvement. Retain an untouched test split for final decisions.

WER/CER now default to `english-unicode-v2`: NFKC, lowercase, Unicode letters/digits/
combining marks and apostrophes; CER excludes spaces. `legacy-ascii-v1` reproduces old
metric normalization. Rescore both checkpoints under one version. Word substitutions,
deletions and insertions are recorded; repetitions/order and reference-text leakage
still require transcript/audio inspection. Evaluator metadata records ASR identity,
speaker model identity and DNSMOS file digest. Pin local judge revisions for strict
reproduction; ASR model name/package version alone does not hash every downloaded file.

## Prioritized ablation matrix

| Experiment | Config/control | Status |
|---|---|---|
| Codec reconstruction | `codec-reconstruct`, original vs reconstructed speech | Synthetic codec path tested; speech evaluation not run |
| Duration oracle vs predicted | `evaluation.yaml`; fixed seed/reference/sampler | Synthetic execution tested; speech attribution not run |
| Full reference / summary / both | `model.reference_paths`; `run-eval.reference_paths` can override for diagnostic path removal | Implemented; no speech comparison |
| Temporal summary | `tiny_temporal_mean.yaml` | Optional, shape/mask/gradient tested |
| Attentive / mean-std pooling | `tiny_temporal_attention.yaml`, `tiny_temporal_statistics.yaml` | Optional; compare to temporal mean to isolate pooling |
| Duration text statistics | `tiny_duration_features.yaml` | Optional; adds character count, ASCII punctuation fraction and word-count features; tested |
| Frame-weighted flow | `tiny_frame_weighted.yaml` | Optional; hand formula and both DDP/resume modes tested |
| Partial speaker balancing | `tiny_speaker_balanced.yaml` | Optional deterministic replacement sampling; no quality measurement |
| Packing P=1 vs P=2 | `tiny_packing_one.yaml` vs Tiny | Config available; separate compatible weights required |
| Steps/guidance/sway | Configured sampler grid; initial noise generated before packing | Analytic solver tests pass; no speech-quality/latency comparison |
| Tiny vs Small | Existing baseline configs | Architecture implemented; comparative training not run |
| Character/phoneme frontend, auxiliary alignment | Requires verified alignment bottleneck and controlled budget | **Deferred; not implemented** |
| Speaker contrastive/teacher identity objective | Requires verified conditioning bottleneck and reliable labels | **Deferred; not implemented** |
| Preference / trajectory distillation | Existing commands, frozen teacher/reference and replay | Synthetic loss/distributed tests; quality gains not established |

All experiment YAMLs are under `configs/experiments`. Train each with the same corpus,
budget and held-out cases only after earlier gates pass. Changing P or D changes compute;
report updates, audio hours seen, GPU-hours, parameter count and memory rather than
calling unequal budgets equal. Replacement speaker sampling can repeat examples within
an epoch or across ranks; alpha=0 retains original without-replacement behavior,
alpha=1 weights each row by inverse speaker recording count. Path removal at inference
is a diagnostic intervention, not proof that a separately trained path-only model is
equivalent. Interactions need experiments after individual changes succeed.

## Profiling and post-training

`--profile` synchronizes GPU stage timers: text encoding, reference summary, iterative
generation and waveform decoding. Reference codec/ASR preparation and model load are
reported separately. Duration timing includes its text/reference encoders and head.
RTF covers generation and decoding; request latency additionally includes preprocessing
and duration prediction; output-file writing is excluded. Peak CUDA allocation is
reported for generation. Detailed per-component memory attribution and repeatable
throughput benchmarks remain to be run on the target eight-GPU hardware.

First-call kernel/model/ASR startup and warm requests must be separated. The evaluation
runner reuses prepared references across variants and reports cache hits and charged
preparation time; this alone can make later requests cheaper. Measure cold start in
fresh processes and alternate variants in repeated warm trials before speed claims.
Current smoke timings are for a tiny random/toy model on tones and do not benchmark
Tiny/Small speech production. `torch.compile`, BF16 and SDPA are available; verify
output tolerances and end-to-end gains before adopting hardware-specific settings.

The only new sampling optimization removes redundant null text encoding, constructing
the tested identical zero feature cache instead. No DiT hidden-state caching, CFG
branch batching or streaming is claimed. g=1 avoids the null evaluation; g=1.5 at
16 steps still costs 32 branch evaluations/calls.

Follow the [README post-training commands](../README.md#post-training) only after a
teacher passes speech gates. Existing synthetic tests verify preference direction,
reference freezing, student gradients and distributed execution. Winner/loser errors
are target-frame/channel normalized and pairs use compatible durations; nevertheless
verify duration distributions in real candidates. The flow-error preference surrogate
is not established as a likelihood objective. Set `--anchor 0` and `--replay-weight 0`
in separate controlled runs to measure each contribution; do not remove both by default.
Replay currently uses baseline per-utterance weighting and duration weight 0.1; when
starting from other loss ablations, document this difference rather than assuming the
pretraining objective carries over automatically.

Use a different ASR judge for final evaluation where feasible and blinded randomized
paired listening for identity/naturalness separately. Report all best-of-N candidates,
selection method and generation/judge cost. Distillation records teacher sampler and
coarse trajectory targets; evaluate same duration/seed/cases against the teacher and
check omissions, repetitions and identity drift. Post-training does not currently
resume optimizer/data state. Neither post-training method has demonstrated a gain on
the user's corpus.

## Local verification

```bash
.venv/bin/ruff check src tests examples scripts
.venv/bin/pytest -q
# Optional bounded real-codec integration test; new output directory required:
.venv/bin/python scripts/smoke_pipeline.py --output /tmp/dacvae-smoke-new
```

The last command explicitly creates six synthetic tones, encodes/merges/audits them,
reconstructs two, runs **two width-32 toy updates**, exports a cross-session fixture,
then generates both duration variants. It is not the Tiny clean-speech overfit gate.
It needs CUDA, codec extras and access to codec weights. Unit tests otherwise run on
CPU and mock ASR/codec where appropriate. See `validation-results.json` for observed
results. No automatic full-corpus, eight-GPU, preference or distillation job was launched.
