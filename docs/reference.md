# DACVAE-TTS — detailed reference (previous README)

English voice-cloning TTS trained from scratch in frozen DACVAE latent space.
Designed for **4 million text/audio rows and 8 GPUs**, including 8×4090 or 8×A100.
The code is a research implementation, not a pretrained voice model. Dataset access
and speaker IDs are still needed before a real training run or quality claim.

| Configuration | Trainable TTS parameters | With the frozen codec | Default global batch on 8 GPUs |
|---|---:|---:|---:|
| Tiny | 13,526,017 | 121,197,188 | 256 utterance pairs |
| Small | 44,361,473 | 152,032,644 | 256 utterance pairs |

Counts include the text encoder, reference encoder and duration head. The codec
alone has 107,671,171 parameters. Frame-budget limits can reduce the actual batch.
The tested `facebook/dacvae-watermarked` checkpoint is 48 kHz, 128 channels,
25 latent frames/s; the adapter probes these properties instead of assuming them.

The [architecture and quality research report](research-report-2026-09-24.md) for the Turkish model
reviews every block, layer and training technique against the 2024–2026 literature, with
the evidence for each decision, and the [roadmap](roadmap.md) links the resulting
GitHub issues. Core capabilities:

- Parallel conditional flow generation with cross-attention to English byte text.
- Reference-audio prefix plus pooled voice conditioning; an optional ASR frontend
  supplies the reference transcript automatically for audio-only requests.
- Two-frame reversible packing, native SDPA, BF16, Muon (AdamW for non-matrix parameters), cached text
  encoding during sampling, optional `torch.compile` and activation checkpointing.
- Streaming JSONL/Parquet preparation, binary latent shards, SQLite metadata,
  train-only normalization, exact-audio deduplication and speaker-disjoint splits.
- DDP, rank-balanced length buckets, gradient accumulation, EMA, cosine LR, validation
  loss, atomic checkpoints and exact same-topology pretraining resume.
- An exact fast-dacvae adaptation with optional codec compilation/CUDA graphs;
  parallel CPU or GPU preparation, spawned training workers and CUDA transfer prefetch.
- WER/CER, official-model DNSMOS and speaker verification evaluation; offline
  preference learning with real-data replay; optional teacher-trajectory distillation.

## Install

Use Python 3.10–3.12. Install a matching PyTorch/torchaudio build for your CUDA driver
first. This environment was tested with Python 3.12 and PyTorch/torchaudio 2.8.0 CUDA 12.8.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e '.[codec,data,dev]'
# Optional, separate frozen evaluation models:
uv pip install -e '.[eval]'
# Optional reference-audio-only convenience API:
uv pip install -e '.[asr]'
dacvae-tts inspect --config configs/tiny.yaml
pytest -q
```

The DACVAE source dependency is pinned to a reviewed revision. Its upstream codec
and checkpoint retain their own licenses. No pretrained TTS or language-model
weights initialize this model. ASR/speaker/DNSMOS judges are external pretrained
evaluators; all TTS training examples and teacher/student data come from your corpus.

## Dataset and preprocessing

JSONL example (or equivalent Parquet columns):

```json
{"id":"en-000001","audio":"audio/000001.wav","text":"A complete spoken sentence.","speaker_id":"speaker_17","language":"en"}
```

Required: `text`, `audio`, **reliable `speaker_id`**. Optional: globally unique `id`,
`language`, `split` (`train`, `val`, `test`). Audio may be a path or a Parquet/HF-style
object with `bytes`/`path`. A directory of JSONL/Parquet files is supported; audio
paths are relative to the manifest's parent or the supplied dataset directory.
Hugging Face datasets must first be exported or downloaded to local Parquet/JSONL.
Column names are configurable with `--text-column`, `--audio-column`, `--speaker-column`.

Do not replace missing speaker IDs with row IDs or one shared speaker ID. Accurate
cross-utterance pairing is central to voice cloning. Text/audio alone does not
establish identity. Each retained speaker needs at least two complete utterances.
This implementation does not diarize or cluster unknown speakers automatically.

Prepare spoken forms for numbers, abbreviations and unusual symbols. The text
frontend normalizes Unicode/whitespace and preserves punctuation; it does not guess
how a date, currency value or acronym should be spoken. Keep transcripts exact.
Default duration bounds are 1–15 seconds and are configurable. Whole rows outside
the bounds are rejected; audio is never cropped while retaining the original text.

```bash
bash scripts/prepare_8gpu.sh /dataset/manifest.jsonl /cache/english
# Or /dataset/parquet-directory instead of the JSONL file.
# Explicit per-GPU throughput controls:
bash scripts/prepare_8gpu.sh /dataset/parquet-directory /cache/english-fast \
  --workers 4 --prefetch 16 --batch-size 8 --bucket-size 256 \
  --batch-seconds 120 --precision fp32
# Optional compiled native-convolution backend and spawned CPU decoding:
bash scripts/prepare_8gpu.sh /dataset/parquet-directory /cache/english-compiled \
  --codec-compile --worker-backend process --workers 2 --prefetch 8
# CPU-only preparation with two codec processes:
PREPARE_PROCESSES=2 bash scripts/prepare_cpu.sh /dataset/parquet-directory /cache/english-cpu
```

This starts one frozen-codec encoder per GPU, writes `part-0` … `part-7`, then merges
their indexes. Each GPU overlaps bounded CPU decoding/resampling with codec encoding,
batches equal hop-rounded lengths, folds frozen encoder weight normalization once,
and leaves the unused decoder off GPU. A CUDA batch OOM retries smaller batches;
single-recording OOMs and other codec failures stop the job instead of dropping data.
The launcher respects an existing eight-device `CUDA_VISIBLE_DEVICES` assignment.
Both launchers use `--codec-backend fast` with native convolutions; compilation and
CUDA graphs are opt-in. The reference backend remains available for comparison.
`src/dacvae_tts/fast_codec.py` documents the fast codec's provenance and its exactness
checks (`tests/test_fast_codec_parallel.py`); the license is in `third_party/fast-dacvae/`. The fork's
channels-last layout is experimental and was substantially slower in strict FP32
on the tested GPU, so it is not the default.

Directory inputs distribute files across encoders; single Parquet inputs distribute
row groups when possible. A single JSONL file is scanned on each worker, but only
that worker's rows are deserialized. Parquet shards avoid the repeated scan at 4M rows.
Encoding uses posterior means, strict FP32 convolution (TF32 disabled), and float16
latent storage. Normalized UTF-8 text bytes are cached in SQLite before training;
training only assembles reference/target segments and the existing special tokens.
Old caches without text bytes still work through the original text fallback.

`--bucket-size` bounds decoded lookahead RAM; `--batch-size` and `--batch-seconds`
bound GPU batches. Exact-length buckets preserve each utterance's convolution
boundaries; arbitrary zero-padding to a larger recording is not used. A recording
must individually fit `--batch-seconds` after rounding to a codec hop. Preparation
records per-partition throughput and logs rejected rows, including corrupt files.
Partially completed partitions must be rerun into new directories; preparation does
not currently support in-place resume.

`--precision bf16` is experimental, changes cached latents, and is **not** the default.
Do not mix FP32/BF16 partitions; merge rejects them. A synthetic real-codec benchmark
found about 5.4% relative latent RMS error with BF16, so validate codec reconstruction
and speech metrics on real data before using it. No BF16 quality claim is made.

`scripts/benchmark_prepare.py` and `scripts/benchmark_fast_codec.py` reproduce the
preparation throughput measurements.

For manually partitioned jobs, call `dacvae-tts prepare --manifest ... --output ...
--shard-index 0 --num-shards 8`, then:

```bash
dacvae-tts merge --inputs /cache/english/part-{0..7} --output /cache/english/merged
```

By default, stable speaker hashes allocate 98%/1%/1% to train/validation/test. Explicit
splits must remain speaker-disjoint. Merge removes exact duplicates and singleton
speakers, rejects split leakage and ID collisions, and recomputes normalization on
retained training frames only. Review split counts and rejection logs before training.
Very small speaker inventories need deliberate split assignment and cannot support
strong claims about general voice cloning.

At the tested codec rate, latent storage is about 6,400 bytes per audio second.
For example, 4M clips averaging 10 seconds need roughly **256 GB** of latent storage,
plus metadata and the original corpus. Actual requirements depend on clip duration.
Merge references absolute binary-shard paths without copying the audio latents;
keep the partition directories and mount them at the same paths on every node.

## Nano recipe (`configs/nano.yaml`)

`configs/nano.yaml` is a 49.9M-parameter configuration whose choices are reviewed in
[the research report](research-report-2026-09-24.md); every option is off in the Tiny/Small
configs, so the earlier baselines are untouched.

- **No speaker labels needed.** `train.pairing: within` cuts the voice prompt from the start of the
  target recording (10–60% of its frames); `train.prompt_dropout` trains prompt-free synthesis.
  Merge with `--keep-singletons` to keep speakers that have one recording. The `joined` text layout
  feeds the transcript as one stream without a prompt/target boundary token; at inference the prompt
  transcript and the target text are joined with a space.
- **Alignment.** Rotary positions in self-attention, length-aware rotary positions in the text
  cross-attention (a frame 40% into the audio starts out attending near 40% of the transcript),
  QK-norm, self-attention blocks in the byte text encoder, and an auxiliary CTC head on intermediate
  generator frames (training only). Output length follows the prompt's speaking rate
  (`model.duration: rule`), which also keeps the length-normalized positions consistent.
- **Objective.** Single-frame tokens (`patch_size: 1`), an EDM-style unit-variance target
  (`prediction: edm`), stratified logit-normal time sampling, frame-weighted loss, context-sharing
  batch expansion (each utterance gets several time/noise draws per step), Muon, EMA warm-up and
  permanent `step-N.pt` snapshots every `keep_every` updates.
- **Data.** `prepare --loudness -16` normalizes every recording to −16 LUFS (DACVAE's own API
  convention) and records it in the cache, so inference prompts receive the same treatment.
  `--quality-column/--min-quality` and `--reject-digits` filter rows before decoding audio.
- **Tokenization happens in `prepare`, not in training.** Each transcript is normalized once and
  stored as ready token ids (`[BOS] byte+4 … [EOS]`, uint16) in the cache's `token_ids` table; the
  training loader only concatenates cached id arrays (prompt ids + target ids for cross-utterance
  pairs). Older caches without the table fall back to cached bytes or raw text.
- **Other languages.** The tokenizer is UTF-8 bytes, so Turkish or any other script needs no
  vocabulary change; pass `--languages tr` (comma-separated tags, or `any`) to `prepare`, since the
  default accepts only English tags, and `--language tr` / `--asr-language tr` to the evaluation and
  ASR-frontend commands so Whisper transcribes the right language.

```bash
HF_TOKEN=... python scripts/prepare_hf_shards.py --repo ORG/DATASET --total 324 --shards 0-323 \
  --output data/corpus --speaker-column speaker --quality-column quality_score --min-quality 55 \
  --reject-digits --loudness -16 --max-seconds 20
dacvae-tts merge --inputs data/corpus/parts/part-* --output data/corpus/merged \
  --keep-singletons --drop-conflicting-duplicates
dacvae-tts train --config configs/nano.yaml --cache data/corpus/merged --output runs/nano --frame-budget 8000
python scripts/monitor.py --run runs/nano --cache data/corpus/merged --cases 48   # WER/SIM per snapshot
```

The shard script streams one Parquet file at a time (download, encode, delete), so the raw corpus
never has to fit on disk, and it is restartable. For a multi-million-row corpus on an eight-GPU
machine, `scripts/prepare_hf_8gpu.sh` runs one encoder per GPU (shards are dealt round-robin) and
merges at the end; each shard is encoded in a fresh process, so memory stays flat over thousands of
shards:

```bash
HF_TOKEN=... bash scripts/prepare_hf_8gpu.sh ORG/DATASET 1500 data/corpus \
  --speaker-column speaker --quality-column quality_score --min-quality 55 --reject-digits \
  --loudness -16 --max-seconds 20
bash scripts/train_8gpu.sh configs/nano.yaml data/corpus/merged runs/nano --frame-budget 16000
```

Budget one encoder at roughly 200x real time per GPU (about 4.7 hours of audio per 85 seconds), so
2,000 hours take about 1.3 hours on eight GPUs. `--frame-budget` sets the padded frames per rank and
micro-batch: 8,000 fits 16 GB with the contrastive term, 16,000 needs about 20 GB.

Two further changes were added on 22 September after the first 40k-update checkpoint (whose
transcripts mostly failed by repeating or dropping a single word): `model.adaln_rank` replaces the
per-block D→9D modulation with one shared modulation plus a rank-64 correction per block (the freed
parameters bought width 384→448 at the same 51M budget), and `train.contrastive_weight` adds
skip/repeat transcript negatives (arXiv:2605.22083): the same audio, noise and time are scored with a
transcript that is wrong by one word, and the true transcript must win by a margin.

Speech-quality results for this recipe are reported in the roadmap document as they become
available; nothing here is a quality claim.

## Experiment tracking (Weights & Biases)

`wandb login` once (the package is installed by `scripts/setup.sh`), then pass
`--wandb-project NAME` to `train` (or set `train.wandb_project` in the YAML). Rank 0 mirrors every
`train.jsonl` record (`train/flow`, `train/ctc`, `train/contrastive`, `train/lr`, gradient norm,
throughput) and every validation record (`val/validation_flow`, `val/validation_text_gain`) to a run
named after the output directory; `--resume` continues the same W&B run, `--wandb-group` groups
sweeps and `--wandb-id` overrides the run id. `scripts/monitor.py --wandb-project NAME` attaches to
the same run and adds `monitor/wer`, `monitor/cer`, `monitor/speaker_similarity` and a few audio
samples per snapshot. Local JSONL logs are always written; `WANDB_MODE=offline` keeps everything on
disk for a later `wandb sync`.

## Eight-GPU pretraining

```bash
bash scripts/train_8gpu.sh configs/tiny.yaml /cache/english/merged runs/tiny
bash scripts/train_8gpu.sh configs/small.yaml /cache/english/merged runs/small
```

The default optimizer is **Muon** for hidden weight matrices (attention, feed-forward, adaptive
modulation and encoder MLP projections; fused key/value and modulation rows are orthogonalized
separately) with AdamW for embeddings, convolution filters, boundary projections, output heads,
biases and gains. Its updates are RMS-matched to AdamW, so `learning_rate` and `weight_decay` keep
their meaning. `train.optimizer: adamw` or `--optimizer adamw` restores the previous plain AdamW.
Muon has not been compared against AdamW on real speech metrics here; the small-scale loss
comparison is summarized in [the research report](research-report-2026-09-24.md) (§3.7, §9).
Post-training reuses the optimizer recorded in the checkpoint unless `--optimizer` is passed.

Tiny defaults to 16 pairs/GPU × 2 accumulation × 8 GPUs; small uses 8 × 4 × 8.
Treat these as starting points for throughput profiling, not GPU-specific limits.
Adjust `--batch-size` and `--accumulation` while keeping the global batch stable.
`--frame-budget` caps padded prompt+target latent frames per rank/microbatch;
`--compile` is optional and should be benchmarked after warmup. Small enables
activation checkpointing by default; disable it in YAML when recomputation hurts
throughput. CUDA uses BF16; CPU/diagnostic runs can use `--precision fp32`.

Cached latents keep the 107.7M-parameter codec out of the TTS training loop. Numeric
row indexes are held in memory; transcripts and latent arrays are loaded on demand.
Bucketing bounds padding waste. Global batch plans give all ranks equal optimizer
steps, and losses are weighted by the actual number of examples across accumulation
and ranks. NCCL topology settings are deliberately left to your cluster environment.

```bash
bash scripts/train_8gpu.sh configs/tiny.yaml /cache/english/merged runs/tiny \
  --resume runs/tiny/last.pt
```

Resume requires the same world size, configuration, cache path and frame budget.
`--stop-after N` checkpoints gracefully at update N without changing the LR schedule.
The last checkpoint includes model/EMA, optimizer, LR step, sampler position and
per-rank RNG. Copy checkpoints you want to retain before `last.pt` is replaced.
Validation here measures flow/duration loss; generate held-out audio to measure WER,
CER, DNSMOS, identity and naturalness. Do not select models on flow loss alone.

For two nodes with four GPUs each (still eight total), run on both nodes with the
appropriate node rank and a reachable rendezvous endpoint:

```bash
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
  --rdzv_id=english-tts --rdzv_backend=c10d --rdzv_endpoint=HOST:29400 \
  -m dacvae_tts train --config configs/tiny.yaml \
  --cache /cache/english/merged --output runs/tiny
```

Use `--node_rank=1` on the second node. All ranks must see the cache at identical paths.

## Voice cloning and evaluation

No speaker ID is used at inference:

```python
from dacvae_tts.inference import Synthesizer

tts = Synthesizer("runs/tiny/last.pt", device="cuda")
tts.synthesize(
    text="These are the words I want the model to say.",
    ref_audio="reference.wav",
    output="outputs/example.wav",
)
```

```bash
dacvae-tts infer --checkpoint runs/tiny/last.pt \
  --ref-audio reference.wav \
  --text "These are the words I want the model to say." \
  --steps 16 --guidance 1.5 --output outputs/example.wav
```

Omitting `reference_text` invokes optional ASR (CPU by default): English `small.en`, or Turkish
`large-v3-turbo` for checkpoints with a `turkish-*` text normalization (`--asr-model/--asr-language` override).
The TTS model still needs the transcript internally; this is not a transcript-free
architecture. Supply `--reference-text "The exact words in the reference."` to bypass
ASR. See [the runnable example](../examples/voice_clone.py); `dacvae-tts codec-reconstruct`,
`make-cases`, `run-eval` and `compare` cover codec diagnostics and bounded ablations.
A trained checkpoint is required; the repository does not ship a trained voice model.

Use a clean complete reference utterance, initially around 3–10 seconds. The learned
duration head predicts output length; `--seconds` overrides it and `--duration-scale`
adjusts it. Full-sequence synthesis currently limits reference/target to 30 seconds
each. Long text should be split into sentences. Output is mono float WAV with a JSON
timing sidecar. RTF includes latent generation and DACVAE decoding, excludes model
loading/reference encoding, and includes compile warmup if enabled on that call.
The upstream watermarking decoder is preserved. This is not a streaming model.

Generate a held-out evaluation pool with one candidate per target:

```bash
dacvae-tts candidates --checkpoint runs/tiny/last.pt --cache /cache/english/merged \
  --split val --limit 1000 --candidates 1 --output outputs/baseline
dacvae-tts evaluate --manifest outputs/baseline/candidates-000.jsonl \
  --dnsmos-model /models/DNSMOS/sig_bak_ovr.onnx --output outputs/baseline-scores.jsonl
```

Obtain the standard non-personalized ONNX model from
[Microsoft DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS/DNSMOS).
Evaluation uses faster-whisper `large-v3` by default and WavLM speaker verification;
both can be changed. Run judges separately from synthesis to avoid competing GPU
allocations. CPU evaluation is supported. Missing DNSMOS means that metric is absent,
not estimated. Preference ranking requires all metrics.

WER/CER are corpus ratios of summed edit counts; insertions can make them exceed 1.
CER excludes spaces; both use documented case/punctuation normalization. DNSMOS
reports SIG, BAK and OVRL, not P.808 MOS. Candidate pools use reconstructed reference
audio for speaker scoring, which must remain consistent across comparisons. Also
audit against original clean references before making a deployment claim.

Speaker cosine is not a cloning success percentage. Calibrate a verifier threshold
on held-out real same/different-speaker pairs, combine it with intelligibility gates,
then report passing utterances and speaker-level intervals. Human listening remains
necessary for naturalness and identity.

## Post-training

First train a good baseline; post-training cannot repair an untrained model. Use only
training-split prompts/texts for preference generation, and keep validation/test out
of the optimization loop.

```bash
dacvae-tts candidates --checkpoint runs/tiny/last.pt --cache /cache/english/merged \
  --split train --limit 10000 --candidates 4 --output outputs/preferences
dacvae-tts evaluate --manifest outputs/preferences/candidates-000.jsonl \
  --dnsmos-model /models/DNSMOS/sig_bak_ovr.onnx --output outputs/preference-scores.jsonl
dacvae-tts rank-pairs --scores outputs/preference-scores.jsonl --output outputs/pairs.jsonl
torchrun --standalone --nproc_per_node=8 -m dacvae_tts post-train \
  --mode preference --checkpoint runs/tiny/last.pt --cache /cache/english/merged \
  --data outputs/pairs.jsonl --output runs/preference --steps 1000
```

Generation supports `--shard-index`/`--num-shards`; launch one job per GPU into the
same output directory and concatenate its distinct JSONL manifests before scoring.
All jobs must use the same seed, limit, sampler settings, cache and checkpoint.
The scripts use small, bounded offline pools; preferences are not built over all 4M
rows at once. Rank weights/gates are starting hyperparameters to validate. Keep raw
per-candidate metrics. Selection skips near ties and penalizes semantic errors;
quality gains cannot compensate for worse voice similarity or transcript accuracy.

The training loss compares winner/loser flow errors relative to a frozen baseline,
uses identical noise/time for each comparison, and adds a winner reconstruction
anchor plus real-data replay. This is an **experimental flow preference surrogate**,
not exact DPO log likelihoods or GRPO. Post-training checkpoints retain model/EMA and
provenance; unlike pretraining they do not currently resume optimizer/data state.
Use short, evaluated experiments and restart a new experiment from a saved checkpoint.

After metric tuning, optionally reduce inference steps:

```bash
dacvae-tts distill-cache --checkpoint runs/preference/preference-001000.pt \
  --cache /cache/english/merged --teacher-steps 32 --student-steps 8 \
  --limit 10000 --output outputs/distill
torchrun --standalone --nproc_per_node=8 -m dacvae_tts post-train \
  --mode distill --checkpoint runs/preference/preference-001000.pt \
  --cache /cache/english/merged --data outputs/distill/trajectories-000.jsonl \
  --output runs/distill --replay-weight 0.1
```

Use the distilled checkpoint with `--steps 8 --guidance 1` and the same sway schedule.
Teacher guidance is already incorporated into its target trajectories. This is an
experimental approximation, not a promise that eight steps preserve quality. The
teacher must be a model trained on this same English dataset.

Generate the same held-out evaluation pool with the new checkpoint and compare:

```bash
dacvae-tts compare --before outputs/baseline-scores.jsonl \
  --after outputs/new-scores.jsonl --output outputs/comparison.json
```

Comparison requires matching utterance IDs, references and seeds, reports paired
speaker-clustered bootstrap intervals, and fails its metric gate unless WER/CER
decrease, DNSMOS increases, and speaker similarity does not decrease with the chosen
confidence rule. Keep evaluator models/preprocessing fixed; use a second ASR model
and blinded listening to detect reward overfitting. Freeze sampling settings for a
model-only comparison, and report step/duration/guidance changes in speed experiments.

## Current validation and limits

Tensor shapes and masks are enforced at the module boundaries in `src/dacvae_tts/contracts.py`.
The pre-nano source is available in git history (before commit `9ec06e2`); default Tiny/Small
architecture and loss weighting remain intact.

Automated tests cover masking, prompt preservation, deterministic inference, an
overfit sanity check, distributed batching, exact pretraining resume, and two-rank
pretraining/preference/distillation execution. Both model sizes have run BF16
forward/backward, fused optimizer and sampling checks on the available GPU. The
real DACVAE encoder/decoder has also been exercised.

Eight physical GPUs, corpus-scale I/O, speech quality, WER/CER/DNSMOS improvements,
and voice-cloning success have **not** been measured here. No real pretraining has
started without the dataset location and speaker metadata. A four-million-row corpus
does not by itself specify hours, speaker diversity or transcript quality. These
determine whether tiny/small capacity is sufficient; the implementation makes the
experiments runnable without promising unmeasured results.
