# Offline audio and text preparation

The `prepare` command already cached posterior-mean DACVAE latents before TTS
training. Previously, it decoded and encoded one recording at a time, and training
normalized/tokenized transcripts on every sampled pair. The optimized path caches
both audio latents and normalized UTF-8 text bytes. DACVAE audio latents remain
continuous, not discrete audio-token IDs.

## Implemented changes

- One process per GPU; four configurable CPU decode/resample threads per process.
  A bounded queue overlaps CPU loading with GPU work, without loading the corpus
  into memory. Required speaker labels and full-utterance transcripts are unchanged.
- A bounded lookahead window groups equal hop-rounded lengths into encoder batches.
  Each waveform receives the original reflection padding before concatenation.
  Arbitrarily padding unequal lengths would change internal convolution boundaries.
- Pinned host transfers and one latent transfer back to CPU per batch. Frozen
  encoder weight normalization is materialized once. The decoder stays off GPU.
- GPU batch limits use both recording count and total padded audio seconds.
  CUDA OOM retries bisected batches; singleton OOM or other codec failures abort.
- JSONL ranks still scan all lines, but parse only their own rows: eight ranks parse
  each record once in total, instead of eight times. Directory/Parquet partitioning
  remains available and avoids redundant full JSONL scans.
- SQLite stores normalized text bytes once. Training adds the same byte offset,
  BOS/SEP/EOS tokens, and segment IDs; tokenizer/checkpoint vocabulary is unchanged.
  Legacy caches without the optional table use text normalization/tokenization.
- Source order is restored after bucketing so retained row IDs, duplicate selection,
  and same-speaker pairing do not depend on loading-thread completion order.
- Train-only statistics use the actual stored float16 values. Corrupt/missing audio
  is logged as a rejected row; codec errors are not mistaken for dataset errors.
- `metadata.json` records encoding options, stage timings, throughput, forward calls,
  and OOM retries. Its completion marker is written after index and statistics.
  Failed caches cannot be resumed in place; use a new output directory.

## Run on eight GPUs

```bash
source .venv/bin/activate
bash scripts/prepare_8gpu.sh /dataset/parquet-directory /cache/english \
  --workers 4 --prefetch 16 --batch-size 8 --bucket-size 256 \
  --batch-seconds 120 --precision fp32
```

These limits are per GPU, not per eight-GPU job. Adjust CPU thread counts for the
available cores and storage bandwidth. `--bucket-size` bounds decoded waveform RAM;
it does not denote the number of input files or the complete dataset size.
Fewer matching lengths within a window means fewer batching opportunities.
The launcher respects eight explicitly assigned `CUDA_VISIBLE_DEVICES` entries,
including UUIDs, and cleans up its workers if interrupted or a waited worker fails.

For a single-GPU serial comparison with the new cache format:

```bash
dacvae-tts prepare --manifest /dataset/manifest.jsonl --output /cache/serial \
  --workers 0 --batch-size 1 --bucket-size 1 --precision fp32 --no-fold-weight-norm
dacvae-tts prepare --manifest /dataset/manifest.jsonl --output /cache/optimized \
  --workers 4 --batch-size 8 --bucket-size 256 --precision fp32
```

Use separate output directories. Inspect each partition's `metadata.json` and
`rejected.jsonl`; merge still rejects speaker leakage and conflicting duplicates.

## Precision and compatibility

FP32 preparation and inference reference encoding explicitly disable cuDNN TF32
convolution, using the same posterior-mean path. The old encoder
inherited PyTorch's ambient TF32 setting, which was enabled in the tested environment.
Real-codec tests found substantially different latents between batch sizes with
TF32 enabled. The speedups below compare **strict FP32 against strict FP32**, not
the old implicit-TF32 configuration. They are not bitwise-equivalence claims.

The optional `--precision bf16` changes cached values and remains experimental.
Merge rejects mixed FP32/BF16 partitions. The synthetic test measured approximately
5.4% relative latent RMS error versus FP32; no real-speech reconstruction or downstream
WER/CER/DNSMOS tolerance has been established. Keep FP32 for the documented baseline.

## Measurements, 19 September 2026

Hardware: one RTX 5070 Ti. PyTorch 2.8.0+cu128. Real frozen
`facebook/dacvae-watermarked`, 48 kHz, 128 channels, 25 frames/second.
Input: 32 deterministic synthetic tone/noise recordings, approximately 120.3 seconds
total, with odd lengths spanning four hop-rounded buckets. Three repetitions,
alternating serial/optimized measurement order, with warmed encoder shapes.

| Measurement | Serial | Optimized | Ratio |
|---|---:|---:|---:|
| Median encoder + transfer time | 0.651 s | 0.503 s | 1.29x |
| Encoder calls | 32 | 5 | — |
| Median WAV-to-cache processing, excluding model setup and merge | 0.654 s | 0.536 s | 1.22x |
| Median total WAV-to-cache job, including model setup, excluding merge | 1.934 s | 1.758 s | 1.10x |

FP32 maximum absolute latent error: 0.0001641; relative latent RMS error:
0.000003731. All output shapes matched. Elementwise comparison passed with
`atol=2e-4, rtol=2e-4`. FP16 cache rounding may amplify individual small differences;
byte-for-byte cache identity is not promised. Frozen weight-normalization folding
alone matched the original single-recording encoder exactly in the diagnostic run.

The BF16 encoder benchmark was approximately 2.19x faster than the strict-FP32
serial encoder, but its maximum latent absolute error was approximately 1.37 and
relative RMS error 0.0543. This is a speed/precision experiment, not a quality result.

Full raw measurements: [FP32 encoder and pipeline](preparation-benchmark.json),
[BF16 encoder experiment](preparation-benchmark-bf16.json).

These small, synthetic runs are not evidence of four-million-row throughput, eight
physical GPU scaling, speech quality, or performance on RTX 4090/A100. The toy length
distribution favors batching. Disk, network storage, transcript lengths, real audio
duration distribution, and host contention may substantially change the result.
The benchmark excludes merge; no merge performance improvement is claimed.

## Reproduce and validate

```bash
python scripts/benchmark_prepare.py --device cuda --pipeline \
  --output /tmp/dacvae-prepare-fp32.json
python scripts/benchmark_prepare.py --device cuda --precision bf16 \
  --output /tmp/dacvae-prepare-bf16.json
pytest -q
ruff check src tests scripts
bash -n scripts/prepare_8gpu.sh
```

The benchmark uses temporary synthetic WAVs and removes its temporary caches.
Model loading requires the codec dependency/checkpoint; no user corpus is touched.
Focused tests cover odd lengths, matching the single encoder, bounded prefetch,
eight-way source coverage, embedded Parquet audio, stereo/resampling, OOM retries,
fatal codec errors, rejected rows, train-only statistics, token identity, legacy
caches, and FP32/BF16 merge rejection.

Validation: 79 tests passed; Ruff, shell syntax, and diff whitespace checks passed.
The real-codec synthetic integration test also completed prepare, merge, cache audit,
codec reconstruction, two toy training updates, case export, and inference using
both predicted and ground-truth durations. This establishes execution, not speech
quality. Launcher tests use mock workers; only one physical GPU was available.
