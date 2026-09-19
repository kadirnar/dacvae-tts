# Fast codec and parallel CPU/GPU pipeline

The TTS architecture and training objective are unchanged. DACVAE remains frozen;
training consumes cached continuous posterior-mean latents and cached UTF-8 text.
No discrete audio tokens, speaker-ID inference lookup, or pretrained TTS network
were added. The improvements below concern codec execution and data movement.

## Source review and compatibility

Reviewed [kadirnar/fast-dacvae at revision
406f2e5](https://github.com/kadirnar/fast-dacvae/tree/406f2e5c803927ef18cc9bbe38d715e5417459b9),
especially `dacvae/optimize.py`. The adaptation in `fast_codec.py` uses its ideas
for channels-last convolution, precomputed decoder groups, compilation, and CUDA
graphs. It preserves this project's actual checkpoint loader and preprocessing.
The fork and the official package both export `dacvae`; **do not install both**.
The regular `.[codec]` install remains sufficient. The fork's actual MIT license
is retained in `third_party/fast-dacvae/LICENSE` and included in built packages.
Meta's source and checkpoint retain their existing licenses.

The pinned fork's full optimizer is not a drop-in posterior encoder: it samples
fixed VAE noise, substitutes a polynomial for Snake, removes watermark decoding,
and returns a replay closure bound to its initial input. Its loader also differs
from the official metadata-aware loader. Those behaviors would change this TTS
representation or API. This adapter instead:

- Extracts the **posterior mean** using the official 48 kHz/1920-hop/128-channel checkpoint.
- Folds frozen weight normalization once; new parameters remain frozen.
- Caches the decoder's sequential groups and retains its complete watermark path,
  including fresh messages. Its recurrent watermark component stays eager.
- Uses native 1D convolution by default. `--codec-layout channels_last` converts
  eligible static convolutions to 2D, preserving their weights/geometry and exact
  sine-based Snake. Unsupported padding/layers fail explicitly.
- Offers `--codec-compile` for the encoder and decoder trunk, with dynamic lengths.
  Compilation has startup costs and may recompile for different guards/shapes.
- Offers bounded `--codec-graphs`: after three encounters of a shape, capture it;
  copy each new input into the graph's buffer and return an owned output. At most
  four encoder and four decoder shapes are captured by default. Other shapes run
  normally. Graphs are CUDA-only, process-local, and not safe for concurrent
  requests on the same Codec object. They consume extra memory; evaluate their
  capture cost and reuse rate on real lengths before enabling them broadly.

Encoding and decoding use FP32 with cuDNN TF32 disabled for numerical comparison.
Decoding now disables ambient autocast/TF32 for both backends; older revisions
could inherit those settings. Encoder precision behavior is unchanged.
BF16 cache encoding remains an explicit experimental setting with the previously
measured numerical differences; this change does not promote it to the default.
Backend, layout, compile settings, precision, checkpoint hash and preprocessing
identity are recorded in metadata. Each merged partition retains its runtime
settings. Backend changes do not change cache schemas, token IDs or TTS weights.

`Codec(..., backend="reference")` and `--codec-backend reference` preserve access
to the original execution path. To also retain encoder weight-normalization hooks
in preparation, add `--no-fold-weight-norm`. The fast backend requires folding.
The original repository implementation is available at commit `9372f63`.

## Parallel preprocessing

```bash
source .venv/bin/activate
# Eight independent encoders, each with bounded parallel CPU loading/resampling.
bash scripts/prepare_8gpu.sh /dataset/parquet-directory /cache/english \
  --codec-compile --workers 2 --worker-backend process --worker-threads 1 \
  --prefetch 8 --batch-size 8 --bucket-size 256 --batch-seconds 120

# CPU-only: two independent encoders; no GPU is required.
PREPARE_PROCESSES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  bash scripts/prepare_cpu.sh /dataset/parquet-directory /cache/english-cpu \
  --workers 2 --worker-backend process --worker-threads 1 --prefetch 4
```

Both wrappers default to the fast backend with native convolutions. Compilation
and graphs remain opt-in. `--worker-backend thread` remains the preparation CLI
default and often suits soundfile/SciPy work that releases the GIL. `process` uses
spawned processes for isolation and Python-heavy decoding/preprocessing. Compare
both; multiprocessing adds startup, serialization and IPC costs. `--workers 0`
disables CPU prefetch. `--worker-threads` applies to spawned preparation workers;
thread workers share the codec process's Torch thread pool.

Each codec process writes separate SQLite/binary shards. Results are persisted in
manifest order; reference pairing and speaker splits are independent of worker
completion order. Windows, queued jobs, audio seconds and batch sizes are bounded.
Encoding batches only equal hop-rounded lengths and preserves each waveform's
reflection padding. Merge runs only after every partition succeeds. Empty GPU
visibility is enforced by the CPU wrapper; the eight-GPU wrapper respects an
explicit eight-entry `CUDA_VISIBLE_DEVICES` mapping. JSONL is scanned per rank;
use Parquet files/row groups to avoid repeated full-file scans at 4M rows.

`prepare_parallel.sh` additionally supports `PREPARE_DEVICE=cuda|cpu` and
`PREPARE_PROCESSES=N` for other local topologies. There is one codec replica per
rank. Budget CPU threads across **all** encoder processes and data workers;
increasing every count can slow the job through contention. Launchers default
OMP/MKL to two threads and OpenBLAS to one unless already configured. Compilation
also uses CPU workers; `TORCHINDUCTOR_COMPILE_THREADS=1` can limit startup contention
when compiling eight ranks simultaneously.

## Parallel training

```bash
# Eight GPU ranks, NCCL, sharded/bucketed data and gradient accumulation.
TRAIN_WORKERS=2 TRAIN_PREFETCH=2 \
  bash scripts/train_8gpu.sh configs/tiny.yaml /cache/english/merged runs/tiny

# CPU DDP uses Gloo and FP32. Useful for correctness/small runs; measure scalability.
TRAIN_PROCESSES=2 TRAIN_WORKERS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  bash scripts/train_cpu.sh configs/tiny.yaml /cache/english-cpu/merged runs/tiny-cpu
```

The baseline already had DDP, length buckets, fused CUDA AdamW, BF16, and cached
latents. These changes add safe spawned persistent DataLoader workers, bounded
prefetch, per-worker Torch thread limits, and optional asynchronous CUDA transfers.
SQLite connections and memory maps reopen inside each worker rather than being
pickled. The worker epoch travels in sampler indices, preserving reference pairing
and exact pretraining resume with persistent workers.

The training config/CLI exposes `worker_threads`/`--worker-threads`,
`prefetch_factor`/`--prefetch-factor`, `loader_start_method`/`--loader-start-method`
(`spawn` or `forkserver`), and `cuda_prefetch`/`--[no-]cuda-prefetch`. CUDA prefetch
copies one batch ahead on a separate stream **within an already-counted
accumulation window**. It retains sample/frame weighting, RNG and resume offsets.
It provides overlap when accumulation exceeds one, and adds an extra resident GPU
batch. CPU training uses the ordinary transfer path. The benefit depends on the
actual CPU/I/O/compute bottleneck; no corpus training speedup is asserted here.
Post-training loaders use the same configurable multiprocessing/prefetch options;
preference replay assembly still occurs on the rank's main process.

The GPU training launcher defaults to two data workers per rank (override with
`TRAIN_WORKERS` or `--workers`). Training uses the offline cache, so the codec is
not evaluated inside each optimizer step.

## Inference

```bash
dacvae-tts infer --checkpoint runs/tiny/last.pt --ref-audio reference.wav \
  --text "This is generated from the reference recording." --output output.wav \
  --codec-backend fast --codec-compile --profile
```

No speaker-ID argument is needed. Omitted reference text still uses the existing
optional ASR frontend (`.[asr]`); otherwise pass the complete reference transcript.
The Python API accepts `Synthesizer(..., codec_options={"backend": "fast",
"compile_model": True})`. Evaluation YAML accepts the same `codec_options` mapping.
`--compile` compiles the TTS generator; `--codec-compile` compiles the codec.

## Reproduce verification and measurements

```bash
pytest -q
python scripts/benchmark_fast_codec.py --device cuda --compile --channels-last \
  --output /tmp/codec-gpu.json
python scripts/benchmark_fast_codec.py --device cpu --seconds 1 --batch-size 1 \
  --repeats 3 --output /tmp/codec-cpu.json
python scripts/benchmark_prepare.py --pipeline --codec-backend fast --codec-compile \
  --output /tmp/prepare-fast.json
```

The codec benchmark reports load/probe time, first-call and warmup costs, individual
warm timings, graph reuse counts, and numeric differences. GPU tests use four
3-second synthetic inputs per encode and one 3-second decode, with the full
watermark decoder and CPU/GPU transfers. CPU tests use one 1-second input.
Fixed latent inputs and fixed watermark seeds isolate decoder differences;
changed inputs and short shape fallbacks exercise graph correctness. These are
synthetic codec measurements, not speech-quality or TTS speed measurements.

Results and integration evidence are recorded below. Eight physical GPUs, a real
corpus, held-out speech metrics and voice-cloning quality have not been evaluated.
Do not transfer the fork's H100 benchmark claims to this different, exact computation.

### Observed results (2026-09-19)

Environment: one RTX 5070 Ti, Ryzen 5 5600 (12 logical CPUs), PyTorch 2.8.0/CUDA
12.8, two Torch threads. GPU medians over 12 warm calls are in
[the raw codec report](fast-codec-gpu-benchmark.json). The reference comparison
already folds encoder weight normalization, matching the previous optimized
preparation implementation. Times include transfers; decoder times retain watermarking.

| Backend, strict FP32 | Encode batch (ms) | Decode one waveform (ms) |
|---|---:|---:|
| Reference | 52.21 | 45.27 |
| Fast, native eager | 52.27 | 45.48 |
| Fast, native graphs | 52.27 | 44.75 |
| Fast, native compiled | 44.91 | 42.63 |
| Fast, native compiled + graphs | 44.61 | 42.14 |
| Fast, channels-last eager | 850.58 | 541.31 |

Compilation plus graphs measured **1.17× encoder and 1.07× decoder throughput** on
this workload. Graphs alone and native eager execution did not demonstrate a
meaningful GPU gain. Channels-last was rejected as the default. Compilation's
first calls took 2.51 s for encoding and 1.17 s for decoding in this run; these are
not fully cold compiler-cache measurements. The later combined variant reused
compiler artifacts. New installations and eight simultaneous compilations may
have substantially different startup costs.

[CPU results](fast-codec-cpu-benchmark.json), three warm repetitions with one
1-second input: encoding 326.73 → 325.29 ms (no material gain), complete decoding
1018.73 → 883.65 ms (1.15×). This is a single-process codec comparison, not a
measurement of CPU process scaling. Multiprocess behavior was checked separately.

[The mixed-length preparation report](fast-codec-preparation-benchmark.json)
uses 32 synthetic clips totaling 120.3 s and three alternating repetitions.
Serial, unfused single-recording encoding took 0.663 s; compiled/batched encoding
took 0.436 s (1.52×). WAV loading/resampling, encoding and cache writes took
0.667 → 0.466 s (1.43×), **excluding setup and merge**. This comparison combines
previous batching improvements with the new backend; it does not isolate the
incremental benefit of this commit. Full wall times, including setup, are retained
in the report. It uses CPU threads; process workers are correctness-tested and
configurable, not claimed faster without a corpus benchmark.

Numerical qualification matters: the fixed 3-second codec test had compiled
latent/waveform relative RMS differences of about 1.3e-6/2.1e-6. The broader
mixed-length tone test had relative latent RMS error 3.79e-6 and maximum absolute
error 2.57e-4. **One of 388,096 values failed the existing per-element
`atol=rtol=2e-4` gate.** The original strict benchmark run failed. Its recorded
throughput run explicitly used `--report-only` to preserve the failure and allow
diagnostic measurements; the gate was not relaxed. Add `--report-only` to the
preparation benchmark command above to reproduce that diagnostic report. These
are small numerical differences, but they are not proof of unchanged speech
quality. Compilation remains opt-in until held-out speech reconstruction and
cache comparisons establish acceptable tolerances.

### Implemented and tested

The full suite passes **95 tests**, including CPU-launcher success/failure and
graph precision separation. Tests cover convolution geometry, no conversion-time RNG consumption, exact Snake
at large amplitudes, frozen weights, bounded fresh-input graph replay, shape
fallback, ordered thread/process preprocessing, reopening SQLite/mmap state under
spawn, CUDA transfer equivalence, two-rank DDP, and exact same-topology checkpoint
resume with two spawned data workers. The eight-GPU launcher is exercised using
mock workers, including explicit GPU mapping and failed-worker cleanup.

[Integration commands and outcomes](parallel-integration.json) record a separate
real-codec check using six synthetic tones:

- CPU preparation with two codec processes and two spawned data workers per rank;
  successful merge of all six records.
- Two-rank CPU/Gloo training with two data workers per rank.
- GPU training with two data workers, accumulation two, and CUDA prefetch on/off:
  **bitwise-identical model weights after two optimizer updates**.
- GPU preparation with the compiled fast codec, CUDA graphs and CPU process workers.
- Reference-audio inference using the resulting toy checkpoint and compiled codec.

To recreate the source fixture used by those commands, first run
`python scripts/smoke_pipeline.py --output /tmp/dacvae-fast-prepare-smoke-20260919`
with a fresh output path. Substitute fresh paths in the integration report before
rerunning its commands. The toy inference produces test signals from an initialized
model; it is not a speech demonstration or evidence of learnability. No expensive
training job or user dataset transformation was launched.

Next: run paired codec reconstruction on a representative English speech subset
with `--codec-backend reference`, fast native eager, and fast native compiled.
Inspect the numerical outlier distribution, reconstruction WER/CER, speaker
similarity and boundary artifacts before choosing a corpus cache backend. Then
measure end-to-end preparation and training throughput on the actual eight-GPU
host with controlled worker counts and identical data/steps.
