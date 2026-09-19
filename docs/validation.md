# Implementation validation — 18 September 2026

These results validate software execution, not speech quality.

## Automated suite

`pytest -q`: **18 passed**. `ruff check src tests`: passed.
Both launch scripts passed `bash -n`. `uv pip check` found no dependency conflicts
after the evaluator protobuf constraint was added.

Coverage includes:

- All trainable parameters receive finite gradients, including checkpointed blocks.
- A fixed synthetic flow-matching problem can be overfit.
- Padding does not change valid outputs, including partial two-frame patches.
- Sampling preserves reference frames, zeroes padding and repeats exactly for a seed.
- Eight-rank bucket plans have equal step counts, no overlapping samples and respect
  the configured padded-frame limit.
- A pretraining run interrupted and resumed at an optimizer boundary matches the
  uninterrupted run's model and EMA tensors exactly on CPU.
- Two-process Gloo pretraining, preference learning and distillation finish and save
  loadable checkpoints.
- Preferences have the intended gradient sign; frozen reference parameters receive
  no gradients; validation examples cannot enter preference selection.
- Corpus WER aggregates edit counts correctly, including insertions above 100%.
- Parquet row-group/row partitioning covers each row once across eight partitions;
  generated IDs remain unique across files.
- Cache merge deduplicates repeated partitions and recalculates train statistics.
- Paired comparison aligns examples and detects an all-metric improvement in a
  controlled fixture.

## GPU and external-model checks

Environment: Python 3.12, PyTorch/torchaudio 2.8.0+cu128, one RTX 5070 Ti.
Tiny and small each completed BF16 forward/backward, a fused AdamW update, and
conditional sampling with finite outputs. Small used activation checkpointing.

The real frozen `facebook/dacvae-watermarked` encoder and decoder ran on GPU.
Observed properties: 48,000 Hz, hop 1,920 samples, 128 posterior channels,
107,671,171 parameters. A one-second input produced 25 frames and decoded to
48,000 finite samples.

An isolated temporary six-row sine-wave fixture exercised preparation, merge,
train-only normalization, a two-update BF16 training run, validation, checkpoint
loading, reference-conditioned inference, candidate generation and trajectory
export. This fixture was deliberately synthetic and was not added to the training
corpus or presented as speech-model training.

The resulting test waveforms also passed through actual faster-whisper `tiny.en`,
Microsoft's standard DNSMOS ONNX model, and `microsoft/wavlm-base-plus-sv` on CPU.
All evaluators returned finite outputs. Their scores on these synthetic signals
are not quality evidence and are intentionally not reported as model results.

## Not established

No eight-physical-GPU run, multi-node run, corpus-scale throughput benchmark,
trained speech sample, cloning success rate, perceptual listening result, or
WER/CER/DNSMOS improvement has been established. The user's dataset path and speaker
metadata are still missing. Optional compilation and the default large-v3 ASR
checkpoint have not been benchmarked. Post-training remains experimental and
requires validation on real held-out speech before any quality claim.
