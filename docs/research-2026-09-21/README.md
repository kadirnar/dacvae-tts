# Research evidence — 21 September 2026

Raw results and the scripts behind [`../iyilestirme-yol-haritasi.md`](../iyilestirme-yol-haritasi.md).
Nothing here is imported by the package, and no model code was changed to produce it.

Hardware/software: one RTX 5070 Ti (16 GB), Ryzen 5 5600, Python 3.12, PyTorch 2.8.0 + CUDA 12.8, BF16
autocast unless noted. Speech: Hugging Face `openslr/librispeech_asr` Parquet export (`test.clean` and the first
`train.clean.100` file). That audio is 16 kHz, resampled to 48 kHz by the repository's `read_audio`; statistics
for genuinely full-band speech may differ. Timings come from a single desktop GPU, not the eight-GPU target.

The scripts were run from a scratch directory with absolute paths; the copies here take paths as arguments
and import the installed `dacvae_tts` package. Run them from the repository root (they read `configs/*.yaml`).

| Script | What it measures | Result file |
|---|---|---|
| `scripts/probe_latents.py OUT_DIR PARQUET` | Per-channel posterior statistics, PCA spectrum, temporal correlation, decoder sensitivity | `results/probe_latents.json` |
| `scripts/probe_gain.py PARQUET OUT_JSON` | Latent and round-trip sensitivity to input level | `results/probe_gain.json` |
| `scripts/sim_batching.py OUT_DIR` | Padding waste of `BucketBatchSampler` on a synthetic corpus | `results/sim_batching.json` (+ `sim_batching_librispeech.json`: same comparison on the real cache) |
| `scripts/bench_model.py OUT_DIR` | CFG batching, launch-bound check, duplicated condition encoding | `results/bench_model.json` |
| `scripts/bench_scaling.py OUT_DIR` | Training throughput vs micro-batch, packing, `torch.compile` | `results/bench_scaling.json` |
| `scripts/bench_loader.py --cache CACHE --out JSON` | Upper bound of the current DataLoader path | `results/bench_loader.json` |
| `scripts/make_librispeech_manifest.py ALL_DIR OUT_DIR` | Manifest with explicit speaker-disjoint splits for the A/B runs | – |
| `scripts/ab_train.py --cache CACHE --out JSON --variants ...` | Small controlled trainings: packing, v / x / EDM parameterisation, time sampler | `results/ab_round1.json` … `ab_round4_long.json` |
| `scripts/ab_train.py ... --optimizer muon [--lr 1e-3]` | Same protocol with the package's Muon optimizer (added 22 September); `--optimizer adamw` is the default and reproduces rounds 1–4 | `results/ab_round6_*.json` |
| `dacvae-tts prepare ... --bucket-size {256,2048,8192}` | Encoder throughput of the repository's own preparation command | `results/prepare_throughput.json` |
| `scripts/ab_larope.py --cache CACHE --out JSON --variants ...` | Length-aware RoPE in cross-attention and text self-attention (scratch subclasses of `FlowTTS`); tracks text/speaker usage over training | `results/ab_round5_*.json` |

## A/B protocol

`ab_train.py` reuses `LatentDataset`, `BucketBatchSampler`, `collate` and `FlowTTS` unchanged. Only the loss
wrapper differs per variant. Train: 3,416 utterances / 51 speakers / 8.06 h; validation: 413 utterances /
8 held-out speakers / 0.76 h. AdamW 3e-4 (Small: 2e-4), betas (0.9, 0.95), weight decay 0.01, 300-step warm-up then
constant, batch 32 pairs, BF16, seed 42, condition dropout 0.1, duration loss weight 0.1.

Metric: flow error on the held-out speakers at a fixed grid t = 0.05 … 0.95 with fixed noise, reported as
velocity MSE and as the implied clean-latent MSE (`x̂ = x_t + (1 − t)·v̂`). All parameterisations are converted
to both quantities, so rows are comparable. `base` reproduces bit-for-bit across runs (1.4973).

The conditioning diagnostic compares the loss at t ∈ {0.25, 0.4, 0.55, 0.7} with the correct conditions
against zeroed text / no reference, and (rounds 3–4) against *wrong* text and a *wrong* speaker prompt taken from
another batch item. The shuffled variants are the cleaner signal: zeroed text is an input combination the
model never sees in training.

**These are learnability probes.** A few thousand updates on 8.8 hours cannot produce intelligible speech;
nothing here is a WER, similarity or naturalness result.
