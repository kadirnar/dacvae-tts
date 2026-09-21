"""Probe frozen-DACVAE latents on real English speech (read-only w.r.t. the repo).

Measures, per channel: spread of posterior means, posterior std, SNR, KL, decoder
sensitivity; plus PCA spectra, temporal correlation and truncation/noise decode tests.
"""

import io
import json
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torchaudio

from dacvae_tts.codec import Codec, read_audio

# usage: python probe_latents.py OUT_DIR LIBRISPEECH_PARQUET [CODEC_CHECKPOINT]
OUT = sys.argv[1]
DEVICE = "cuda"
torch.manual_seed(0)
np.random.seed(0)

parquet = sys.argv[2]  # e.g. openslr/librispeech_asr all/test.clean/0000.parquet
table = pq.ParquetFile(parquet)
print("schema:", table.schema_arrow.names, "rows:", table.metadata.num_rows, flush=True)

# Up to 8 utterances from each speaker, 3-15 s.
per_speaker, rows = {}, []
for batch in table.iter_batches(batch_size=64):
    for row in batch.to_pylist():
        speaker = row["speaker_id"]
        if per_speaker.get(speaker, 0) >= 8:
            continue
        per_speaker[speaker] = per_speaker.get(speaker, 0) + 1
        rows.append(row)
print("selected", len(rows), "utterances from", len(per_speaker), "speakers", flush=True)

codec = Codec(sys.argv[3] if len(sys.argv) > 3 else "facebook/dacvae-watermarked", DEVICE)
model = codec.model
print("codec:", codec.sample_rate, codec.hop_length, codec.latent_dim, flush=True)


@torch.inference_mode()
def posterior(audio):
    x = model._pad(audio.to(DEVICE).reshape(1, 1, -1))
    with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=False, allow_tf32=False):
        mean, scale = model.quantizer.in_proj(model.encoder(x)).chunk(2, dim=1)
    std = torch.nn.functional.softplus(scale) + 1e-4
    return mean[0].T.float().cpu(), std[0].T.float().cpu()


means, stds, utterances = [], [], []
started = time.time()
for row in rows:
    audio = read_audio(io.BytesIO(row["audio"]["bytes"]), codec.sample_rate)
    seconds = len(audio) / codec.sample_rate
    if not 3 <= seconds <= 15:
        continue
    mu, sd = posterior(audio)
    means.append(mu)
    stds.append(sd)
    utterances.append({"text": row["text"], "audio": audio, "speaker": row["speaker_id"]})
print(f"encoded {len(means)} utterances in {time.time() - started:.1f}s", flush=True)

M = torch.cat(means).double()  # [frames, C]
S = torch.cat(stds).double()
frames, channels = M.shape
report = {"utterances": len(means), "frames": frames, "hours": frames / 25 / 3600}

ch_mean, ch_std = M.mean(0), M.std(0)
post_var = S.square().mean(0)
snr = ch_std.square() / post_var
kl = 0.5 * (M.square() + S.square() - S.square().log() - 1).mean(0)  # nats / frame / channel
report["channel"] = {
    "mean_abs_max": float(ch_mean.abs().max()),
    "std_min": float(ch_std.min()),
    "std_median": float(ch_std.median()),
    "std_max": float(ch_std.max()),
    "posterior_std_median": float(S.mean(0).median()),
    "posterior_std_min": float(S.mean(0).min()),
    "posterior_std_max": float(S.mean(0).max()),
    "snr_quantiles_min_10_50_90_max": [float(q) for q in np.quantile(snr.numpy(), [0, 0.1, 0.5, 0.9, 1])],
    "channels_snr_below_1": int((snr < 1).sum()),
    "channels_snr_below_4": int((snr < 4).sum()),
    "channels_snr_above_25": int((snr > 25).sum()),
    "kl_total_nats_per_frame": float(kl.sum()),
    "kl_quantiles_min_10_50_90_max": [float(q) for q in np.quantile(kl.numpy(), [0, 0.1, 0.5, 0.9, 1])],
    "total_raw_variance": float(ch_std.square().sum()),
}


def spectrum(X):
    X = X - X.mean(0)
    cov = X.T @ X / (len(X) - 1)
    eig, vec = torch.linalg.eigh(cov)
    eig, vec = eig.flip(0).clamp_min(0), vec.flip(1)
    cumulative = eig.cumsum(0) / eig.sum()
    return eig, vec, cumulative


eig_raw, vec_raw, cum_raw = spectrum(M)
Z = (M - ch_mean) / ch_std  # exactly what the TTS model sees
eig_norm, vec_norm, cum_norm = spectrum(Z)
ks = [4, 8, 16, 24, 32, 48, 64, 96, 128]
report["pca"] = {
    "raw_cumulative_variance": {k: float(cum_raw[k - 1]) for k in ks},
    "normalized_cumulative_variance": {k: float(cum_norm[k - 1]) for k in ks},
    "raw_participation_ratio": float(eig_raw.sum().square() / eig_raw.square().sum()),
    "normalized_participation_ratio": float(eig_norm.sum().square() / eig_norm.square().sum()),
    "raw_components_for_90_95_99": [int((cum_raw < q).sum()) + 1 for q in (0.9, 0.95, 0.99)],
    "normalized_components_for_90_95_99": [int((cum_norm < q).sum()) + 1 for q in (0.9, 0.95, 0.99)],
}

# Temporal structure: how redundant are adjacent frames (relevant to 2-frame packing)?
lag = []
for mu in means:
    z = (mu.double() - ch_mean) / ch_std
    a, b = z[:-1], z[1:]
    lag.append(((a * b).mean(0) - a.mean(0) * b.mean(0)) / (a.std(0) * b.std(0) + 1e-9))
lag = torch.stack(lag).mean(0)
report["temporal"] = {
    "lag1_autocorr_quantiles_min_10_50_90_max": [
        float(q) for q in np.quantile(lag.numpy(), [0, 0.1, 0.5, 0.9, 1])
    ],
    "lag1_autocorr_mean": float(lag.mean()),
}

# ---- Decoder experiments -------------------------------------------------------------
mel = torchaudio.transforms.MelSpectrogram(
    codec.sample_rate, n_fft=2048, hop_length=512, n_mels=128, f_max=12000
).to(DEVICE)


def logmel(wave):
    return mel(wave.to(DEVICE)).clamp_min(1e-5).log()


def distance(a, b):
    n = min(a.numel(), b.numel())
    return float((logmel(a[:n]) - logmel(b[:n])).abs().mean())


subset = list(range(0, len(means), max(1, len(means) // 32)))[:32]
base = {i: codec.decode(means[i]) for i in subset}
ch_std32, ch_mean32 = ch_std.float(), ch_mean.float()
basis = vec_raw.float()

decode_report = {}
# (1) original -> codec reconstruction, for scale.
decode_report["original_vs_reconstruction"] = float(
    np.mean([distance(utterances[i]["audio"], base[i]) for i in subset])
)
# (2) the VAE's own sampling noise (what the decoder was trained to tolerate).
decode_report["posterior_sample_vs_mean"] = float(
    np.mean([distance(codec.decode(means[i] + stds[i] * torch.randn_like(stds[i])), base[i]) for i in subset])
)
# (3) PCA truncation in raw latent space.
decode_report["pca_truncation"] = {}
truncated_audio = {}
for k in (8, 16, 32, 48, 64, 96):
    P = basis[:, :k]
    values = []
    for i in subset:
        centered = means[i] - ch_mean32
        wave = codec.decode(centered @ P @ P.T + ch_mean32)
        truncated_audio[(k, i)] = wave
        values.append(distance(wave, base[i]))
    decode_report["pca_truncation"][k] = float(np.mean(values))
# (4) isotropic error in the NORMALIZED space (the space of the MSE training loss).
decode_report["normalized_isotropic_noise"] = {}
noisy_audio = {}
for beta in (0.05, 0.1, 0.2, 0.3, 0.5):
    values = []
    for i in subset:
        wave = codec.decode(means[i] + beta * ch_std32 * torch.randn_like(means[i]))
        noisy_audio[(beta, i)] = wave
        values.append(distance(wave, base[i]))
    decode_report["normalized_isotropic_noise"][beta] = float(np.mean(values))
# (5) same total normalized-space error energy, but only on low- or high-SNR channels.
order = snr.argsort()
low, high = order[: channels // 2], order[channels // 2 :]
decode_report["half_channel_noise_beta_0.3"] = {}
for name, index in (("lowest_snr_half", low), ("highest_snr_half", high)):
    values = []
    for i in subset:
        noise = torch.zeros_like(means[i])
        noise[:, index] = 0.3 * ch_std32[index] * torch.randn(len(means[i]), len(index))
        values.append(distance(codec.decode(means[i] + noise), base[i]))
    decode_report["half_channel_noise_beta_0.3"][name] = float(np.mean(values))
# (6) per-channel sensitivity: one normalized std of noise on a single channel.
probe_ids = subset[:6]
sensitivity = torch.zeros(channels)
for c in range(channels):
    values = []
    for i in probe_ids:
        noise = torch.zeros_like(means[i])
        noise[:, c] = ch_std32[c] * torch.randn(len(means[i]))
        values.append(distance(codec.decode(means[i] + noise), base[i]))
    sensitivity[c] = float(np.mean(values))
rank_corr = float(
    np.corrcoef(np.argsort(np.argsort(snr.numpy())), np.argsort(np.argsort(sensitivity.numpy())))[0, 1]
)
decode_report["per_channel_sensitivity"] = {
    "quantiles_min_10_50_90_max": [float(q) for q in np.quantile(sensitivity.numpy(), [0, 0.1, 0.5, 0.9, 1])],
    "spearman_with_snr": rank_corr,
    "share_of_total_sensitivity_in_top_32_channels": float(
        sensitivity.sort(descending=True).values[:32].sum() / sensitivity.sum()
    ),
    "share_of_total_sensitivity_in_bottom_64_channels": float(
        sensitivity.sort().values[:64].sum() / sensitivity.sum()
    ),
}
report["decoder"] = decode_report
report["per_channel_table"] = [
    {
        "channel": c,
        "mean": float(ch_mean[c]),
        "std": float(ch_std[c]),
        "posterior_std": float(S[:, c].mean()),
        "snr": float(snr[c]),
        "kl": float(kl[c]),
        "lag1": float(lag[c]),
        "sensitivity": float(sensitivity[c]),
    }
    for c in range(channels)
]

torch.save(
    {
        "subset": subset,
        "texts": [utterances[i]["text"] for i in subset],
        "base": base,
        "truncated": truncated_audio,
        "noisy": noisy_audio,
        "sample_rate": codec.sample_rate,
    },
    OUT + "/probe_audio.pt",
)
with open(OUT + "/probe_latents.json", "w") as stream:
    json.dump(report, stream, indent=1)
summary = {k: v for k, v in report.items() if k != "per_channel_table"}
print(json.dumps(summary, indent=1))
