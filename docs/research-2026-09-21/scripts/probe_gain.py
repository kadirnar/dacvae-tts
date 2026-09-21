"""How do DACVAE latents react to input level? (loudness-normalisation probe)"""

import io
import json
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
import torchaudio

from dacvae_tts.codec import Codec, read_audio

codec = Codec("facebook/dacvae-watermarked", "cuda")
path = sys.argv[1]  # LibriSpeech parquet; results are written to sys.argv[2]
rows = pq.ParquetFile(path).read_row_group(0).to_pylist()[:24]
mel = torchaudio.transforms.MelSpectrogram(48000, n_fft=2048, hop_length=512, n_mels=128, f_max=12000)


def logmel(wave):
    return mel(wave).clamp_min(1e-5).log()


out = {g: {"cos": [], "norm_ratio": [], "rel_change": [], "roundtrip_mel": []} for g in (-18, -12, -6, 6, 12)}
rms_db = []
for row in rows:
    audio = read_audio(io.BytesIO(row["audio"]["bytes"]), 48000)
    rms_db.append(float(20 * torch.log10(audio.square().mean().sqrt())))
    z0 = codec.encode(audio).cpu()
    base = codec.decode(z0)
    ref = distance0 = float((logmel(base[: len(audio)]) - logmel(audio)).abs().mean())
    for g in out:
        scale = 10 ** (g / 20)
        if float(audio.abs().max()) * scale > 1.0:
            continue
        z = codec.encode(audio * scale).cpu()
        out[g]["cos"].append(float(torch.nn.functional.cosine_similarity(z.flatten(), z0.flatten(), dim=0)))
        out[g]["norm_ratio"].append(float(z.norm() / z0.norm()))
        out[g]["rel_change"].append(float((z - z0).norm() / z0.norm()))
        decoded = codec.decode(z)[: len(audio)] / scale  # undo gain before comparing
        out[g]["roundtrip_mel"].append(float((logmel(decoded) - logmel(audio)).abs().mean()) / ref)
summary = {
    "input_rms_dbfs_min_median_max": [float(np.min(rms_db)), float(np.median(rms_db)), float(np.max(rms_db))],
    "gain_db": {
        g: {k: float(np.mean(v)) for k, v in d.items()} | {"n": len(d["cos"])} for g, d in out.items()
    },
}
print(json.dumps(summary, indent=1))
json.dump(summary, open(sys.argv[2], "w"), indent=1)
