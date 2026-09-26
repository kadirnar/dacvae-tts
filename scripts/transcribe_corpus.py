"""Re-transcribe every corpus clip with Whisper and score DNSMOS, for transcript/quality-based filtering.

Writes OUTPUT/scores.jsonl with one row per clip: uid (matching the latent cache: the row's explicit `id` when it has
one, as the scripts/data/ manifests do, else "data/train-XXXXX.parquet:<row>"), given text, Whisper hypothesis,
Turkish-normalized WER/CER, duration, quality_score, DNSMOS SIG/BAK/OVRL.
Raon-OpenTTS / Emilia practice: cut the worst ~15% CER tail and DNSMOS OVRL < ~2.8-3.0 before the final training stage.

  python scripts/transcribe_corpus.py --raw data/raw/data --output outputs/corpus-scores --device cuda --dnsmos models/sig_bak_ovr.onnx
  python scripts/transcribe_corpus.py --raw data/sources/cv-tr/manifest --output outputs/scores-cv-tr --device cuda \
      --dnsmos models/sig_bak_ovr.onnx      # a scripts/data/ manifest
"""

import argparse
import io
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

from dacvae_tts.metrics import DNSMOS, TorchDNSMOS, error_counts


def row_uid(name, index, row):
    """The uid `prepare` stores: the explicit `id` column if present, else "<data/file>:<row>" (HF shards)."""
    return str(row["id"]) if row.get("id") is not None else f"{name}:{index}"


def row_audio(row, root):
    """Embedded bytes ({"bytes": ...}), or a path relative to the manifest directory, as `prepare` accepts."""
    audio = row["audio"]
    if isinstance(audio, dict):
        if audio.get("bytes"):
            return audio["bytes"]
        audio = audio["path"]
    path = Path(audio)
    return (path if path.is_absolute() else root / path).read_bytes()


def decode(item):
    uid, blob = item
    try:
        audio, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
        audio = audio.mean(1)
        if sr != 16000:
            import math

            g = math.gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
        return uid, audio, None
    except Exception as error:  # corrupt mp3 frames etc.
        return uid, None, str(error)


_dnsmos = None


def dnsmos_worker(args):
    global _dnsmos
    path, uid, audio = args
    if _dnsmos is None:
        _dnsmos = DNSMOS(path)
    try:
        return uid, _dnsmos(audio)
    except Exception as error:
        return uid, {"dnsmos_error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True, help="Directory with train-XXXXX-of-XXXXX.parquet shards")
    parser.add_argument("--output", required=True)
    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default="tr")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dnsmos", help="Path to sig_bak_ovr.onnx")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Rows per shard (0 = all), for smoke tests")
    parser.add_argument(
        "--metric-normalization", choices=["turkish-v1", "turkish-v2"], default="turkish-v1",
        help="Normalization of the WER/CER written per clip (turkish-v2 for caches prepared with turkish-v2)",
    )
    parser.add_argument("--chunk-rows", type=int, default=2000, help="Rows read (and held in RAM) at a time")
    parser.add_argument(
        "--dnsmos-device", default="cpu",
        help="cpu: onnxruntime worker pool (the official runtime); cuda: the same model converted to PyTorch "
             "(TorchDNSMOS, matches onnxruntime to ~1e-5 MOS) and batched on the GPU, ~100x faster",
    )
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "scores.jsonl"
    done = set()
    if log.exists():
        done = {json.loads(line)["uid"] for line in log.read_text().splitlines() if line.strip()}
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    # HF Whisper with true cross-clip batching: every clip is <= 30 s, i.e. exactly one Whisper window, so a batch of
    # 32 clips is one generate() call. faster-whisper's batched pipeline only batches VAD segments *within* a clip.
    hf_name = args.asr_model if "/" in args.asr_model else f"openai/whisper-{args.asr_model}"
    processor = WhisperProcessor.from_pretrained(hf_name)
    model = WhisperForConditionalGeneration.from_pretrained(hf_name, torch_dtype=torch.float16).to(args.device).eval()
    pool = mp.get_context("spawn").Pool(args.workers)
    gpu_dnsmos = TorchDNSMOS(args.dnsmos, args.dnsmos_device) if args.dnsmos and args.dnsmos_device != "cpu" else None
    files = sorted(Path(args.raw).glob("*.parquet"))
    started = time.time()
    total = 0

    def transcribe(audios):
        # Log-mel on the scoring device: numpy features cost ~1 s per 48 clips on a busy CPU (more than decoding).
        features = processor(audios, sampling_rate=16000, return_tensors="pt", device=args.device).input_features
        with torch.inference_mode():
            ids = model.generate(
                features.to(args.device, torch.float16), language=args.language, task="transcribe",
                num_beams=1, max_new_tokens=220,
            )
        return processor.batch_decode(ids, skip_special_tokens=True)

    def row_chunks(file):
        """(first row index, rows) over groups of row groups: a 4.7 GB shard read whole took ~25 GB of RAM."""
        source = pq.ParquetFile(file)
        offset, groups = 0, max(1, args.chunk_rows // max(1, source.metadata.row_group(0).num_rows))
        for first in range(0, source.num_row_groups, groups):
            rows = source.read_row_groups(list(range(first, min(first + groups, source.num_row_groups)))).to_pylist()
            yield offset, rows
            offset += len(rows)

    def submit(name, file, base, rows):
        """Start decoding one chunk in the worker pool; the GPU transcribes the previous chunk meanwhile."""
        if args.limit:
            if base >= args.limit:
                return None
            rows = rows[: args.limit - base]
        meta = {row_uid(name, base + i, r): r for i, r in enumerate(rows)}
        items = [(uid, row_audio(r, file.parent)) for uid, r in meta.items() if uid not in done]
        return meta, [uid for uid, _ in items], pool.map_async(decode, items, chunksize=8)

    with open(log, "a") as stream:
        for file in files:
            name = f"data/{file.name}"
            chunks = row_chunks(file)
            following = next(chunks, None)
            current = submit(name, file, *following) if following is not None else None
            while current is not None:
                following = next(chunks, None)
                upcoming = submit(name, file, *following) if following is not None else None
                meta, uids, handle = current
                decoded = handle.get()
                current = upcoming
                dnsmos_jobs = [(args.dnsmos, uid, audio) for uid, audio, err in decoded if err is None and audio is not None]
                dnsmos_results = None
                if args.dnsmos and args.dnsmos_device == "cpu":
                    dnsmos_results = pool.map_async(dnsmos_worker, dnsmos_jobs, chunksize=4)
                results = {}
                usable = [(uid, audio) for uid, audio, err in decoded if err is None and audio is not None and len(audio) >= 1600]
                # Similar lengths per batch: generation runs until the longest transcript of the batch ends.
                usable.sort(key=lambda item: len(item[1]))
                for uid, audio, err in decoded:
                    if err is not None or audio is None or len(audio) < 1600:
                        results[uid] = {"error": err or "too short"}
                for start in range(0, len(usable), args.batch_size):
                    chunk = usable[start : start + args.batch_size]
                    hypotheses = transcribe([a for _, a in chunk])
                    for (uid, _), hypothesis in zip(chunk, hypotheses):
                        hypothesis = hypothesis.strip()
                        try:
                            counts = error_counts(meta[uid]["text"], hypothesis, args.metric_normalization)
                        except ValueError as error:
                            counts = {"error": str(error)}
                        results[uid] = {"hypothesis": hypothesis, **counts}
                    if (start // args.batch_size) % 20 == 0:
                        print(f"  {file.name}: {start + len(chunk)}/{len(usable)} transcribed, {(time.time() - started) / 60:.1f} min", flush=True)
                if dnsmos_results is not None:
                    for uid, score in dnsmos_results.get():
                        results.setdefault(uid, {}).update(score)
                elif args.dnsmos:
                    for start in range(0, len(dnsmos_jobs), 512):
                        chunk = dnsmos_jobs[start : start + 512]
                        for (_, uid, _), score in zip(chunk, gpu_dnsmos.score_many([audio for _, _, audio in chunk])):
                            results.setdefault(uid, {}).update(score)
                for uid in uids:
                    r = meta[uid]
                    record = {
                        "uid": uid,
                        "text": r["text"],
                        "duration_seconds": r.get("duration_seconds"),
                        "quality_score": r.get("quality_score"),
                        "speaker": r.get("speaker") if r.get("speaker") is not None else r.get("speaker_id"),
                        **results.get(uid, {}),
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                total += len(uids)
                elapsed = time.time() - started
                print(f"{file.name}: {len(uids)} rows, total {total}, {elapsed/60:.1f} min", flush=True)
                del decoded
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
