"""Turkish YODAS manifests for `dacvae-tts prepare` (YouTube, CC-BY-3.0), one tar shard at a time.

Variants - the same YouTube utterances with the same utt ids, so use ONE variant per subset (mixing them stores the
same speech twice at different sampling rates, which `merge` cannot deduplicate). tr000 sizes as of 2026-09:
- `yodas`  espnet/yodas: utterance-level 16 kHz, bandwidth <= 8 kHz; 64 shards of ~1 GB.
           data/<subset>/audio/NNNNNNNN.tar.gz + text/NNNNNNNN.txt ("utt text") + duration/NNNNNNNN.txt.
- `yodas2` espnet/yodas2 (default): video-level 24 kHz (<= 12 kHz) + utterance timestamps in text/NNNNNNNN.json;
           155 shards of ~1.4 GB (whole videos, untranscribed stretches included). Segments are cut here (seek and
           read, not a full decode) and stored as 16-bit FLAC.
- `sidon`  sarulab-speech/yodas2_sidon: YODAS2 restored with Sidon (arXiv 2509.17052; model MIT, data CC-BY-3.0),
           24 kHz WebDataset <subset>/train-NNNNN.tar.gz (<key>.flac + <key>.metadata.json); 43 shards of ~3 GB.
           Sidon raised Turkish FLEURS DNSMOS 3.07 -> 3.45 with CER unchanged; restored audio is still a model
           output, so compare SIM-o against `yodas2` before preferring it.
Subsets: tr000 = manual captions (588.7 h), tr100 = automatic captions (4,067.7 h). "Manual" only means uploaded by
the channel: many are translated subtitles of foreign-language videos (TED-Ed style), so the Whisper re-transcription
and CER cut below is required for YODAS, not optional. Espnet's loading scripts need the legacy `datasets` script
support, so the shard files are read directly (huggingface_hub download or --local-dir).

Speakers: YODAS has no speaker labels (and its video ids are not YouTube ids, so no channel either). speaker =
"yodas/<video id>" groups a video's utterances; interviews and panels put several voices under one label, and
cross-utterance prompting would pair them. Verify with speaker embeddings (e.g. WavLM-SV cosine to the video
centroid, drop < ~0.6) before relying on cross-utterance pairs, or use --speaker-mode utterance (each clip its own
speaker; within-utterance prompting only; merge with --keep-singletons).
Captions: HTML entities are unescaped; captions with markup ([Müzik], (Alkış), ♪) are dropped as `caption_markup`.

Chain (5% sample first, shards spread over the subset: yodas 0,21,42 / yodas2 0,20,40,60,80,100,120,140 / sidon 0,21;
a shard's videos are downloaded whole, so sampling by shard saves the download that --sample-fraction cannot):
  python scripts/data/prepare_yodas.py --variant yodas2 --subset tr000 --shards 0,20,40,60,80,100,120,140 \
      --output data/sources/yodas2-tr000-sample --freya-sentences data/eval/freya_tr_eval.jsonl --vad
  # read summary.json: removed/removed_fraction, kept_bandwidth_histogram, chars_per_second_bounds; then the full run
  # in shard groups sharing the sample's chars/s cut (one process per group, separate outputs):
  python scripts/data/prepare_yodas.py --variant yodas2 --subset tr000 --shards 0-77 --cps-bounds LOW,HIGH \
      --output data/sources/yodas2-tr000/g0 --freya-sentences data/eval/freya_tr_eval.jsonl --vad
  python scripts/transcribe_corpus.py --raw data/sources/yodas2-tr000/g0/manifest --output outputs/scores-yodas2-tr000-g0 \
      --device cuda --dnsmos models/sig_bak_ovr.onnx
  python scripts/make_drop_list.py --scores outputs/scores-yodas2-tr000-g0/scores.jsonl --max-cer 0.1 --min-ovrl 2.8 \
      --min-words 2 --output data/drop-yodas2-tr000-g0.json
  dacvae-tts prepare --manifest data/sources/yodas2-tr000/g0/manifest --output data/cache/yodas2-tr000-g0 --device cuda \
      --speaker-column speaker --text-normalization turkish-v1 --languages tr --loudness -16 --min-seconds 1 --max-seconds 20
  jq -s add data/drop-*.json > data/drop-all.json
  dacvae-tts merge --inputs data/tr55/parts/part-* data/cache/yodas2-tr000-* --output data/tr-plus-yodas \
      --drop-uids data/drop-all.json --keep-singletons
`--text-normalization turkish-v1`, `--loudness -16` and the default seed must match the podcast cache, or `merge`
refuses the mix.
"""

import argparse
import html
import io
import json
import os
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest_pipeline import ManifestBuilder, add_arguments  # noqa: E402

from dacvae_tts.datafilters import check_training_source  # noqa: E402
from dacvae_tts.turkish import tr_lower  # noqa: E402

LICENSE = "CC-BY-3.0"
VARIANTS = {
    "yodas": dict(repo="espnet/yodas", audio="data/{subset}/audio/{index:08d}.tar.gz",
                  text="data/{subset}/text/{index:08d}.txt", duration="data/{subset}/duration/{index:08d}.txt"),
    "yodas2": dict(repo="espnet/yodas2", audio="data/{subset}/audio/{index:08d}.tar.gz",
                   text="data/{subset}/text/{index:08d}.json"),
    "sidon": dict(repo="sarulab-speech/yodas2_sidon", audio="{subset}/train-{index:05d}.tar.gz"),
}
MARKUP = re.compile(r"[\[\]♪♫]|\((?:müzik|muzik|alkış|alkis|gülüş|gülme|kahkaha|music|applause|laugh)[^)]*\)")


def clean_caption(text):
    return " ".join(html.unescape(str(text or "")).split())


def caption_reason(text):
    return "caption_markup" if MARKUP.search(tr_lower(text)) else None


def parse_utt_id(utt_id):
    """`<video>-<index>-<start cs>-<end cs>`; video ids may contain '-', so split from the right."""
    video, index, start, end = utt_id.rsplit("-", 3)
    return video, int(index), int(start) / 100, int(end) / 100


def speaker_of(video, utt_id, mode):
    return f"yodas/{video}" if mode == "video" else f"yodas/{utt_id}"


def shard_list(spec):
    result = []
    for part in spec.split(","):
        first, _, last = part.partition("-")
        result.extend(range(int(first), int(last or first) + 1))
    return result


class ShardFiles:
    """Shard files from a local mirror (--local-dir, never deleted) or downloaded one at a time and deleted."""

    def __init__(self, repo, local_dir, raw_dir, keep):
        self.repo, self.local_dir, self.raw_dir, self.keep = repo, local_dir, Path(raw_dir), keep

    def available(self, pattern, subset):
        prefix = pattern.format(subset=subset, index=0).rsplit("/", 1)[0] + "/"
        if self.local_dir:
            names = [str(p.relative_to(self.local_dir)) for p in Path(self.local_dir, prefix).glob("*")]
        else:
            from huggingface_hub import list_repo_files

            names = list_repo_files(self.repo, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
        indices = []
        for name in names:
            for index in [int(s) for s in re.findall(r"(\d{5,8})\.tar\.gz$", name)]:
                if name == pattern.format(subset=subset, index=index):
                    indices.append(index)
        return sorted(indices)

    def get(self, relative):
        if self.local_dir:
            path = Path(self.local_dir) / relative
            if not path.exists():
                raise FileNotFoundError(path)
            return path
        from huggingface_hub import hf_hub_download

        for attempt in range(6):
            try:
                return Path(hf_hub_download(self.repo, relative, repo_type="dataset", local_dir=self.raw_dir,
                                            token=os.environ.get("HF_TOKEN")))
            except Exception as error:  # hours of downloading see transient failures
                if attempt == 5:
                    raise
                print(f"download retry {attempt + 1} for {relative}: {error}", flush=True)
                time.sleep(10 * (attempt + 1))

    def release(self, path):
        if not self.local_dir and not self.keep and path is not None:
            Path(path).unlink(missing_ok=True)


def read_spans(blob, spans):
    """[(audio, sample_rate)] per (start, end) span: seek-and-read (FLAC/WAV) so an hour-long video is not decoded
    whole for a few utterances; a full decode if seeking fails at any span, so results never mix the two paths."""
    try:
        results = []
        with sf.SoundFile(io.BytesIO(blob)) as stream:
            sample_rate, frames = stream.samplerate, stream.frames
            if not stream.seekable():
                raise RuntimeError("not seekable")
            for start, end in spans:
                first, last = max(0, int(round(start * sample_rate))), min(frames, int(round(end * sample_rate)))
                stream.seek(first)
                audio = stream.read(max(0, last - first), dtype="float32", always_2d=True)
                results.append((audio.mean(axis=1), sample_rate))
        return results
    except (sf.LibsndfileError, RuntimeError):
        audio, sample_rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        return [(audio[int(round(s * sample_rate)) : int(round(e * sample_rate))], sample_rate) for s, e in spans]


def longform_candidates(builder, variant, subset, video, blob, utterances, member, speaker_mode):
    """Cut one video's utterances; only the sampled ones that pass the text rules are decoded."""
    decode = []
    for utt_id, text, start, end in utterances:
        uid = f"{variant}/{subset}/{utt_id}"
        if not builder.selected(uid):
            yield {"id": uid, "skip": True}
            continue
        text = clean_caption(text)
        candidate = dict(
            id=uid, text=text, speaker=speaker_of(video, utt_id, speaker_mode), duration=end - start,
            session_id=video, source_recording=member, start_seconds=start, end_seconds=end,
            meta={"utt_id": utt_id, "video_id": video, "subset": subset}, reject=caption_reason(text),
        )
        if builder.precheck(dict(candidate)):
            yield candidate  # rejected again (and counted) by the builder, without audio
        else:
            decode.append(candidate)
    if not decode:
        return
    try:
        spans = read_spans(blob, [(c["start_seconds"], c["end_seconds"]) for c in decode])
    except Exception as error:
        for candidate in decode:
            yield {**candidate, "reject": "decode_error", "meta": {**candidate["meta"], "error": str(error)[:200]}}
        return
    for candidate, (audio, sample_rate) in zip(decode, spans):
        yield {**candidate, "array": audio, "sample_rate": sample_rate}


def yodas_shard(builder, files, subset, index, speaker_mode):
    spec = VARIANTS["yodas"]
    paths = []
    try:
        paths.append(files.get(spec["text"].format(subset=subset, index=index)))
        paths.append(files.get(spec["duration"].format(subset=subset, index=index)))
        texts, durations = {}, {}
        for line in paths[0].read_text(encoding="utf-8").splitlines():
            fields = line.strip().split(maxsplit=1)
            if fields:
                texts[fields[0]] = fields[1] if len(fields) > 1 else ""
        for line in paths[1].read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) == 2:
                durations[fields[0]] = float(fields[1])
        paths.append(files.get(spec["audio"].format(subset=subset, index=index)))
        with tarfile.open(paths[2], mode="r|gz") as tar:
            for member in tar:
                if builder.exhausted:
                    return
                utt_id = Path(member.name).stem
                if not member.isfile() or utt_id not in texts:
                    continue
                uid = f"yodas/{subset}/{utt_id}"
                if not builder.selected(uid):
                    yield {"id": uid, "skip": True}  # streaming tar skips the unread member data
                    continue
                video, _, start, end = parse_utt_id(utt_id)
                text = clean_caption(texts[utt_id])
                candidate = dict(
                    id=uid, text=text, speaker=speaker_of(video, utt_id, speaker_mode), duration=durations.get(utt_id),
                    session_id=video, source_recording=f"{paths[2].name}:{member.name}", start_seconds=start,
                    end_seconds=end, meta={"utt_id": utt_id, "video_id": video, "subset": subset},
                    reject=caption_reason(text),
                )
                if not builder.precheck(dict(candidate)):
                    candidate["audio_bytes"] = tar.extractfile(member).read()
                yield candidate
    finally:
        for path in paths:
            files.release(path)


def yodas2_shard(builder, files, subset, index, speaker_mode):
    spec = VARIANTS["yodas2"]
    paths = []
    try:
        paths.append(files.get(spec["text"].format(subset=subset, index=index)))
        videos = {}
        for entry in json.loads(paths[0].read_text(encoding="utf-8")):
            videos[entry["audio_id"]] = [(k, v, *parse_utt_id(k)[2:]) for k, v in sorted(entry["text"].items())]
        paths.append(files.get(spec["audio"].format(subset=subset, index=index)))
        with tarfile.open(paths[1], mode="r|gz") as tar:
            for member in tar:
                if builder.exhausted:
                    return
                video = Path(member.name).stem
                if not member.isfile() or video not in videos:
                    continue
                blob = tar.extractfile(member).read()
                yield from longform_candidates(builder, "yodas2", subset, video, blob, videos[video],
                                               f"{paths[1].name}:{member.name}", speaker_mode)
    finally:
        for path in paths:
            files.release(path)


def sidon_utterances(metadata):
    """metadata.json utterances as (utt_id, text, start, end); accepts list-of-dicts or dict-of-lists."""
    utterances = metadata.get("utterances") or []
    if isinstance(utterances, dict):
        utterances = [dict(zip(utterances, values)) for values in zip(*utterances.values())]
    return [(u["utt_id"], u["text"], float(u["start"]), float(u["end"])) for u in utterances]


def sidon_shard(builder, files, subset, index, speaker_mode):
    """WebDataset members arrive as <key>.flac / <key>.metadata.json; a pair is processed once both are read."""
    audio_path = None
    pending = {}
    try:
        audio_path = files.get(VARIANTS["sidon"]["audio"].format(subset=subset, index=index))
        with tarfile.open(audio_path, mode="r|gz") as tar:
            for member in tar:
                if builder.exhausted:
                    return
                directory, _, base = member.name.rpartition("/")
                key, _, extension = base.partition(".")
                if not member.isfile() or extension not in {"flac", "metadata.json"}:
                    continue
                entry = pending.setdefault(f"{directory}/{key}", {})
                entry[extension] = tar.extractfile(member).read()
                if len(entry) < 2:
                    continue
                del pending[f"{directory}/{key}"]
                metadata = json.loads(entry["metadata.json"])
                video = metadata.get("video_id") or key
                yield from longform_candidates(builder, "sidon", subset, video, entry["flac"],
                                               sidon_utterances(metadata), f"{audio_path.name}:{key}", speaker_mode)
    finally:
        files.release(audio_path)


SHARD_READERS = {"yodas": yodas_shard, "yodas2": yodas2_shard, "sidon": sidon_shard}


def candidates(builder, files, variant, subset, shards, speaker_mode):
    for index in shards:
        if builder.exhausted:
            return
        print(f"{variant}/{subset} shard {index}", flush=True)
        yield from SHARD_READERS[variant](builder, files, subset, index, speaker_mode)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="yodas2")
    parser.add_argument("--subset", default="tr000", help="tr000 (manual captions) or tr100 (automatic captions)")
    parser.add_argument("--repo", help="Override the variant's repository id")
    parser.add_argument("--shards", default="all", help="e.g. 0-2 or 0-9,40 (shard file indices), or all")
    parser.add_argument("--local-dir", help="Local mirror with the repository's file layout (files are kept)")
    parser.add_argument("--raw-dir", help="Download directory (default OUTPUT/raw); shards are deleted after use")
    parser.add_argument("--keep-downloads", action="store_true")
    parser.add_argument("--speaker-mode", choices=["video", "utterance"], default="video")
    add_arguments(parser)
    args = parser.parse_args(argv)
    if not args.subset.startswith("tr"):
        raise SystemExit("This preparer is for the Turkish subsets (tr000, tr100)")
    spec = VARIANTS[args.variant]
    repo = args.repo or spec["repo"]
    check_training_source(repo, args.local_dir)
    files = ShardFiles(repo, args.local_dir, args.raw_dir or Path(args.output) / "raw", args.keep_downloads)
    shards = files.available(spec["audio"], args.subset) if args.shards == "all" else shard_list(args.shards)
    if not shards:
        raise SystemExit(f"No {args.variant} shards found for {args.subset}")
    builder = ManifestBuilder(
        args, source=f"{args.variant}-{args.subset}", license=LICENSE,
        origin={"repo": repo, "variant": args.variant, "subset": args.subset, "shards": shards,
                "speaker_mode": args.speaker_mode},
    )
    summary = builder.run(candidates(builder, files, args.variant, args.subset, shards, args.speaker_mode))
    if not args.local_dir and not args.keep_downloads and not args.raw_dir:
        # The shard files are deleted after use; drop the default download folder with hf_hub_download's metadata.
        shutil.rmtree(Path(args.output) / "raw", ignore_errors=True)
    return summary


if __name__ == "__main__":
    main()
