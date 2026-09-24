"""Shared manifest builder of the Turkish source preparers (prepare_yodas.py, prepare_common_voice.py,
prepare_issai_tsc.py).

A preparer turns its source into a stream of candidate utterances; this module applies the filters of
`dacvae_tts.datafilters` cheapest first and writes Parquet parts that `dacvae-tts prepare` reads as they are:

  sampling (--sample-fraction, --limit) -> source rules (e.g. Common Voice down-votes) -> empty text
  -> Freya-TR-Eval exclusion -> Turkish normalization check (the rows `prepare --text-normalization turkish-v1`
  would reject anyway) -> duration from metadata -> decode -> duration / silence -> optional restoration hook
  -> bandwidth (-50 dB) -> optional Silero VAD speech ratio -> chars/s Tukey fences over the whole run
  -> optional per-speaker cap

Output directory:
  manifest/part-XXXXX.parquet  id, audio{bytes,path}, text, speaker, language="tr", source, license,
                               duration_seconds, sample_rate, bandwidth_hz, speech_ratio, chars_per_second,
                               session_id, source_recording, start_seconds, end_seconds, restored, source_meta
  summary.json                 rows and hours removed by each filter, removal fraction, kept bandwidth / speech
                               ratio distribution, chars/s fences, speaker counts, warnings, next commands
  rejected.jsonl               one line per removed row: id, reason and the measured value

Audio is embedded (the original encoded bytes by default, so nothing is transcoded; FLAC for segments cut out of
long recordings or restored audio), which keeps a manifest self-contained after the raw download is deleted and
lets `scripts/transcribe_corpus.py --raw OUTPUT/manifest` read it directly. Ids are namespaced by source
("yodas2/tr000/<utt>", "common-voice/tr/<clip>", "issai-tsc/Train/<file>") because `merge` requires globally
unique uids. There is no `split` column on purpose: `prepare` assigns splits by speaker hash, as for the podcasts.

The chars/s cut needs the whole run's distribution, so by default rows are staged and filtered at the end (one
extra local read/write of the parts). `--cps-bounds LOW,HIGH` (e.g. copied from a 5% sample's summary.json)
makes it a single pass and gives all shard groups of a large source the same cut.
"""

import hashlib
import io
import json
import math
import shutil
import time
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf

from dacvae_tts.datafilters import (
    FreyaExclusion,
    SileroVAD,
    bandwidth_hz,
    chars_per_second,
    duration_ok,
    iqr_bounds,
    load_callable,
    load_sentences,
    text_characters,
)
from dacvae_tts.turkish import normalize_turkish

LANGUAGE = "tr"
REMOVAL_BUDGET = 0.30  # Raon-OpenTTS: -15% worst helps, -50% hurts; leave room for the later CER/DNSMOS cut
MIN_ROWS_FOR_IQR = 20
BANDWIDTH_EDGES_KHZ = [4, 8, 12, 16, 20]


def schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("id", pa.string()),
            ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
            ("text", pa.string()),
            ("speaker", pa.string()),
            ("language", pa.string()),
            ("source", pa.string()),
            ("license", pa.string()),
            ("duration_seconds", pa.float64()),
            ("sample_rate", pa.int32()),
            ("bandwidth_hz", pa.float64()),
            ("speech_ratio", pa.float64()),
            ("chars_per_second", pa.float64()),
            ("session_id", pa.string()),
            ("source_recording", pa.string()),
            ("start_seconds", pa.float64()),
            ("end_seconds", pa.float64()),
            ("restored", pa.string()),
            ("source_meta", pa.string()),
        ]
    )


def add_arguments(parser):
    group = parser.add_argument_group("manifest and filters (shared by all Turkish source preparers)")
    group.add_argument("--output", required=True, help="Directory for manifest/, summary.json, rejected.jsonl")
    group.add_argument(
        "--freya-sentences",
        help="Freya-TR-Eval sentences (.jsonl with `text`, .txt, .csv/.tsv), e.g. data/eval/freya_tr_eval.jsonl; "
        "required unless --no-freya-check",
    )
    group.add_argument("--no-freya-check", action="store_true", help="Skip the evaluation-sentence exclusion")
    group.add_argument(
        "--freya-near-threshold",
        type=float,
        help="Also drop word 3-gram near-duplicates of evaluation sentences at this overlap (e.g. 0.8); off by default",
    )
    group.add_argument("--min-seconds", type=float, default=1.0)
    group.add_argument("--max-seconds", type=float, default=20.0, help="The Turkish cache uses prepare --max-seconds 20")
    group.add_argument(
        "--cps-iqr-k", type=float, default=1.5, help="Tukey fence multiplier over this run's chars/s; 0 disables"
    )
    group.add_argument(
        "--cps-bounds", help="LOW,HIGH fixed chars/s cut (e.g. from a sample run's summary.json); single pass"
    )
    group.add_argument(
        "--min-bandwidth", type=float, default=0.0, help="Minimum -50 dB bandwidth in Hz; 0 only records bandwidth_hz"
    )
    group.add_argument("--bandwidth-threshold-db", type=float, default=50.0)
    group.add_argument("--vad", action="store_true", help="Measure the Silero VAD speech ratio (lazy torch.hub load)")
    group.add_argument("--vad-threshold", type=float, default=0.5)
    group.add_argument("--min-speech-ratio", type=float, default=0.8)
    group.add_argument(
        "--restore",
        metavar="MODULE:FUNCTION",
        help="Optional restoration hook f(audio: float32 mono, sample_rate) -> (audio, sample_rate), applied before "
        "bandwidth/VAD are measured (e.g. a Sidon wrapper); off by default, no dependency is added",
    )
    group.add_argument(
        "--audio-format",
        choices=["original", "flac"],
        default="original",
        help="Embed the source's own encoded bytes (default) or re-encode everything as 16-bit FLAC",
    )
    group.add_argument("--max-per-speaker", type=int, default=0, help="Cap kept clips per speaker (0 = no cap)")
    group.add_argument("--sample-fraction", type=float, default=1.0, help="Deterministic id-hash sample, e.g. 0.05")
    group.add_argument("--limit", type=int, default=0, help="Stop after this many sampled candidates (0 = all)")
    group.add_argument("--seed", type=int, default=42, help="Salt of the sampling hash")
    group.add_argument("--rows-per-part", type=int, default=1000, help="transcribe_corpus.py loads one part at a time")
    group.add_argument("--workers", type=int, default=4, help="Audio decode/measure workers; 0 is serial")
    group.add_argument("--worker-backend", choices=["thread", "process"], default="thread")


def flac_bytes(audio, sample_rate):
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(audio, -1.0, 1.0), sample_rate, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


def decode(candidate):
    """(mono float32, sample rate); embeds a path's original bytes so the manifest does not depend on the file."""
    if candidate.get("array") is not None:
        return np.asarray(candidate["array"], dtype=np.float32).reshape(-1), int(candidate["sample_rate"])
    if candidate.get("audio_bytes") is None:
        candidate["audio_bytes"] = Path(candidate["path"]).read_bytes()
    audio, sample_rate = sf.read(io.BytesIO(candidate["audio_bytes"]), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sample_rate


_HOOKS = {}


def hook(kind, spec):
    """VAD / restoration objects are built once per worker (thread or spawned process) from picklable specs."""
    if callable(spec):
        return spec
    key = (kind, spec)
    if key not in _HOOKS:
        _HOOKS[key] = SileroVAD(threshold=spec) if kind == "vad" else load_callable(spec)
    return _HOOKS[key]


def measure(candidate, config):
    """Decode and measure one candidate; returns a manifest row or {"id", "reject", ...measured values}."""
    uid = candidate["id"]
    try:
        audio, sample_rate = decode(candidate)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("empty or nonfinite audio")
    except Exception as error:  # corrupt MP3 frames, truncated archive members, missing files
        return {"id": uid, "reject": "decode_error", "error": str(error)[:300]}
    duration = len(audio) / sample_rate
    if not duration_ok(duration, config["min_seconds"], config["max_seconds"]):
        return {"id": uid, "reject": "duration", "duration_seconds": duration}
    if float(np.sqrt(np.mean(np.square(audio)))) < 1e-5:
        return {"id": uid, "reject": "silent", "duration_seconds": duration}
    restored = None
    if config.get("restore") is not None:
        audio, sample_rate = hook("restore", config["restore"])(audio, sample_rate)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        restored = config["restore"] if isinstance(config["restore"], str) else getattr(
            config["restore"], "__name__", "callable"
        )
    bandwidth = bandwidth_hz(audio, sample_rate, config["bandwidth_threshold_db"])
    if bandwidth < config["min_bandwidth"]:
        return {"id": uid, "reject": "bandwidth", "duration_seconds": duration, "bandwidth_hz": bandwidth}
    ratio = None
    if config.get("vad") is not None:
        ratio = float(hook("vad", config["vad"])(audio, sample_rate))
        if ratio < config["min_speech_ratio"]:
            return {"id": uid, "reject": "speech_ratio", "duration_seconds": duration, "speech_ratio": ratio}
    original = restored is None and candidate.get("audio_bytes") is not None and config["audio_format"] == "original"
    return {
        "id": uid,
        "audio": {"bytes": candidate["audio_bytes"] if original else flac_bytes(audio, sample_rate), "path": None},
        "text": candidate["text"],
        "speaker": candidate["speaker"],
        "language": LANGUAGE,
        "source": config["source"],
        "license": config["license"],
        "duration_seconds": duration,
        "sample_rate": int(sample_rate),
        "bandwidth_hz": bandwidth,
        "speech_ratio": ratio,
        "chars_per_second": chars_per_second(candidate["text"], duration),
        "session_id": candidate.get("session_id"),
        "source_recording": candidate.get("source_recording"),
        "start_seconds": candidate.get("start_seconds"),
        "end_seconds": candidate.get("end_seconds"),
        "restored": restored,
        "source_meta": json.dumps(candidate.get("meta") or {}, ensure_ascii=False, sort_keys=True),
    }


class PartWriter:
    def __init__(self, directory, rows_per_part):
        self.directory, self.rows_per_part = Path(directory), rows_per_part
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows, self.parts, self.written = [], [], 0

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= self.rows_per_part:
            self.flush()

    def flush(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not self.rows:
            return
        path = self.directory / f"part-{len(self.parts):05d}.parquet"
        # Row groups of 256 let `prepare --num-shards N` split one part across ranks.
        pq.write_table(pa.Table.from_pylist(self.rows, schema=schema()), path, row_group_size=256)
        self.parts.append(path)
        self.written += len(self.rows)
        self.rows = []


def quantiles(values):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(values):
        return None
    p = np.percentile(values, [5, 10, 25, 50, 75, 90, 95])
    return dict(zip(["p5", "p10", "p25", "p50", "p75", "p90", "p95"], map(float, p)), mean=float(values.mean()))


def bandwidth_histogram(values):
    labels = ["<=4k", "4-8k", "8-12k", "12-16k", "16-20k", ">20k"]
    bins = np.digitize(np.asarray(values, dtype=np.float64) / 1000, BANDWIDTH_EDGES_KHZ, right=True)
    counts = np.bincount(bins, minlength=len(labels))
    return dict(zip(labels, map(int, counts)))


class ManifestBuilder:
    """Filter a candidate stream into manifest parts and a per-filter yield report.

    A candidate is a dict: id, text, speaker and one of audio_bytes / path / (array + sample_rate); optional
    duration (metadata, filters before decoding), session_id, source_recording, start_seconds, end_seconds,
    meta (source columns kept as JSON), reject (a source-specific reason, e.g. Common Voice down-votes), or
    skip=True for a candidate the source already sampled out via `selected()` without reading its audio.
    """

    def __init__(self, args, *, source, license, origin=None, vad=None, restore=None):
        self.args, self.source, self.license, self.origin = args, source, license, origin or {}
        if not 0 < args.sample_fraction <= 1:
            raise ValueError("--sample-fraction must be in (0, 1]")
        if args.rows_per_part < 1 or args.limit < 0 or args.max_per_speaker < 0:
            raise ValueError("--rows-per-part must be positive; --limit/--max-per-speaker nonnegative")
        if not 0 < args.min_seconds <= args.max_seconds:
            raise ValueError("Invalid duration bounds")
        self.output = Path(args.output)
        if (self.output / "manifest").exists() and any((self.output / "manifest").iterdir()):
            raise ValueError(f"{self.output}/manifest already exists; use a new output directory")
        if args.freya_sentences:
            self.freya = FreyaExclusion(load_sentences(args.freya_sentences), args.freya_near_threshold)
        elif args.no_freya_check:
            self.freya = None
        else:
            raise ValueError("Pass --freya-sentences (Freya-TR-Eval must never enter training) or --no-freya-check")
        self.fixed_bounds = None
        if args.cps_bounds:
            low, high = (float(v) for v in args.cps_bounds.split(","))
            self.fixed_bounds = (low, high)
        self.config = dict(
            source=source,
            license=license,
            min_seconds=args.min_seconds,
            max_seconds=args.max_seconds,
            min_bandwidth=args.min_bandwidth,
            bandwidth_threshold_db=args.bandwidth_threshold_db,
            vad=vad if vad is not None else (args.vad_threshold if args.vad else None),
            min_speech_ratio=args.min_speech_ratio,
            restore=restore if restore is not None else args.restore,
            audio_format=args.audio_format,
        )
        self.counts, self.removed_seconds = Counter(), Counter()
        self.evaluated = self.sampled_out = 0
        self.evaluated_seconds, self.unknown_duration = 0.0, 0
        self.kept_speakers = Counter()
        self.kept = {"duration": [], "bandwidth": [], "speech_ratio": [], "cps": [], "sample_rate": Counter()}
        self.cps_values = []
        self.bounds = self.fixed_bounds

    # --- sampling -------------------------------------------------------------------------------------------
    def selected(self, uid):
        """Deterministic, order-independent sample: the same ids are chosen in every run with the same seed."""
        if self.args.sample_fraction >= 1:
            return True
        digest = hashlib.sha256(f"{self.args.seed}:{uid}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64 < self.args.sample_fraction

    @property
    def exhausted(self):
        return bool(self.args.limit) and self.evaluated >= self.args.limit

    # --- filtering ------------------------------------------------------------------------------------------
    def reject(self, uid, reason, log, **values):
        self.counts[reason] += 1
        seconds = values.get("duration_seconds")
        if seconds is not None:
            self.removed_seconds[reason] += seconds
        log.write(json.dumps({"id": uid, "reason": reason, **values}, ensure_ascii=False) + "\n")

    def precheck(self, candidate):
        """Metadata-only rules: nothing here decodes audio."""
        if candidate.get("reject"):
            return candidate["reject"]
        text = " ".join(str(candidate.get("text") or "").split())
        candidate["text"] = text
        if not text or not text_characters(text):
            return "empty_text"
        if self.freya is not None:
            reason = self.freya.match(text)
            if reason:
                return reason
        try:
            normalize_turkish(text)
        except ValueError:
            return "text_unnormalizable"
        if not str(candidate.get("speaker") or "").strip():
            return "missing_speaker"
        duration = candidate.get("duration")
        if duration is not None and not duration_ok(duration, self.args.min_seconds, self.args.max_seconds):
            return "duration"
        return None

    def _prechecked(self, candidates, log):
        for candidate in candidates:
            if self.exhausted:
                break
            if candidate.get("skip") or not self.selected(candidate["id"]):
                self.sampled_out += 1
                continue
            self.evaluated += 1
            reason = self.precheck(candidate)
            if reason:
                # Hours are exact for rows that reach decoding; metadata-only rejects count when the source
                # provides a duration (YODAS, Common Voice clip_durations.tsv), otherwise they are counted apart.
                duration = candidate.get("duration")
                if duration is None:
                    self.unknown_duration += 1
                else:
                    self.evaluated_seconds += duration
                self.reject(candidate["id"], reason, log, **({} if duration is None else {"duration_seconds": duration}))
                continue
            yield candidate

    def _final_gate(self, row, writer, log):
        """chars/s fences and the per-speaker cap: the last filters, applied inline or after staging."""
        low, high = self.bounds or (-math.inf, math.inf)
        if not low <= row["chars_per_second"] <= high:
            self.reject(
                row["id"], "chars_per_second", log,
                duration_seconds=row["duration_seconds"], chars_per_second=row["chars_per_second"],
            )
            return
        if self.args.max_per_speaker and self.kept_speakers[row["speaker"]] >= self.args.max_per_speaker:
            self.reject(row["id"], "speaker_cap", log, duration_seconds=row["duration_seconds"])
            return
        self.kept_speakers[row["speaker"]] += 1
        self.kept["duration"].append(row["duration_seconds"])
        self.kept["bandwidth"].append(row["bandwidth_hz"])
        self.kept["speech_ratio"].append(row["speech_ratio"])
        self.kept["cps"].append(row["chars_per_second"])
        self.kept["sample_rate"][row["sample_rate"]] += 1
        writer.add(row)

    def run(self, candidates):
        from dacvae_tts.prepare import ordered_prefetch

        started = time.perf_counter()
        self.output.mkdir(parents=True, exist_ok=True)
        staged = self.fixed_bounds is None and self.args.cps_iqr_k > 0
        final = PartWriter(self.output / "manifest", self.args.rows_per_part)
        staging = PartWriter(self.output / ".staging", self.args.rows_per_part) if staged else None
        with open(self.output / "rejected.jsonl", "w") as log:
            rows = ordered_prefetch(
                partial(measure, config=self.config),
                self._prechecked(candidates, log),
                self.args.workers,
                max(1, 4 * self.args.workers),
                self.args.worker_backend,
            )
            for row in rows:
                if row.get("duration_seconds") is not None:
                    self.evaluated_seconds += row["duration_seconds"]
                else:
                    self.unknown_duration += 1  # undecodable
                if "reject" in row:
                    values = {k: v for k, v in row.items() if k not in {"id", "reject"}}
                    self.reject(row["id"], row["reject"], log, **values)
                    continue
                if staging is not None:
                    self.cps_values.append(row["chars_per_second"])
                    staging.add(row)
                else:
                    self._final_gate(row, final, log)
            if staging is not None:
                staging.flush()
                self._finalize_staged(staging, final, log)
        final.flush()
        summary = self.summary(time.perf_counter() - started, final)
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(json.dumps({k: summary[k] for k in ("source", "evaluated", "kept", "removed", "removed_fraction")}))
        for warning in summary["warnings"]:
            print(f"WARNING: {warning}")
        return summary

    def _finalize_staged(self, staging, final, log):
        import pyarrow.parquet as pq

        if len(self.cps_values) >= MIN_ROWS_FOR_IQR:
            self.bounds = iqr_bounds(self.cps_values, self.args.cps_iqr_k)
        for part in staging.parts:
            for row in pq.read_table(part).to_pylist():
                self._final_gate(row, final, log)
            part.unlink()
        shutil.rmtree(staging.directory, ignore_errors=True)

    # --- report ---------------------------------------------------------------------------------------------
    def summary(self, seconds, writer):
        kept = writer.written
        removed = dict(sorted(self.counts.items(), key=lambda item: -item[1]))
        removed_total = sum(self.counts.values())
        fraction = removed_total / self.evaluated if self.evaluated else 0.0
        kept_hours = sum(self.kept["duration"]) / 3600
        warnings = []
        if fraction > REMOVAL_BUDGET:
            warnings.append(
                f"manifest filters removed {fraction:.1%} of evaluated rows (> {REMOVAL_BUDGET:.0%}); Raon-OpenTTS "
                "found removing 50% hurts (Seed-TTS WER 2.19 -> 2.32) - inspect summary.removed before the CER/DNSMOS cut"
            )
        if self.freya is None:
            warnings.append("Freya-TR-Eval exclusion was disabled (--no-freya-check); do not train on this manifest")
        if self.args.cps_iqr_k > 0 and self.fixed_bounds is None and len(self.cps_values) < MIN_ROWS_FOR_IQR:
            warnings.append(f"fewer than {MIN_ROWS_FOR_IQR} rows: chars/s fences were not applied")
        median_bandwidth = float(np.median(self.kept["bandwidth"])) if self.kept["bandwidth"] else None
        if median_bandwidth is not None and median_bandwidth < 12000:
            warnings.append(
                f"median kept bandwidth is {median_bandwidth / 1000:.1f} kHz; a latent TTS reproduces its training "
                "bandwidth (the podcast cache is 12-16 kHz) - consider --min-bandwidth, restoration, or a smaller mix share"
            )
        manifest = self.output / "manifest"
        return {
            "source": self.source,
            "license": self.license,
            "origin": self.origin,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "arguments": {k: v for k, v in vars(self.args).items() if isinstance(v, (str, int, float, bool, type(None)))},
            "manifest": str(manifest),
            "parts": len(writer.parts),
            "sampled_out": self.sampled_out,
            "evaluated": self.evaluated,
            "kept": kept,
            "removed": removed,
            "removed_fraction": fraction,
            "evaluated_hours": self.evaluated_seconds / 3600,
            "evaluated_without_known_duration": self.unknown_duration,
            "removed_hours": {k: v / 3600 for k, v in sorted(self.removed_seconds.items())},
            "kept_hours": kept_hours,
            "speakers": len(self.kept_speakers),
            "speakers_with_2plus_clips": sum(1 for n in self.kept_speakers.values() if n >= 2),
            "clips_per_speaker": quantiles(list(self.kept_speakers.values())),
            "chars_per_second_bounds": list(self.bounds) if self.bounds else None,
            "chars_per_second_mode": "fixed" if self.fixed_bounds else ("iqr" if self.args.cps_iqr_k > 0 else "off"),
            "kept_chars_per_second": quantiles(self.kept["cps"]),
            "kept_duration_seconds": quantiles(self.kept["duration"]),
            "kept_bandwidth_hz": quantiles(self.kept["bandwidth"]),
            "kept_bandwidth_histogram": bandwidth_histogram(self.kept["bandwidth"]) if self.kept["bandwidth"] else {},
            "kept_speech_ratio": quantiles(self.kept["speech_ratio"]),
            "kept_sample_rates": {str(k): v for k, v in sorted(self.kept["sample_rate"].items())},
            "freya_sentences": len(self.freya) if self.freya is not None else 0,
            "wall_seconds": seconds,
            "warnings": warnings,
            "next_steps": next_steps(self.output, self.source),
        }


def next_steps(output, source):
    name = str(source).replace("/", "-")
    manifest = f"{output}/manifest"
    return [
        f"python scripts/transcribe_corpus.py --raw {manifest} --output outputs/scores-{name} --device cuda "
        "--dnsmos models/sig_bak_ovr.onnx",
        f"python scripts/make_drop_list.py --scores outputs/scores-{name}/scores.jsonl --max-cer 0.1 --min-ovrl 2.8 "
        f"--min-words 2 --output data/drop-{name}.json",
        f"dacvae-tts prepare --manifest {manifest} --output data/cache/{name} --device cuda --speaker-column speaker "
        "--text-normalization turkish-v1 --languages tr --loudness -16 --min-seconds 1 --max-seconds 20",
        f"dacvae-tts merge --inputs data/tr55/parts/part-* data/cache/{name} --output data/tr-mix "
        "--drop-uids data/drop-all.json --keep-singletons   # drop-all.json: jq -s add data/drop-*.json",
    ]
