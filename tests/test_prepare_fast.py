import json
import sqlite3
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

from dacvae_tts.codec import Codec
from dacvae_tts.data import LatentDataset, collate
from dacvae_tts.prepare import encode_records, merge, ordered_prefetch, prepare, source_rows
from dacvae_tts.text import tokenize, tokenize_bytes


class ToyCodec(Codec):
    """Biased convolutions make padding at the wrong boundary observable."""

    def __init__(self, *args, **kwargs):
        self.runtime = {"backend": "reference"}
        self.device = torch.device("cpu")
        self.sample_rate, self.hop_length, self.latent_dim = 8000, 8, 4
        self.checkpoint, self.weights_sha256 = "fixture", "fixture"
        self.model = SimpleNamespace(
            encoder=nn.Sequential(
                nn.Conv1d(1, 4, 5, stride=2, padding=2), nn.GELU(), nn.Conv1d(4, 4, 8, stride=4, padding=2)
            ).eval(),
            quantizer=SimpleNamespace(in_proj=nn.Conv1d(4, 8, 1).eval()),
            _pad=lambda x: torch.nn.functional.pad(x, (0, (-x.size(-1)) % 8), mode="reflect"),
        )


def test_equal_hop_batches_preserve_single_encoding():
    codec = ToyCodec()
    audios = [torch.randn(n) for n in (17, 23, 24)]
    batched = codec.encode_batch(audios)
    assert batched.shape == (3, 3, 4)
    for audio, encoded in zip(audios, batched):
        torch.testing.assert_close(encoded, codec.encode(audio), atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="equal hop-rounded"):
        codec.encode_batch([torch.randn(24), torch.randn(25)])
    with pytest.raises(ValueError, match="finite mono"):
        codec.encode_batch([torch.full((24,), float("nan"))])
    with pytest.raises(ValueError, match="too short"):
        codec.encode_batch([torch.ones(8)])
    with pytest.raises(ValueError, match="requires CUDA"):
        codec.encode_batch(audios, "bf16")


def test_bucket_restores_order_and_splits_only_oom():
    codec = ToyCodec()
    records = [{"audio": torch.randn(n)} for n in (33, 24, 23, 34, 17, 25)]
    original = codec.encode_batch
    calls = []

    def limited(audios, precision):
        calls.append(len(audios))
        if len(audios) > 2:
            raise torch.cuda.OutOfMemoryError("synthetic test")
        return original(audios, precision)

    codec.encode_batch = limited
    counters = {"encoder_calls": 0, "oom_retries": 0}
    output = encode_records(codec, records, 8, 1, "fp32", counters)
    assert counters["oom_retries"] == 1
    assert counters["encoder_calls"] == len(calls)
    for record, encoded in zip(records, output):
        torch.testing.assert_close(encoded, codec.encode(record["audio"]), atol=1e-6, rtol=1e-6)
    codec.encode_batch = lambda *args: (_ for _ in ()).throw(RuntimeError("broken codec"))
    with pytest.raises(RuntimeError, match="broken codec"):
        encode_records(codec, records, 8, 1, "fp32", counters)


def test_prefetch_is_bounded_and_ordered():
    consumed = []
    started = threading.Event()

    def rows():
        for i in range(20):
            consumed.append(i)
            yield i

    def work(i):
        started.set()
        return i * 2

    stream = ordered_prefetch(work, rows(), 2, 3)
    assert next(stream) == 0
    assert started.is_set() and len(consumed) == 4  # three queued plus one refill
    assert list(stream) == list(range(2, 40, 2))
    assert list(ordered_prefetch(work, range(3), 0, 1)) == [0, 2, 4]


def test_jsonl_partitions_parse_only_assigned_rows(tmp_path, monkeypatch):
    import dacvae_tts.prepare as module

    path = tmp_path / "source.jsonl"
    path.write_text("\n".join(json.dumps({"text": str(i)}) + "\n" for i in range(33)))
    loads = json.loads
    calls = []

    def count_loads(data):
        calls.append(1)
        return loads(data)

    monkeypatch.setattr(module.json, "loads", count_loads)
    rows = [row for rank in range(8) for row in source_rows(path, rank, 8)]
    assert len(calls) == 33  # not 8 * 33
    assert {row["id"] for row in rows} == {str(i) for i in range(33)}


def args_for(manifest, output, **overrides):
    values = dict(
        manifest=manifest,
        output=output,
        codec="fixture",
        device="cpu",
        text_column="text",
        audio_column="audio",
        speaker_column="speaker_id",
        text_normalization="unicode-v1",
        min_seconds=0.02,
        max_seconds=1.0,
        seed=42,
        shard_index=0,
        num_shards=1,
        workers=2,
        prefetch=3,
        batch_size=4,
        bucket_size=8,
        batch_seconds=4,
        precision="fp32",
    )
    return SimpleNamespace(**(values | overrides))


def test_prepare_merge_cached_text_and_legacy_equivalence(tmp_path, monkeypatch):
    import dacvae_tts.prepare as module

    torch.manual_seed(11)
    codec = ToyCodec()
    monkeypatch.setattr(module, "Codec", lambda *args, **kwargs: codec)
    rows = []
    for i, n in enumerate((400, 399, 397, 480, 477, 479, 560, 557)):
        path = tmp_path / f"{i}.wav"
        sf.write(path, np.random.default_rng(i).normal(0, 0.1, n).astype(np.float32), 8000, subtype="FLOAT")
        rows.append(
            dict(
                id=str(i),
                audio=path.name,
                text=" José’s café. ",
                speaker_id=f"speaker-{i // 2}",
                split="train" if i < 6 else "val",
            )
        )
    rows.extend(
        [
            dict(rows[0], id="duplicate"),
            dict(rows[0], id="corrupt", audio="missing.wav"),
            dict(rows[0], id="empty-text", text=" "),
        ]
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(map(json.dumps, rows)))
    outputs = [tmp_path / "serial", tmp_path / "parallel"]
    prepare(args_for(manifest, outputs[0], workers=0, batch_size=1, bucket_size=1))
    prepare(args_for(manifest, outputs[1]))
    for output in outputs:
        meta = json.loads((output / "metadata.json").read_text())
        assert (meta["accepted"], meta["rejected"]) == (8, 3)
        stats = torch.load(output / "stats.pt", weights_only=True)
        assert stats["count"] == 3 * 50 + 3 * 60  # excludes validation
        assert meta["text_tokenizer"] == "utf8-bytes-v1"
    datasets = [LatentDataset(output) for output in outputs]
    for i in range(6):
        torch.testing.assert_close(
            datasets[0].row(i)["latents"], datasets[1].row(i)["latents"], atol=1e-3, rtol=1e-3
        )
        assert datasets[0].row(i)["uid"] == datasets[1].row(i)["uid"]
    optimized = json.loads((outputs[1] / "metadata.json").read_text())["preparation"]
    assert optimized["encoder_calls"] == 3
    merged = tmp_path / "merged"
    merge(SimpleNamespace(inputs=[outputs[1]], output=merged))
    dataset = LatentDataset(merged)
    items = [dataset[0], dataset[1]]
    cached = collate(items)
    assert items[0]["text_bytes"] == "José's café.".encode()
    assert items[0]["token_ids"][0] == 1 and items[0]["token_ids"][-1] == 3  # BOS ... EOS, prepared once
    for item in items:
        item.pop("token_ids")
        item.pop("reference_token_ids")
    bytes_fallback = collate(items)
    for item in items:
        item.pop("text_bytes")
        item.pop("reference_text_bytes")
    fallback = collate(items)
    for key in cached:
        torch.testing.assert_close(cached[key], fallback[key])
        torch.testing.assert_close(cached[key], bytes_fallback[key])
    with sqlite3.connect(merged / "index.sqlite") as db:
        db.execute("DROP TABLE text_tokens")
        db.execute("DROP TABLE token_ids")
    legacy = LatentDataset(merged)
    assert legacy.row(0)["text_bytes"] is None and legacy.row(0)["token_ids"] is None
    torch.testing.assert_close(collate([legacy[0]])["tokens"], cached["tokens"][:1])


@pytest.mark.parametrize("reference,target", [("", "Hello"), ("José", "Café — déjà vu!"), ("A", "Z" * 1000)])
def test_cached_bytes_preserve_special_ids(reference, target):
    from dacvae_tts.text import normalize

    direct = tokenize(reference, target)
    cached = tokenize_bytes(normalize(reference).encode() if reference else b"", normalize(target).encode())
    for before, after in zip(direct, cached):
        assert torch.equal(before, after)


def test_failed_encode_never_marks_cache_complete(tmp_path, monkeypatch):
    import dacvae_tts.prepare as module

    codec = ToyCodec()
    codec.encode_batch = lambda *args: (_ for _ in ()).throw(RuntimeError("codec failed"))
    monkeypatch.setattr(module, "Codec", lambda *args, **kwargs: codec)
    sf.write(tmp_path / "sample.wav", np.ones(400) * 0.1, 8000)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(dict(audio="sample.wav", text="Hello", speaker_id="one")))
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="codec failed"):
        prepare(args_for(manifest, output))
    assert not (output / "metadata.json").exists()


def test_eight_partitions_embedded_parquet_and_stereo_resampling(tmp_path, monkeypatch):
    import io

    import dacvae_tts.prepare as module

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    codec = ToyCodec()
    monkeypatch.setattr(module, "Codec", lambda *args, **kwargs: codec)
    rows = []
    for i in range(16):
        buffer = io.BytesIO()
        waveform = np.random.default_rng(i).normal(0, 0.1, (800 + i, 2)).astype(np.float32)
        sf.write(buffer, waveform, 16000, format="WAV", subtype="FLOAT")
        rows.append(
            dict(
                audio={"bytes": buffer.getvalue()},
                text=f"Sentence {i}.",
                speaker_id=f"speaker-{i % 8}",
                split="train",
            )
        )
    manifest = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pylist(rows), manifest, row_group_size=2)
    parts = [tmp_path / f"part-{rank}" for rank in range(8)]
    for rank, output in enumerate(parts):
        prepare(args_for(manifest, output, num_shards=8, shard_index=rank))
    output = tmp_path / "merged"
    merge(SimpleNamespace(inputs=parts, output=output))
    dataset = LatentDataset(output)
    assert len(dataset) == 16
    assert {dataset.row(i)["uid"] for i in range(16)} == {str(i) for i in range(16)}
    for i in range(16):
        assert dataset.row(i)["text_bytes"] is not None
        assert dataset[i]["uid"] != dataset[i]["reference_uid"]


def test_merge_rejects_mixed_encoding_precision(cache, tmp_path):
    import shutil

    other = tmp_path / "bf16"
    shutil.copytree(cache, other)
    metadata = json.loads((other / "metadata.json").read_text())
    metadata["encoder_precision"] = "bf16"
    (other / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="different encoder precisions"):
        merge(SimpleNamespace(inputs=[cache, other], output=tmp_path / "merged"))


def write_manifest(directory, name, count, **columns):
    rows = []
    for i in range(count):
        path = directory / f"{name}-{i}.wav"
        audio = np.random.default_rng([*name.encode(), i]).normal(0, 0.1, 400 + 8 * i).astype(np.float32)
        sf.write(path, audio, 8000, subtype="FLOAT")
        rows.append(dict(id=f"{name}-{i}", audio=path.name, text="Hello there.", speaker_id=f"{name}-{i // 2}",
                         split="train", **columns))
    manifest = directory / f"{name}.jsonl"
    manifest.write_text("\n".join(map(json.dumps, rows)))
    return manifest


def test_source_licenses_reach_prepare_and_merge_metadata(tmp_path, monkeypatch):
    # scripts/data/ manifests name each row's source and license (YODAS is CC-BY-3.0, which needs attribution);
    # prepare used to drop both columns, so no cache recorded which licenses it contains.
    import dacvae_tts.prepare as module

    codec = ToyCodec()
    monkeypatch.setattr(module, "Codec", lambda *args, **kwargs: codec)
    yodas = write_manifest(tmp_path, "yodas", 4, source="yodas2-tr000", license="CC-BY-3.0")
    lines = yodas.read_text().splitlines()
    rejected = dict(json.loads(lines[0]), id="yodas-missing", audio="missing.wav")
    yodas.write_text("\n".join([*lines, json.dumps(rejected)]))
    manifests = {
        "yodas": yodas,
        "cv": write_manifest(tmp_path, "cv", 2, source="common-voice-tr", license="CC0-1.0"),
        "plain": write_manifest(tmp_path, "plain", 2),  # a dataset without the columns
    }
    for name, manifest in manifests.items():
        prepare(args_for(manifest, tmp_path / f"cache-{name}", max_seconds=2.0))
    meta = {name: json.loads((tmp_path / f"cache-{name}" / "metadata.json").read_text()) for name in manifests}
    assert meta["yodas"]["sources"] == {"yodas2-tr000": {"licenses": ["CC-BY-3.0"], "prepared_rows": 4}}
    assert meta["yodas"]["rejected"] == 1  # rejected rows are not counted
    assert "sources" not in meta["plain"]
    # The index schema is untouched: the licenses live in metadata only.
    schemas = set()
    for name in manifests:
        with sqlite3.connect(tmp_path / f"cache-{name}" / "index.sqlite") as db:
            schemas.add(tuple(db.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()))
    assert len(schemas) == 1
    merged = tmp_path / "merged"
    merge(SimpleNamespace(inputs=[tmp_path / f"cache-{name}" for name in manifests], output=merged))
    assert json.loads((merged / "metadata.json").read_text())["sources"] == {
        "(unrecorded)": {"licenses": [], "prepared_rows": 2},
        "common-voice-tr": {"licenses": ["CC0-1.0"], "prepared_rows": 2},
        "yodas2-tr000": {"licenses": ["CC-BY-3.0"], "prepared_rows": 4},
    }
    merge(SimpleNamespace(inputs=[tmp_path / "cache-plain"], output=tmp_path / "plain-merged"))
    assert "sources" not in json.loads((tmp_path / "plain-merged" / "metadata.json").read_text())
    assert module.merge_source_summaries([meta["yodas"]["sources"], meta["yodas"]["sources"]]) == {
        "yodas2-tr000": {"licenses": ["CC-BY-3.0"], "prepared_rows": 8}
    }


def test_folded_codec_parameters_stay_frozen(tmp_path, monkeypatch):
    import sys

    toy = ToyCodec()
    model = nn.Module()
    model.sample_rate, model.hop_length = toy.sample_rate, toy.hop_length
    model.encoder = toy.model.encoder
    model.quantizer = nn.Module()
    model.quantizer.in_proj = toy.model.quantizer.in_proj
    model.decoder = nn.Linear(4, 4)
    model._pad = toy.model._pad
    with pytest.warns(FutureWarning):
        torch.nn.utils.weight_norm(model.encoder[0])
    monkeypatch.setitem(
        sys.modules, "dacvae", SimpleNamespace(DACVAE=SimpleNamespace(load=lambda path: model))
    )
    path = tmp_path / "fixture.weights"
    path.write_bytes(b"synthetic checkpoint identity")
    original = Codec(str(path), "cpu", encoder_only=True)
    audio = torch.randn(401)
    expected = original.encode(audio)
    folded = Codec(str(path), "cpu", encoder_only=True, fold_weight_norm=True)
    assert not any(parameter.requires_grad for parameter in folded.model.parameters())
    assert not hasattr(folded.model.encoder[0], "weight_g")
    torch.testing.assert_close(folded.encode(audio), expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="encoder-only"):
        folded.decode(expected)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("device_kind", ["cuda", "cpu"])
def test_launcher_preserves_gpu_assignment_and_skips_merge_on_failure(
    tmp_path, monkeypatch, fail, device_kind
):
    import os
    import subprocess
    import sys
    from pathlib import Path

    binary = tmp_path / "bin"
    binary.mkdir()
    stub = binary / "dacvae-tts"
    stub.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, sys
from pathlib import Path
args = sys.argv[1:]
output = Path(args[args.index('--output') + 1])
output.mkdir(parents=True, exist_ok=True)
if args[0] == 'prepare':
    rank = int(args[args.index('--shard-index') + 1])
    if os.environ.get('FAIL_PREPARE') and rank == 0:
        sys.exit(9)
    (output / 'launch.json').write_text(json.dumps({
        'device': os.environ['CUDA_VISIBLE_DEVICES'], 'args': args}))
else:
    (output / 'merged').touch()
"""
    )
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    devices = [f"GPU-{i}" for i in (7, 4, 1, 5, 0, 6, 2, 3)]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(devices))
    if device_kind == "cpu":
        monkeypatch.setenv("PREPARE_PROCESSES", "2")
        devices = ["", ""]
    if fail:
        monkeypatch.setenv("FAIL_PREPARE", "1")
    else:
        monkeypatch.delenv("FAIL_PREPARE", raising=False)
    script = Path(__file__).resolve().parents[1] / (
        "scripts/prepare_8gpu.sh" if device_kind == "cuda" else "scripts/prepare_cpu.sh"
    )
    output = tmp_path / "cache"
    result = subprocess.run(
        ["bash", str(script), "manifest.jsonl", str(output), "--batch-size", "4"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert (result.returncode != 0) == fail
    assert (output / "merged/merged").exists() != fail
    if not fail:
        for rank, device in enumerate(devices):
            launched = json.loads((output / f"part-{rank}/launch.json").read_text())
            assert launched["device"] == device
            assert launched["args"][launched["args"].index("--device") + 1] == device_kind
            assert launched["args"][-2:] == ["--batch-size", "4"]
