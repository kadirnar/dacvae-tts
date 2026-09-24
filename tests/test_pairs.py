"""Training pairs (issue #11): cross-utterance prompts, short targets, tail silence, quiet cuts, char CTC."""

import hashlib
import importlib.util
import json
import sqlite3
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from dacvae_tts.config import Config, ModelConfig, TrainConfig
from dacvae_tts.data import (
    SCHEMA,
    BucketBatchSampler,
    LatentDataset,
    ShardWriter,
    collate,
    load_stats,
    save_stats,
)
from dacvae_tts.model import FlowTTS, ctc_alignment_loss, flow_loss
from dacvae_tts.text import (
    BOS,
    BYTE_OFFSET,
    CHAR_VOCAB_SIZE,
    CTC_CHARS,
    EOS,
    VOCAB_SIZE,
    assemble,
    char_ctc_targets,
    ctc_text,
    encode_ids,
    join_ids,
    tokenize_bytes,
)
from dacvae_tts.training import Objective, load_model

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = 4
TEXTS = ["İstanbul'da KIRMIZI elma.", "Işık hâlâ yanıyor!", "Bugün kitap okudum", "Çok güzel, değil mi?"]
SPEAKERS = {"a": [30, 42, 25, 38], "b": [50, 20, 33]}
# Multiples of 1/8 off the 1/4 grid of the utterance frames: exact in float16, never equal to a speech frame.
SILENCE = torch.tensor([0.125, -0.375, 0.625, -0.875])
NANO = dict(
    latent_dim=4,
    width=32,
    heads=2,
    depth=2,
    text_depth=1,
    patch_size=1,
    positions="rope",
    prediction="edm",
    text_layout="joined",
    duration="rule",
    ctc_layer=1,
)
WITHIN = dict(pairing="within", layout="joined")


def build_cache(path, speakers=SPEAKERS, token_ids=True, silence=False, silence_at=None):
    """Integer-valued latents (exact in float16): digests depend on neither the platform RNG nor libm."""
    path.mkdir()
    writer = ShardWriter(path, CHANNELS, shard_bytes=2048)
    sums, squares, count, row = np.zeros(CHANNELS), np.zeros(CHANNELS), 0, 0
    with sqlite3.connect(path / "index.sqlite") as db:
        db.executescript(SCHEMA)
        for speaker, lengths in speakers.items():
            for number, frames in enumerate(lengths):
                values = (np.arange(frames * CHANNELS).reshape(frames, CHANNELS) * 5 + row * 3) % 17 - 8
                z = values.astype(np.float32) / 4
                if silence_at is not None:
                    z[silence_at(frames)] = SILENCE.numpy()
                shard, offset = writer.write(z)
                uid, text = f"{speaker}-{number}", TEXTS[row % len(TEXTS)]
                db.execute(
                    "INSERT INTO samples VALUES(NULL,?,?,?,?,?,?,?,?,?,?)",
                    (uid, speaker, text, "a.wav", shard, offset, frames, "train", frames * 100, uid),
                )
                if token_ids:
                    db.execute(
                        "INSERT INTO token_ids VALUES(?,?)", (uid, encode_ids(text.encode()).tobytes())
                    )
                sums += z.astype(np.float64).sum(0)
                squares += np.square(z.astype(np.float64)).sum(0)
                count, row = count + frames, row + 1
    writer.close()
    save_stats(path / "stats.pt", count, torch.from_numpy(sums), torch.from_numpy(squares))
    meta = dict(
        latent_dim=CHANNELS,
        sample_rate=2500,
        hop_length=100,
        posterior="mean",
        checkpoint="test-codec",
        merged=True,
    )
    (path / "metadata.json").write_text(json.dumps(meta))  # 25 frames per second, like DACVAE
    if silence:
        write_silence(path)
    return path


def write_silence(path, checkpoint="test-codec"):
    torch.save(
        {"raw": SILENCE, "frame": SILENCE, "codec": {"checkpoint": checkpoint, "latent_dim": 4}},
        path / "silence.pt",
    )


def stream_digest(cache, **options):
    digest = hashlib.sha256()
    for pairing, layout, dropout, fraction in (
        ("within", "joined", 0.3, (0.1, 0.6)),
        ("cross", "segments", 0.0, (0.1, 0.5)),
        ("cross", "joined", 0.0, (0.1, 0.5)),
    ):
        data = LatentDataset(cache, "train", 42, pairing, layout, fraction, dropout, **options)
        digest.update(data.costs.astype(np.int64).tobytes())
        for epoch in range(3):
            items = [data[(epoch, i)] for i in range(len(data))]
            for item in items:
                for key in sorted(item):
                    value = item[key]
                    digest.update(key.encode())
                    if isinstance(value, (torch.Tensor, np.ndarray)):
                        digest.update(np.ascontiguousarray(np.asarray(value)).tobytes())
                    else:
                        digest.update(repr(value).encode())
            for start in range(0, len(items), 3):
                batch = collate(items[start : start + 3])
                for key in sorted(batch):
                    digest.update(key.encode() + batch[key].numpy().tobytes())
            sampler = BucketBatchSampler(data.costs, 2, seed=42, frame_budget=180, bucket_size=4)
            sampler.epoch = epoch
            digest.update(repr(list(sampler)).encode())
    return digest.hexdigest()


def test_options_off_reproduce_the_previous_data_stream(tmp_path):
    """Digests recorded with the code before issue #11: items, batches, costs and sampler plans."""
    off = dict(cross_prompt_prob=0.0, long_prompt_prob=0.0, tail_silence_prob=0.0, prompt_cut="random")
    expected = {
        True: "6d26b29eca4dadc654025972c49a56c861869a03899ad28c2c4bc556346cc87a",
        False: "c57cc580373d146c6c1589a987d76a389ef31831346def3454030d1c3e243054",
    }
    for token_ids, digest in expected.items():
        cache = build_cache(tmp_path / f"ids{token_ids}", token_ids=token_ids)
        assert stream_digest(cache) == digest
        assert stream_digest(cache, **off, ctc_targets="bytes") == digest


def legacy_ctc(logits, token_valid, tokens, drop):
    """ctc_alignment_loss before issue #11, verbatim."""
    targets = [row[row >= BYTE_OFFSET] for row in tokens]
    lengths = torch.tensor([len(row) for row in targets], device=logits.device)
    loss = F.ctc_loss(
        logits.float().log_softmax(-1).transpose(0, 1),
        torch.cat(targets),
        token_valid.sum(1),
        lengths,
        blank=0,
        reduction="none",
        zero_infinity=True,
    )
    return (loss / lengths.clamp_min(1)).masked_fill(drop, 0)


def test_byte_ctc_is_unchanged_and_ignores_character_targets(tmp_path):
    torch.manual_seed(0)
    rows = [tokenize_bytes(b"", b"abc de", "joined")[0], tokenize_bytes(b"", b"xyzxy z", "joined")[0]]
    tokens = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
    logits, valid = torch.randn(2, 30, VOCAB_SIZE), torch.ones(2, 30, dtype=torch.bool)
    drop = torch.tensor([False, True])
    assert torch.equal(
        ctc_alignment_loss(logits, valid, tokens, drop), legacy_ctc(logits, valid, tokens, drop)
    )
    model = FlowTTS(ModelConfig(**NANO)).train()
    assert model.ctc.out_features == VOCAB_SIZE
    batch = long_batch(labels="chars")
    plain = {k: v for k, v in batch.items() if not k.startswith("ctc_")}
    torch.manual_seed(3)
    with_keys = flow_loss(model, batch, dropout=0.0, return_details=True)["ctc"]
    torch.manual_seed(3)
    assert torch.equal(with_keys, flow_loss(model, plain, dropout=0.0, return_details=True)["ctc"])


def test_cross_prompts_join_other_utterances_of_the_speaker(tmp_path):
    cache = build_cache(tmp_path / "ids", {**SPEAKERS, "solo": [40]})
    text_cache = build_cache(tmp_path / "text", {**SPEAKERS, "solo": [40]}, token_ids=False)
    options = dict(**WITHIN, cross_prompt_prob=1.0, cross_prompt_max_seconds=4.0)  # 100 frames
    data, text_data = LatentDataset(cache, **options), LatentDataset(text_cache, **options)
    index_of = {data.row(i)["uid"]: i for i in range(len(data))}
    counts = []
    for epoch in range(4):
        costs = data.epoch_costs(epoch)
        assert (costs <= data.costs).all()
        for i in range(len(data)):
            item, row = data[(epoch, i)], data.row(i)
            references = item["reference_uid"].split("|")
            if row["speaker"] == "solo":  # one utterance only: the within cut stays
                assert references == [row["uid"]]
                assert torch.equal(torch.cat([item["reference"], item["target"]]), row["latents"])
                continue
            counts.append(len(references))
            assert row["uid"] not in references and len(set(references)) == len(references) <= 3
            assert all(r.split("-")[0] == row["speaker"] for r in references)
            assert torch.equal(item["target"], row["latents"])  # the whole utterance is the target
            rows = [data.row(index_of[r]) for r in references]
            assert torch.equal(item["reference"], torch.cat([r["latents"] for r in rows]))
            assert len(item["reference"]) <= 100 and costs[i] == len(item["reference"]) + len(item["target"])
            batch = collate([item])
            text = " ".join([*(r["text"] for r in rows), row["text"]])
            assert batch["tokens"][0].tolist() == [BOS, *(v + BYTE_OFFSET for v in text.encode()), EOS]
            assert (batch["segments"] == 1).all()
            assert int(batch["prompt_mask"][0].sum()) == len(item["reference"])
            assert int(batch["valid"][0].sum()) == len(item["reference"]) + len(item["target"])
            assert torch.equal(batch["prompt"][0][batch["prompt_mask"][0]], item["reference"])
            # Caches without token ids tokenize the same joined transcript.
            assert torch.equal(collate([text_data[(epoch, i)]])["tokens"], batch["tokens"])
    assert {1, 2} <= set(counts)
    assert data[(1, 0)]["reference_uid"] == data[(1, 0)]["reference_uid"]  # deterministic per (epoch, index)
    assert len({data[(epoch, 0)]["reference_uid"] for epoch in range(6)}) > 1


def test_join_ids_matches_joined_bytes():
    parts = [encode_ids("bir".encode()), encode_ids("iki üç".encode())]
    tokens, _ = assemble(join_ids(parts), encode_ids(b"dort"), "joined")
    assert torch.equal(tokens, tokenize_bytes("bir iki üç".encode(), b"dort", "joined")[0])
    with pytest.raises(ValueError):
        join_ids([])


def test_cross_prompts_respect_the_frame_budget_dropout_and_limits(tmp_path):
    cache = build_cache(tmp_path / "c", {**SPEAKERS, "solo": [40]})
    data = LatentDataset(cache, **WITHIN, cross_prompt_prob=0.6, cross_prompt_max_seconds=3.0)
    budget = int(data.costs.max())  # the tightest budget the static bound admits
    sampler = BucketBatchSampler(
        data.costs, 4, frame_budget=budget, bucket_size=8, epoch_costs=data.epoch_costs
    )
    cross = 0
    for epoch in range(5):
        sampler.epoch = epoch
        for batch in sampler:
            items = [data[key] for key in batch]
            frames = [len(item["reference"]) + len(item["target"]) for item in items]
            assert frames == [int(data.epoch_costs(epoch)[i]) for _, i in batch]
            assert max(frames) * len(frames) <= budget
            cross += sum(item["reference_uid"] != item["uid"] for item in items)
    assert cross > 0
    with pytest.raises(ValueError, match="static bound"):
        BucketBatchSampler(data.costs, 4, epoch_costs=lambda epoch: data.costs + 1).batches()
    dropped = LatentDataset(cache, **WITHIN, prompt_dropout=1.0, cross_prompt_prob=1.0)
    assert all(len(dropped[(0, i)]["reference"]) == 0 for i in range(len(dropped)))  # dropout keeps its rate
    tight = LatentDataset(cache, **WITHIN, cross_prompt_prob=1.0, cross_prompt_max_seconds=0.5)  # 12 frames
    assert all(tight[(0, i)]["reference_uid"] == tight[(0, i)]["uid"] for i in range(len(tight)))
    with pytest.raises(ValueError):
        LatentDataset(cache, cross_prompt_prob=0.5)  # cross pairing
    with pytest.raises(ValueError):
        LatentDataset(cache, **WITHIN, long_prompt_prob=0.5, prompt_fraction_long_max=0.3)


def test_long_prompts_cover_short_targets(tmp_path):
    cache = build_cache(tmp_path / "c")
    base = LatentDataset(cache, **WITHIN, prompt_fraction=(0.1, 0.5))
    long = LatentDataset(cache, **WITHIN, prompt_fraction=(0.1, 0.5), long_prompt_prob=1.0)
    half = LatentDataset(cache, **WITHIN, prompt_fraction=(0.1, 0.5), long_prompt_prob=0.5)
    assert (long.costs == base.costs).all()
    kinds = set()
    for epoch in range(6):
        for i in range(len(base)):
            frames = int(base.lengths[i])
            fraction = len(long[(epoch, i)]["reference"]) / frames
            assert 0.5 - 0.5 / frames <= fraction <= 0.85 + 0.5 / frames
            plain, mixed = base[(epoch, i)], half[(epoch, i)]
            same = torch.equal(plain["reference"], mixed["reference"])
            mixed_fraction = len(mixed["reference"]) / frames
            assert same or 0.5 - 0.5 / frames <= mixed_fraction <= 0.85 + 0.5 / frames
            kinds.add(same)
    assert kinds == {True, False}  # unselected rows keep exactly their baseline cut


def test_tail_silence_appends_the_silence_frame(tmp_path):
    cache = build_cache(tmp_path / "c", silence=True)
    base = LatentDataset(cache, **WITHIN)
    data = LatentDataset(cache, **WITHIN, tail_silence_prob=1.0, tail_silence_max_seconds=0.4)  # 10 frames
    assert (data.costs == base.costs + 10).all() and (data.epoch_costs(0) == data.costs).all()
    silence = (SILENCE - data.mean) / data.std
    seen = set()
    for epoch in range(4):
        for i in range(len(data)):
            item, plain = data[(epoch, i)], base[(epoch, i)]
            n = item["tail_silence"]
            assert 1 <= n <= 10 and len(item["target"]) == len(plain["target"]) + n
            assert torch.equal(item["reference"], plain["reference"]) and torch.equal(
                item["target"][:-n], plain["target"]
            )
            assert torch.equal(item["target"][-n:], silence.expand(n, -1))
            seen.add(n)
    assert len(seen) > 3
    batch = collate([data[(0, 0)], data[(0, 1)]])
    targets = (batch["valid"] & ~batch["prompt_mask"]).sum(1)  # the loss covers the padding
    assert targets.tolist() == [len(data[(0, 0)]["target"]), len(data[(0, 1)]["target"])]
    some = LatentDataset(cache, **WITHIN, tail_silence_prob=0.5)
    padded = [some[(0, i)]["tail_silence"] for i in range(len(some))]
    assert 0 in padded and max(padded) > 0
    cross = LatentDataset(cache, tail_silence_prob=1.0)  # cross pairing pads its targets too
    assert (cross.costs == LatentDataset(cache).costs + 20).all() and cross[(0, 0)]["tail_silence"] >= 1
    (cache / "silence.pt").unlink()
    with pytest.raises(FileNotFoundError, match="silence_latent.py"):
        LatentDataset(cache, **WITHIN, tail_silence_prob=0.3)
    write_silence(cache, checkpoint="another-codec")
    with pytest.raises(ValueError, match="another codec"):
        LatentDataset(cache, **WITHIN, prompt_cut="quiet")


def test_quiet_cut_ends_the_prompt_on_the_silence_closest_frame(tmp_path):
    cache = build_cache(tmp_path / "c", silence=True, silence_at=lambda frames: frames // 3)
    base = LatentDataset(cache, **WITHIN, prompt_fraction=(0.1, 0.6))
    quiet = LatentDataset(cache, **WITHIN, prompt_fraction=(0.1, 0.6), prompt_cut="quiet")
    assert quiet.quiet_window == 8 and (quiet.costs == base.costs).all()
    hits = 0
    for epoch in range(8):
        for i in range(len(base)):
            latents = base.row(i)["latents"]
            frames, sampled = len(latents), len(base[(epoch, i)]["reference"])
            item = quiet[(epoch, i)]
            cut = len(item["reference"])
            window = range(max(sampled - 9, 0), min(sampled + 7, frames - 2) + 1)
            distance = (latents - quiet.silence).square().sum(-1)
            best = min(window, key=lambda f: (float(distance[f]), abs(f - sampled + 1)))
            assert cut == best + 1 and 1 <= cut < frames and abs(cut - sampled) <= 8
            assert torch.equal(torch.cat([item["reference"], item["target"]]), latents)
            if frames // 3 in window:
                assert torch.equal(item["reference"][-1], quiet.silence)  # the prompt ends in the pause
                hits += 1
    assert hits > 10
    dropped = LatentDataset(cache, **WITHIN, prompt_dropout=1.0, prompt_cut="quiet")
    assert len(dropped[(0, 0)]["reference"]) == 0


def test_character_ctc_text():
    assert ctc_text("İstanbul'da KIRMIZI elma!") == "istanbulda kırmızı elma"
    assert ctc_text("IŞIK ılık, İĞNE") == "ışık ılık iğne"
    assert ctc_text("Hâlâ kâr   ediyor... 3 kez") == "hala kar ediyor kez"
    assert ctc_text("Café-bar: ÇÖĞÜŞ") == "cafe bar çöğüş"
    assert ctc_text("İzmir") == "izmir"  # decomposed dotted capital I
    assert ctc_text("!!! 42") == ""
    assert CHAR_VOCAB_SIZE == 34 and CTC_CHARS[0] == " "


def decode(row, length):
    return "".join(CTC_CHARS[int(v) - 1] for v in row[:length])


def test_character_ctc_targets_from_model_tokens():
    joined, _ = tokenize_bytes(b"", "İstanbul'da KIRMIZI.".encode(), "joined")
    segments, _ = tokenize_bytes("Merhaba.".encode(), "Dünya!".encode())
    padded = torch.nn.utils.rnn.pad_sequence([joined, segments], batch_first=True)
    targets, lengths = char_ctc_targets(padded)
    assert decode(targets[0], lengths[0]) == "istanbulda kırmızı"
    assert decode(targets[1], lengths[1]) == "merhaba dünya"  # reference and target words, one space
    assert (targets[1, lengths[1] :] == 0).all() and targets.shape == (2, int(lengths.max()))
    assert int(lengths[0]) < int((joined >= BYTE_OFFSET).sum())  # fewer labels than bytes
    cross, _ = assemble(
        join_ids([encode_ids(b"bir"), encode_ids(b"iki")]), encode_ids("ÜÇ".encode()), "joined"
    )
    targets, lengths = char_ctc_targets([cross])
    assert decode(targets[0], lengths[0]) == "bir iki üç"


def long_batch(labels="bytes", frames=64):
    torch.manual_seed(0)
    items = []
    for index, prompt in enumerate((5, 0)):
        latents = torch.randn(frames + 3 * index, 4)
        item = dict(reference=latents[:prompt], target=latents[prompt:], reference_text="", layout="joined")
        item["text"] = ["İstanbul'da KIRMIZI elma.", "Işık hâlâ yanıyor!"][index]
        if labels == "chars":
            item["ctc_targets"] = "chars"
        items.append(item)
    return collate(items)


def test_character_ctc_head_and_loss():
    model = FlowTTS(ModelConfig(**NANO, ctc_targets="chars")).train()
    assert model.ctc.out_features == CHAR_VOCAB_SIZE
    batch = long_batch("chars")
    assert batch["ctc_targets"].shape[0] == 2 and batch["ctc_target_lengths"].tolist() == [23, 17]
    torch.manual_seed(1)
    ctc = flow_loss(model, batch, dropout=0.0, return_details=True)["ctc"]
    assert ctc.shape == (2,) and torch.isfinite(ctc).all() and (ctc > 0).all()
    plain = {k: v for k, v in batch.items() if not k.startswith("ctc_")}
    torch.manual_seed(1)
    assert torch.equal(
        flow_loss(model, plain, dropout=0.0, return_details=True)["ctc"], ctc
    )  # built on the fly
    losses = Objective(model, expansion=2, ctc_weight=0.1).train()(batch)
    assert losses["ctc"].shape == (4,)
    (losses["loss"].mean() + losses["ctc"].mean()).backward()
    assert model.ctc.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="mix"):
        collate(
            [
                dict(
                    reference=torch.zeros(0, 4),
                    target=torch.ones(3, 4),
                    text="a",
                    reference_text="",
                    layout="joined",
                    ctc_targets="chars",
                ),
                dict(
                    reference=torch.zeros(0, 4),
                    target=torch.ones(3, 4),
                    text="b",
                    reference_text="",
                    layout="joined",
                ),
            ]
        )


def test_dataset_flags_character_targets(tmp_path):
    cache = build_cache(tmp_path / "c")
    data = LatentDataset(cache, **WITHIN, ctc_targets="chars")
    batch = collate([data[(0, i)] for i in range(3)])
    expected = char_ctc_targets(batch["tokens"])
    assert torch.equal(batch["ctc_targets"], expected[0]) and torch.equal(
        batch["ctc_target_lengths"], expected[1]
    )
    assert "ctc_targets" not in collate([LatentDataset(cache, **WITHIN)[(0, 0)]])


def test_old_checkpoints_load_strictly_and_heads_differ(tmp_path):
    config = Config(ModelConfig(**NANO), TrainConfig(pairing="within")).to_dict()
    del config["model"]["ctc_targets"]
    for name in ("cross_prompt_prob", "long_prompt_prob", "tail_silence_prob", "prompt_cut"):
        del config["train"][name]
    model = FlowTTS(ModelConfig(**NANO))
    torch.save(
        {"config": config, "model": model.state_dict(), "ema": model.state_dict()}, tmp_path / "old.pt"
    )
    loaded, _ = load_model(tmp_path / "old.pt")  # strict load_state_dict
    assert loaded.ctc.out_features == VOCAB_SIZE and loaded.cfg.ctc_targets == "bytes"
    chars = FlowTTS(ModelConfig(**NANO, ctc_targets="chars"))
    assert chars.ctc.weight.shape == (CHAR_VOCAB_SIZE, 32)
    with pytest.raises(RuntimeError):
        model.load_state_dict(chars.state_dict())


def test_configuration_guards():
    with pytest.raises(ValueError):
        TrainConfig(pairing="within", cross_prompt_prob=1.5)
    with pytest.raises(ValueError):
        TrainConfig(cross_prompt_prob=0.4)  # the default pairing is cross
    with pytest.raises(ValueError):
        TrainConfig(
            pairing="within", prompt_fraction_max=0.9, long_prompt_prob=0.2, prompt_fraction_long_max=0.85
        )
    with pytest.raises(ValueError):
        TrainConfig(pairing="within", prompt_cut="edge")
    with pytest.raises(ValueError):
        TrainConfig(pairing="within", cross_prompt_max_utterances=0)
    TrainConfig(tail_silence_prob=0.3)  # tail silence also pads cross-pairing targets
    TrainConfig(pairing="within", prompt_fraction_max=0.9)  # long max is only checked when enabled
    with pytest.raises(ValueError):
        ModelConfig(ctc_targets="chars")  # no CTC head
    with pytest.raises(ValueError):
        ModelConfig(**{**NANO, "ctc_targets": "phones"})


def test_training_runs_with_every_pair_option(cache, tmp_path):
    from dacvae_tts import training

    torch.save({"raw": torch.zeros(4), "codec": {"checkpoint": "test-codec"}}, cache / "silence.pt")
    config = tmp_path / "pairs.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {**NANO, "ctc_targets": "chars"},
                "train": {
                    "steps": 3,
                    "warmup": 1,
                    "batch_size": 2,
                    "accumulation": 2,
                    "workers": 0,
                    "precision": "fp32",
                    "log_every": 1,
                    "checkpoint_every": 3,
                    "validate_every": 3,
                    "pairing": "within",
                    "ctc_weight": 0.1,
                    "cross_prompt_prob": 0.5,
                    "cross_prompt_max_seconds": 0.5,
                    "long_prompt_prob": 0.5,
                    "tail_silence_prob": 0.5,
                    "prompt_cut": "quiet",
                },
            }
        )
    )
    names = "steps batch_size accumulation workers precision learning_rate optimizer worker_threads".split()
    names += "prefetch_factor loader_start_method cuda_prefetch compile stop_after init_from resume".split()
    args = types.SimpleNamespace(
        **dict.fromkeys(names),
        config=str(config),
        cache=str(cache),
        output=str(tmp_path / "run"),
        device="cpu",
        frame_budget=120,
        no_validation=False,
    )
    training.train(args)
    _, saved = load_model(tmp_path / "run" / "last.pt")
    assert saved["step"] == 3 and Config.from_dict(saved["config"]).train.prompt_cut == "quiet"
    records = [json.loads(line) for line in (tmp_path / "run" / "train.jsonl").read_text().splitlines()]
    assert all(np.isfinite(r["ctc"]) for r in records if "ctc" in r)


class FakeCodec:
    """Stands in for DACVAE: a level-dependent frame with padding artifacts at both edges."""

    def __init__(self, checkpoint, device, encoder_only=False, loudness=None):
        self.checkpoint, self.sample_rate, self.hop_length = checkpoint, 2500, 100

    @property
    def metadata(self):
        return dict(
            checkpoint=self.checkpoint, sample_rate=2500, hop_length=100, latent_dim=4, posterior="mean"
        )

    def encode(self, audio):
        frames = len(audio) // self.hop_length
        level = audio[: frames * self.hop_length].reshape(frames, -1).std(1, keepdim=True)
        z = SILENCE.repeat(frames, 1) + level
        z[0] = z[-1] = 9.0
        return z


def test_silence_latent_script(tmp_path, monkeypatch):
    from dacvae_tts import codec

    cache = build_cache(tmp_path / "c")
    monkeypatch.setattr(codec, "Codec", FakeCodec)
    spec = importlib.util.spec_from_file_location("silence_latent", ROOT / "scripts" / "silence_latent.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    with pytest.warns(UserWarning, match="Legacy metadata"):
        script.main(["--cache", str(cache), "--device", "cpu", "--probe", "3"])
    saved = torch.load(cache / "silence.pt", weights_only=True)
    stats = load_stats(cache)
    assert torch.allclose(saved["raw"], SILENCE, atol=1e-3)  # edges trimmed, -80 dBFS noise negligible
    assert torch.allclose(saved["frame"], (saved["raw"] - stats["mean"]) / stats["std"])
    assert set(saved["report"]["cache_frame_distance_quantiles"]) == {"p1", "p5", "p25", "p50"}
    data = LatentDataset(cache, **WITHIN, tail_silence_prob=1.0)
    assert torch.allclose(data.silence, saved["frame"])
    with pytest.raises(SystemExit):
        script.main(["--cache", str(cache), "--device", "cpu"])  # never overwrites silently
