import pickle
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader

from dacvae_tts.data import LatentDataset, collate
from dacvae_tts.fast_codec import CudaGraphPool, ExactSnake2d, convert
from dacvae_tts.parallel import device_batches, loader_options
from dacvae_tts.prepare import ordered_prefetch, prepare_record


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("length", [16, 17, 31])
def test_conversion_preserves_geometry_weights_and_rng(transpose, length):
    factory = nn.ConvTranspose1d if transpose else nn.Conv1d
    original = factory(4, 6, 5, stride=2, padding=2, dilation=2, groups=2).eval()
    rng = torch.get_rng_state()
    fast = convert(original)
    assert torch.equal(rng, torch.get_rng_state())
    assert not any(p.requires_grad for p in fast.parameters())
    x = torch.randn(2, 4, length)
    expected = original(x)
    result = fast(x.unsqueeze(2).to(memory_format=torch.channels_last)).squeeze(2)
    torch.testing.assert_close(result, expected)


def test_exact_snake_handles_large_amplitudes():
    alpha = torch.rand(1, 8, 1) + 0.1
    original = SimpleNamespace(alpha=alpha)
    x = torch.linspace(-30, 30, 2 * 8 * 21).reshape(2, 8, 21)
    expected = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).square()
    torch.testing.assert_close(ExactSnake2d(original)(x.unsqueeze(2)).squeeze(2), expected, atol=0, rtol=0)


def test_conversion_rejects_unsupported_or_unfolded_layers():
    with pytest.raises(ValueError, match="Unsupported"):
        convert(nn.ReLU())
    padded = nn.Conv1d(2, 2, 3, padding=1, padding_mode="reflect")
    with pytest.raises(ValueError, match="zero-padding"):
        convert(padded)
    with pytest.warns(FutureWarning):
        normalized = nn.utils.weight_norm(nn.Conv1d(2, 2, 3))
    with pytest.raises(ValueError, match="Fold"):
        convert(normalized)


def test_spawn_audio_preprocessing_matches_threads_and_serial(tmp_path):
    rows = []
    for i in range(6):
        path = tmp_path / f"{i}.wav"
        sf.write(path, np.random.default_rng(i).normal(0, 0.1, (800 + i, 2)), 16000)
        rows.append(dict(id=str(i), audio=path.name, text=" Unicode café. ", speaker_id="s", split="train"))
    args = SimpleNamespace(
        text_column="text",
        audio_column="audio",
        speaker_column="speaker_id",
        min_seconds=0.01,
        max_seconds=10,
        seed=42,
    )
    function = partial(prepare_record, args=args, root=tmp_path, sample_rate=8000)
    expected = list(ordered_prefetch(function, rows, 0, 2))
    for backend in ("thread", "process"):
        result = list(ordered_prefetch(function, rows, 2, 3, backend))
        for want, got in zip(expected, result):
            torch.testing.assert_close(got.pop("audio"), want["audio"], atol=0, rtol=0)
            assert got == {k: v for k, v in want.items() if k != "audio"}


def test_spawn_dataset_after_parent_opened_sqlite_and_mmaps(cache):
    dataset = LatentDataset(cache)
    expected = collate([dataset[(3, 0)], dataset[(3, 1)]])
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._db is None and not restored._maps
    loader = DataLoader(
        restored,
        batch_sampler=[[(3, 0), (3, 1)]],
        collate_fn=collate,
        **loader_options(2, "cpu", prefetch_factor=1),
    )
    for key, tensor in next(iter(loader)).items():
        torch.testing.assert_close(tensor, expected[key], atol=0, rtol=0)


def test_cpu_transfer_and_loader_validation():
    batches = [{"x": torch.arange(i + 1)} for i in range(3)]
    result = list(device_batches(batches, "cpu"))
    assert len(result) == 3
    for want, got in zip(batches, result):
        torch.testing.assert_close(want["x"], got["x"])
    assert loader_options(0, "cpu") == dict(num_workers=0, pin_memory=False)
    with pytest.raises(ValueError, match="spawn or forkserver"):
        loader_options(1, "cuda", start_method="fork")
    with pytest.raises(ValueError, match="positive"):
        loader_options(1, "cpu", prefetch_factor=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_replay_uses_fresh_inputs_and_owned_outputs():
    pool = CudaGraphPool(lambda x: x.sin() * 2 + x, max_shapes=1, warmup_calls=2)
    with torch.inference_mode():
        x = torch.randn(3, 5, device="cuda")
        first = pool(x)
        captured = pool(x)
        y = torch.randn_like(x)
        changed = pool(y)
        torch.testing.assert_close(changed, y.sin() * 2 + y)
        torch.testing.assert_close(captured, first, atol=0, rtol=0)
        assert not torch.equal(changed, first)
        other = torch.randn(2, 7, device="cuda")
        for _ in range(4):
            torch.testing.assert_close(pool(other), other.sin() * 2 + other)
    assert len(pool.graphs) == 1 and pool.hits == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_prefetch_values_and_rng_are_unchanged():
    batches = [{"x": torch.randn(i + 1, 13).pin_memory()} for i in range(4)]
    rng = torch.cuda.get_rng_state()
    result = [batch["x"].square().cpu() for batch in device_batches(batches, "cuda")]
    assert torch.equal(rng, torch.cuda.get_rng_state())
    for want, got in zip(batches, result):
        torch.testing.assert_close(want["x"].square(), got)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_cache_separates_autocast_precision():
    pool = CudaGraphPool(lambda x: x @ x.T, max_shapes=2, warmup_calls=1)
    with torch.inference_mode():
        x = torch.randn(8, 8, device="cuda")
        full = pool(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reduced = pool(x)
            torch.testing.assert_close(reduced, x @ x.T)
        assert full.dtype == torch.float32 and reduced.dtype == torch.bfloat16
        torch.testing.assert_close(pool(x), full, atol=0, rtol=0)
    assert len(pool.graphs) == 2
