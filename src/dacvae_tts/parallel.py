"""Bounded CPU data workers and optional one-batch-ahead CUDA transfers."""

from functools import partial

import torch

from .data import move_batch


def initialize_worker(worker_id=0, threads=1):
    # PyTorch's intra-op pool otherwise multiplies across ranks and workers.
    torch.set_num_threads(threads)


def loader_options(workers, device, prefetch_factor=2, worker_threads=1, start_method="spawn"):
    if workers < 0 or prefetch_factor < 1 or worker_threads < 1:
        raise ValueError("workers must be nonnegative; prefetch_factor and worker_threads must be positive")
    if start_method not in {"spawn", "forkserver"}:
        raise ValueError("Use spawn or forkserver to avoid forking an initialized CUDA runtime")
    result = dict(num_workers=workers, pin_memory=torch.device(device).type == "cuda")
    if workers:
        result.update(
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=start_method,
            worker_init_fn=partial(initialize_worker, threads=worker_threads),
        )
    return result


def device_batches(batches, device, enabled=True):
    """Overlap H2D with compute, only inside an already-counted accumulation window.

    Host batches stay alive throughout transfers. No future training window is
    consumed, preserving exact checkpoint/resume offsets. CUDA tensors are recorded
    on the consumer stream before the next copy is queued on the transfer stream.
    """
    device = torch.device(device)
    if not enabled or device.type != "cuda":
        for batch in batches:
            yield move_batch(batch, device)
        return
    if not batches:
        return
    stream = torch.cuda.Stream(device=device)
    consumer = torch.cuda.current_stream(device)
    stream.wait_stream(consumer)
    with torch.cuda.stream(stream):
        pending = move_batch(batches[0], device)
    try:
        for index in range(len(batches)):
            consumer.wait_stream(stream)
            current = pending
            for tensor in current.values():
                tensor.record_stream(consumer)
            if index + 1 < len(batches):
                with torch.cuda.stream(stream):
                    pending = move_batch(batches[index + 1], device)
            yield current
    finally:
        consumer.wait_stream(stream)
