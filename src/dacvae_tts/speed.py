"""Opt-in training throughput options: sync-free finiteness checks.

The generator is small (66.5M) and overhead/memory bound, not FLOP bound: at 0.6 s per update an
RTX 4090 runs at roughly 9% of its bf16 peak. The gains therefore come from fewer host-device
synchronizations, fused pointwise kernels and less recomputation, not from faster matrix products.
Every option is off by default, and the defaults reproduce the previous training numerics exactly.
"""

import torch
import torch.distributed as dist


class NonfiniteWatch:
    """Sync-free replacement of `if not torch.isfinite(x): raise` (`train.strict_checks: false`).

    Every such `if` makes the host wait for the GPU, once per micro-batch for the loss and once per
    update for the gradient norm. The watch keeps the first update with a nonfinite objective or
    gradient norm on the device and raises at the next point where the host synchronizes anyway (a
    log record, validation or a checkpoint), always before anything is written, so no checkpoint
    ever holds nonfinite weights. The price: a diverged run stops up to `log_every` updates late.
    Under DDP the first offending update is reduced over ranks so that all ranks raise together.
    """

    KINDS = ("objective", "gradient")
    NEVER = torch.iinfo(torch.int64).max

    def __init__(self, device):
        self.first = torch.full((len(self.KINDS),), self.NEVER, dtype=torch.int64, device=device)

    def note(self, kind, value, step):
        slot = self.KINDS.index(kind)
        bad = ~torch.isfinite(value.detach()).all()
        self.first[slot] = torch.where(bad, self.first[slot].clamp(max=step + 1), self.first[slot])

    def check(self):
        first = self.first.clone()
        if dist.is_initialized():
            dist.all_reduce(first, op=dist.ReduceOp.MIN)
        for kind, update in zip(self.KINDS, first.tolist()):
            if update != self.NEVER:
                raise FloatingPointError(f"Nonfinite {kind} at update {update} (deferred check)")
