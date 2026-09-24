"""Training-only teacher alignment: speech-REPA frame features and TLA-SA speaker embeddings.

Both terms read intermediate generator states (`FlowTTS.forward(return_hidden=...)`) and targets that
were precomputed once per cache row (scripts/extract_teacher_features.py, teacher.py). The heads live
on the model so that the optimizer, EMA and checkpoints include them, but the sampler never calls
them: inference cost is zero and they are only created when configured.

speech-REPA, the speech term of A-DMA (arXiv:2505.19595; F5-small, LibriTTS 585 h): CTC at block 8
plus alignment of block 12 to the last HuBERT layer took WER 2.60 -> 2.35 and SIM 0.586 -> 0.609,
main table 2.68 -> 1.97 with ~2x faster convergence; the loss is the negative cosine over every
frame, weight 1.0. BareWave (arXiv:2606.09048) reports WER 3.32 -> 2.86 with WavLM. The projector
is a k=3 convolution (iREPA, arXiv:2512.10794: convolutional projectors beat MLPs), and HASTE
(arXiv:2505.16792) stops the term at a fixed step once it plateaus (`repa_stop_step`).

TLA-SA (arXiv:2511.09995): time-averaged block states over the generated frames, one MLP per block
into a speaker-verification embedding space, cosine to the utterance embedding, blocks mixed by
softmax(MLP(time embedding)) with a negative-entropy regularizer (alpha 0.01) against collapse onto
one block; L = L_CFM + 0.5 L_TLA. F5-like LibriTTS Sim-WavLM 0.398 -> 0.458 and 0.500 -> 0.571 with
a different SV model (ERes2Net); CosyVoice 2 0.606 -> 0.644 with 2.9x faster SIM convergence, WER
within 0.1. Train on embeddings of a speaker model other than the evaluation's (WavLM-large ECAPA
SIM-o), e.g. SpeechBrain ECAPA, so that a SIM gain is not metric gaming.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import sanitize


class RepaProjector(nn.Module):
    """Packed block states [B,N,D] -> teacher frames [B,L,dim]: Conv1d(k=3) -> SiLU -> 1x1 conv.

    The last convolution emits `patch` frames per packed position, unpacked like the velocity head.
    """

    def __init__(self, width, dim, patch=1):
        super().__init__()
        self.patch = patch
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1), nn.SiLU(), nn.Conv1d(width, dim * patch, 1)
        )

    def forward(self, states, frames):
        y = self.net(states.transpose(1, 2)).transpose(1, 2)
        return y.reshape(y.size(0), y.size(1) * self.patch, -1)[:, :frames]


class SpeakerAlignment(nn.Module):
    """TLA-SA heads: one small MLP per aligned block plus the time -> block-weight network.

    The weight network starts at zero output, i.e. uniform block weights (maximum entropy). It reads
    its own sinusoidal time features, so the generator's time embedding is not shaped by this term.
    """

    def __init__(self, width, dim, blocks, hidden=256):
        super().__init__()
        self.width = width
        self.heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, dim))
                for _ in range(blocks)
            ]
        )
        self.weights = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, blocks))
        nn.init.zeros_(self.weights[-1].weight)
        nn.init.zeros_(self.weights[-1].bias)

    def forward(self, states, mask, time):
        """States: list of [B,N,D]; mask [B,N] frames to average; time [B] -> ([B,K,dim], weights [B,K])."""
        from .model import sinusoidal  # model.py imports this module

        count = mask.sum(1, keepdim=True).clamp_min(1)
        pooled = [sanitize(h, mask).sum(1) / count.to(h.dtype) for h in states]
        embeddings = torch.stack([head(x) for head, x in zip(self.heads, pooled, strict=True)], 1)
        features = sinusoidal(time * 1000, self.width).to(pooled[0].dtype)
        return embeddings, self.weights(features).float().softmax(-1)


def cosine_distance(prediction, target, eps=1e-8):
    """1 - cos along the last axis, in FP32; 0 for identical directions, 2 for opposite ones."""
    return 1 - F.cosine_similarity(prediction.float(), target.float(), dim=-1, eps=eps)


def masked_distance(prediction, target, mask):
    """Per-example mean of 1 - cos over the selected frames of [B,L,E] tensors -> [B]."""
    distance = cosine_distance(prediction, target).masked_fill(~mask, 0)
    return distance.sum(1) / mask.sum(1).clamp_min(1)


def weighted_alignment(embeddings, speaker, weights):
    """TLA-SA terms per example: sum_k w_k (1 - cos(e_k, s)) and the negative entropy sum_k w_k log w_k."""
    alignment = (weights * cosine_distance(embeddings, speaker[:, None])).sum(-1)
    return alignment, (weights * weights.clamp_min(1e-12).log()).sum(-1)


def teacher_layers(model, repa=True, tla=True):
    """Generator blocks whose outputs the enabled teacher terms read (sorted, 1-based)."""
    layers = set()
    if repa and getattr(model, "repa", None) is not None:
        layers.add(model.cfg.repa_layer)
    if tla and getattr(model, "tla", None) is not None:
        layers.update(model.cfg.tla_layers)
    return tuple(sorted(layers))


def teacher_terms(model, batch, hidden, time, drop, repa=True, tla=True, repa_frames="all"):
    """Per-example teacher terms [N] from one flow forward pass.

    Frame selection: `repa_frames: all` covers every valid frame, because prompt frames are real
    speech that the network sees clean; examples whose prompt was removed for classifier-free
    guidance (or a summary-only reference path) keep only their target frames, since the zeroed prompt
    positions carry nothing to align. TLA-SA averages the target frames only (prompt frames are clean
    copies) and skips condition-dropped examples, whose voice was removed on purpose. Heads that exist
    but are switched off (a zero weight, or after `repa_stop_step`) receive an exact zero through
    `teacher_idle`, which keeps DDP's every-parameter-has-a-gradient contract.
    """
    if getattr(model, "repa", None) is None and getattr(model, "tla", None) is None:
        return {}
    valid, prompt_mask = batch["valid"], batch["prompt_mask"]
    target = valid & ~prompt_mask
    terms = {}
    if repa and model.repa is not None:
        if "teacher" not in batch:
            raise ValueError("speech-REPA needs teacher frames in the batch; set train.teacher_features")
        teacher = batch["teacher"]
        if teacher.shape[:2] != valid.shape or teacher.size(-1) != model.cfg.repa_dim:
            raise ValueError(f"Teacher frames must be [B,L,{model.cfg.repa_dim}] like the latents")
        blind = drop if model.cfg.reference_paths != "summary" else torch.ones_like(drop)
        mask = target if repa_frames == "target" else torch.where(blind[:, None], target, valid)
        prediction = model.repa(hidden[model.cfg.repa_layer], valid.size(1))
        terms["repa"] = masked_distance(prediction, teacher, mask)
    if tla and model.tla is not None:
        if "speaker_embedding" not in batch:
            raise ValueError("TLA-SA needs utterance speaker embeddings; set train.speaker_embeddings")
        speaker = batch["speaker_embedding"]
        if speaker.shape != (valid.size(0), model.cfg.tla_dim):
            raise ValueError(f"Speaker embeddings must be [B,{model.cfg.tla_dim}]")
        p = model.cfg.patch_size
        packed = F.pad(target, (0, (-target.size(1)) % p)).reshape(target.size(0), -1, p).any(-1)
        states = [hidden[layer] for layer in model.cfg.tla_layers]
        embeddings, weights = model.tla(states, packed, time)
        alignment, negentropy = weighted_alignment(embeddings, speaker, weights)
        terms["tla"], terms["tla_entropy"] = alignment.masked_fill(drop, 0), negentropy.masked_fill(drop, 0)
    idle = [
        module
        for module, used in ((model.repa, "repa" in terms), (model.tla, "tla" in terms))
        if module is not None and not used
    ]
    if idle:
        zero = sum(p.sum() for module in idle for p in module.parameters()) * 0
        terms["teacher_idle"] = zero.float().expand(valid.size(0))
    return terms
