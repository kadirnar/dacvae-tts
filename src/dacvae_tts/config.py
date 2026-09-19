from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    latent_dim: int = 128
    width: int = 256
    depth: int = 8
    heads: int = 4
    text_depth: int = 3
    patch_size: int = 2
    ff_mult: int = 3
    cond_dropout: float = 0.1
    reference_encoder: str = "mlp"
    reference_pooling: str = "mean"
    reference_paths: str = "both"
    duration_features: str = "baseline"

    def __post_init__(self):
        if min(self.latent_dim, self.width, self.depth, self.heads, self.patch_size) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads or self.width % 2:
            raise ValueError("width must be even and divisible by heads")
        if not 0 <= self.cond_dropout < 1:
            raise ValueError("Invalid condition dropout")
        if self.reference_encoder not in {"mlp", "temporal"} or self.reference_pooling not in {
            "mean",
            "mean_std",
            "attention",
        }:
            raise ValueError("Unsupported reference encoder or pooling")
        if self.reference_paths not in {"both", "full", "summary"}:
            raise ValueError("reference_paths must be both, full or summary")
        if self.duration_features not in {"baseline", "text_stats"}:
            raise ValueError("duration_features must be baseline or text_stats")
        if self.text_depth < 1 or self.ff_mult < 1:
            raise ValueError("Text depth and feed-forward multiplier must be positive")


@dataclass
class TrainConfig:
    steps: int = 200000
    batch_size: int = 4
    accumulation: int = 4
    learning_rate: float = 3e-4
    warmup: int = 5000
    weight_decay: float = 0.01
    ema_decay: float = 0.999
    precision: str = "bf16"
    workers: int = 4
    checkpoint_every: int = 2000
    validate_every: int = 1000
    log_every: int = 50
    grad_checkpoint: bool = False
    compile: bool = False
    seed: int = 42
    flow_reduction: str = "utterance"
    duration_weight: float = 0.1
    speaker_balance: float = 0.0
    diagnostics_every: int = 0

    def __post_init__(self):
        if (
            min(
                self.steps,
                self.batch_size,
                self.accumulation,
                self.checkpoint_every,
                self.validate_every,
                self.log_every,
            )
            < 1
        ):
            raise ValueError("Training counts/intervals must be positive")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if self.workers < 0 or self.warmup < 0 or not 0 <= self.ema_decay < 1:
            raise ValueError("Invalid training configuration")
        if self.flow_reduction not in {"utterance", "frame"} or self.duration_weight < 0:
            raise ValueError("Invalid loss reduction or duration weight")
        if not 0 <= self.speaker_balance <= 1 or self.diagnostics_every < 0:
            raise ValueError("Invalid sampling or diagnostic settings")


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def load(cls, path):
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    @classmethod
    def from_dict(cls, obj):
        return cls(ModelConfig(**obj["model"]), TrainConfig(**obj.get("train", {})))

    def to_dict(self):
        return asdict(self)
