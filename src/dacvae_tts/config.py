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
    positions: str = "absolute"
    qk_norm: bool = False
    text_attention: int = 0
    prediction: str = "velocity"
    text_layout: str = "segments"
    duration: str = "head"
    ctc_layer: int = 0
    adaln_rank: int = 0  # 0: one D->9D modulation per block; r>0: shared modulation + rank-r per block
    # DiT block options (issue #9), all off by default: off, the model and its state_dict are unchanged.
    long_skip: bool = False  # input embedding -> output head skip, h_L + Linear0(LN([h_0, h_L])); +~2D^2
    value_residual: bool = False  # self-attention v_l <- l1 v_l + l2 v_1 (ResFormer); 2 scalars per block
    ffn_conv_kernel: int = 0  # odd k > 0: residual depthwise time conv on the FFN hidden units (5 suggested)
    attn_gate: str = "none"  # head: per-head 2*sigmoid output gate on generator self-/cross-attention
    ffn_activation: str = "gelu"  # swiglu: generator FFN as SwiGLU at equal parameters (hidden 2/3 of GELU's)

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
        if self.text_depth < 1 or self.ff_mult < 1 or self.text_attention < 0:
            raise ValueError("Text depth and feed-forward multiplier must be positive")
        if self.positions not in {"absolute", "rope"} or self.prediction not in {"velocity", "edm"}:
            raise ValueError("positions must be absolute or rope; prediction must be velocity or edm")
        if self.positions == "rope" and (self.width // self.heads) % 2:
            raise ValueError("Rotary positions need an even attention head width")
        if self.text_layout not in {"segments", "joined"} or self.duration not in {"head", "rule"}:
            raise ValueError("text_layout must be segments or joined; duration must be head or rule")
        if self.adaln_rank < 0:
            raise ValueError("adaln_rank must be nonnegative")
        if not 0 <= self.ctc_layer <= self.depth:
            raise ValueError("ctc_layer must be 0 (off) or the index of a generator block")
        if self.text_layout == "joined" and self.duration == "head":
            raise ValueError("The duration head needs separate transcripts; use duration: rule when joined")
        if self.ffn_conv_kernel < 0 or (self.ffn_conv_kernel and self.ffn_conv_kernel % 2 == 0):
            raise ValueError("ffn_conv_kernel must be 0 (off) or odd, so the convolution stays centred")
        if self.attn_gate not in {"none", "head"}:
            raise ValueError("attn_gate must be none or head")
        if self.ffn_activation not in {"gelu", "swiglu"}:
            raise ValueError("ffn_activation must be gelu or swiglu")


@dataclass
class TrainConfig:
    steps: int = 200000
    batch_size: int = 4
    accumulation: int = 4
    learning_rate: float = 3e-4
    warmup: int = 5000
    weight_decay: float = 0.01
    optimizer: str = "muon"
    muon_momentum: float = 0.95
    ema_decay: float = 0.999
    precision: str = "bf16"
    workers: int = 4
    worker_threads: int = 1
    prefetch_factor: int = 2
    loader_start_method: str = "spawn"
    cuda_prefetch: bool = True
    checkpoint_every: int = 2000
    validate_every: int = 1000
    log_every: int = 50
    grad_checkpoint: bool = False
    compile: object = False  # False, True (whole objective) or "model" (generator only)
    seed: int = 42
    flow_reduction: str = "utterance"
    duration_weight: float = 0.1
    speaker_balance: float = 0.0
    diagnostics_every: int = 0
    pairing: str = "cross"
    prompt_fraction_min: float = 0.1
    prompt_fraction_max: float = 0.5
    prompt_dropout: float = 0.0
    time_sampling: str = "uniform"
    batch_expansion: int = 1
    keep_every: int = 0
    ctc_weight: float = 0.0
    contrastive_weight: float = 0.0  # skip/repeat text negatives (RobustSpeechFlow-style hinge)
    contrastive_margin: float = 0.1  # required loss gap, relative to the positive loss
    wandb_project: str = ""  # set (or pass --wandb-project) to mirror the JSONL logs to Weights & Biases

    def __post_init__(self):
        if self.worker_threads < 1 or self.prefetch_factor < 1:
            raise ValueError("worker_threads and prefetch_factor must be positive")
        if self.loader_start_method not in {"spawn", "forkserver"}:
            raise ValueError("loader_start_method must be spawn or forkserver")
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
        if self.optimizer not in {"muon", "adamw"} or not 0 <= self.muon_momentum < 1:
            raise ValueError("optimizer must be muon or adamw, with Muon momentum in [0,1)")
        if self.workers < 0 or self.warmup < 0 or not 0 <= self.ema_decay < 1:
            raise ValueError("Invalid training configuration")
        if self.flow_reduction not in {"utterance", "frame"} or self.duration_weight < 0:
            raise ValueError("Invalid loss reduction or duration weight")
        if not 0 <= self.speaker_balance <= 1 or self.diagnostics_every < 0:
            raise ValueError("Invalid sampling or diagnostic settings")
        if self.pairing not in {"cross", "within"} or self.time_sampling not in {"uniform", "logit_normal"}:
            raise ValueError("pairing must be cross or within; time_sampling uniform or logit_normal")
        if (
            not 0 <= self.prompt_fraction_min <= self.prompt_fraction_max < 1
            or not 0 <= self.prompt_dropout <= 1
        ):
            raise ValueError("Invalid prompt fraction range or prompt dropout")
        if self.compile not in {False, True, "model"}:
            raise ValueError("compile must be false, true or model")
        if self.contrastive_weight < 0 or self.contrastive_margin < 0:
            raise ValueError("contrastive settings must be nonnegative")
        if self.batch_expansion < 1 or self.keep_every < 0 or self.ctc_weight < 0:
            raise ValueError("batch_expansion must be positive and keep_every nonnegative")


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def load(cls, path):
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def __post_init__(self):
        if self.train.pairing == "within" and self.model.text_layout != "joined":
            raise ValueError("Within-utterance prompts need model.text_layout: joined")

    @classmethod
    def from_dict(cls, obj):
        return cls(ModelConfig(**obj["model"]), TrainConfig(**obj.get("train", {})))

    def to_dict(self):
        return asdict(self)
