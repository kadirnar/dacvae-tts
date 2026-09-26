import warnings
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
    final_adaln: bool = False  # output LayerNorm shift/scale from the condition (rank adaln_rank, zero-init)
    # condition += Linear0(masked mean of the segment-1 byte encodings); +D^2. Segment 1 is the target text in
    # the `segments` layout, the whole joined stream (prompt transcript + target) in `joined`.
    cond_text_pool: bool = False
    # CTC labels of the auxiliary head. `chars`: Turkish lower-case letters + space, no punctuation (34
    # classes with the blank). A two-byte Turkish letter is one phone but two byte labels, and fast speakers
    # (16-19 bytes/s against 25 fps) leave byte CTC barely feasible; zero_infinity then zeroes those examples.
    ctc_targets: str = "bytes"
    # Text input units. `chars`: one token per character (text.UNIT_CHARACTERS): ASCII keeps its byte id, so the
    # vocabulary and embedding shape stay the byte model's, but the Turkish two-byte letters (ç ğ ı ö ş ü and
    # capitals, ~12 % of letters) are one token instead of two: an even length-aware RoPE diagonal, one CTC label
    # per letter, and a character duration rule at inference. Converted from the cached byte ids at load time.
    text_units: str = "bytes"
    # Frozen speaker-verification embedding (e.g. SpeechBrain ECAPA 192-d) added to the voice condition through a
    # zero-init, bias-free Linear (~0.1M): a speaker prior from ~7k-200k training speakers next to the in-context
    # prompt (Koel-TTS unseen SIM: in-context 0.637, SV vector 0.619; MiniMax-Speech: encoder + prompt 0.746 vs prompt
    # 0.726, a frozen SV vector raised WER). 0 is off. Training reads a speaker store (train.speaker_condition).
    speaker_condition_dim: int = 0
    # Multi-clip speaker context (`vector`): a small transformer over the latents of further recordings of the voice
    # (other utterances of the speaker in training, extra clips + the prompt at inference; no positions, so clip order
    # does not matter) -> attention pooling -> a zero-init vector added to the voice condition. Irodori-TTS: CAM++
    # SIM one clip 0.661 -> 30 s 0.752 -> 120 s 0.775 (766M, per-block speaker K/V); XTTS averages its reference
    # latents over clips; Koel-TTS saw a cross-attention encoder overfit seen speakers, hence one vector. ~2.2M
    # parameters at the defaults (+3.4 % at width 512); starts as the baseline, so it can warm-start from a checkpoint.
    speaker_context: str = "none"
    speaker_context_width: int = 256
    speaker_context_layers: int = 3
    speaker_context_heads: int = 4
    speaker_context_patch: int = 4  # latent frames per context token (25 fps -> 6.25 tokens/s)
    # Quality condition: the target utterance's DNSMOS P.835 [SIG, BAK, OVRL], centred at 3 and scaled by 1/0.5, enters
    # the voice condition through a zero-init, bias-free Linear (3 x width). Found-data corpora mix clean and noisy
    # recordings; labelling the recording quality lets the model learn from all of them and still be asked for clean
    # speech (QA-MDT, arXiv:2405.15863: a quality token on a quality-imbalanced corpus, p-MOS 3.80 -> 4.05, FAD
    # 5.76 -> 5.20; Lyth & King, arXiv:2402.01912: SNR/reverberation labels give high-fidelity TTS from found data).
    # Training reads a per-utterance score store (train.quality_scores); inference asks for `quality_target`. The
    # guidance null branch drops it with the voice, so CFG also steers toward the requested quality.
    quality_condition: bool = False
    quality_target: tuple = (3.6, 4.1, 3.3)  # SIG, BAK, OVRL requested at inference (about the corpus's top decile)
    # Training-only teacher heads (alignment.py); absent from the module and its checkpoints when off.
    repa_layer: int = 0  # speech-REPA: block predicting teacher SSL frames; 0 off, 10-11 with CTC at 8
    repa_dim: int = 0  # width of the stored teacher frames (after the extraction PCA, e.g. 256)
    tla_layers: object = ()  # TLA-SA: blocks aligned to the utterance speaker embedding; (), "all" or a list
    tla_dim: int = 0  # width of the stored speaker embeddings (192 for SpeechBrain ECAPA)
    tla_hidden: int = 256  # width of the per-block heads and of the time -> block-weight network
    # Residual-branch dropout in the generator blocks (attention outputs, FFN hidden activation); F5-TTS's
    # DiT uses 0.1. Parameter-free, so checkpoints load with any value; 0 draws no random numbers at all.
    dropout: float = 0.0

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
        if self.text_units not in {"bytes", "chars"}:
            raise ValueError("text_units must be bytes or chars")
        if self.ctc_targets not in {"bytes", "chars"} or (self.ctc_targets == "chars" and not self.ctc_layer):
            raise ValueError("ctc_targets must be bytes or chars; chars needs a CTC head (ctc_layer > 0)")
        if self.ctc_layer and self.patch_size > 1 and self.ctc_targets == "bytes":
            # CTC needs at least one input frame per label. The head reads packed frames: DACVAE's 25 latent
            # frames/s become 12.5/s at patch_size 2, below the 16-19 bytes/s of normal speech, so most examples
            # are infeasible and zero_infinity silently zeroes their loss. A warning: such configs still load.
            warnings.warn(
                f"ctc_layer with patch_size {self.patch_size} and byte targets: the CTC head sees "
                f"{25 / self.patch_size:g} packed frames/s against 16-19 transcript bytes/s, so zero_infinity "
                "zeroes most examples; use ctc_targets: chars or patch_size: 1",
                stacklevel=3,
            )
        self.tla_layers = teacher_blocks(self.tla_layers, self.depth)
        if not 0 <= self.repa_layer <= self.depth or (self.repa_layer and self.repa_dim < 1):
            raise ValueError("repa_layer must be 0 (off) or a generator block, with a positive repa_dim")
        if self.repa_layer and self.repa_layer == self.ctc_layer:
            # A-DMA (arXiv:2505.19595) ablation: CTC and speech alignment on one block is worse than either
            # split (WER 2.69 vs 2.35 with CTC at 8, SSL at 12); text alignment wants the earlier block.
            raise ValueError("repa_layer must differ from ctc_layer; align speech after the CTC block")
        if self.tla_layers and (self.tla_dim < 1 or self.tla_hidden < 1):
            raise ValueError("tla_layers need positive tla_dim and tla_hidden")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        if self.speaker_condition_dim < 0:
            raise ValueError("speaker_condition_dim must be 0 (off) or the embedding width")
        self.quality_target = tuple(float(v) for v in self.quality_target)  # YAML lists vs tuples on resume
        if len(self.quality_target) != 3 or not all(1 <= v <= 5 for v in self.quality_target):
            raise ValueError("quality_target must be three DNSMOS values [SIG, BAK, OVRL] in [1, 5]")
        if self.speaker_context not in {"none", "vector"}:
            raise ValueError("speaker_context must be none or vector")
        if self.speaker_context != "none" and (
            min(self.speaker_context_width, self.speaker_context_layers, self.speaker_context_heads,
                self.speaker_context_patch) < 1
            or self.speaker_context_width % self.speaker_context_heads
        ):
            raise ValueError("speaker_context needs positive width/layers/heads/patch, width divisible by heads")


def teacher_blocks(value, depth):
    """TLA-SA block selection as a sorted tuple: () off, "all" every block, or explicit 1-based indices.

    Normalized so that YAML lists, JSON lists and checkpointed tuples compare equal on resume/warm start.
    """
    if value in ("", None, "none"):
        return ()
    if value == "all":
        return tuple(range(1, depth + 1))
    if isinstance(value, str) or not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        raise ValueError('tla_layers must be "all" or a list of block indices')
    blocks = tuple(sorted(set(value)))
    if blocks and not 1 <= blocks[0] <= blocks[-1] <= depth:
        raise ValueError("tla_layers must index generator blocks 1..depth")
    return blocks


@dataclass
class TrainConfig:
    steps: int = 200000
    batch_size: int = 4
    accumulation: int = 4
    learning_rate: float = 3e-4
    warmup: int = 5000
    weight_decay: float = 0.01
    # `all` decays every parameter (every model trained so far); `matrices` leaves vectors, scalars and embedding
    # tables undecayed (optim.decay_exempt: norm gains, biases, gates, value-residual weights)
    weight_decay_scope: str = "all"
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
    grad_checkpoint: object = False  # false, true (every block), selective, or N (every N-th block)
    compile: object = False  # False, True (whole objective), "model" (generator only) or "blocks"
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
    # Throughput options (speed.py). Every default reproduces the previous training numerics exactly.
    strict_checks: bool = True  # false: skip value checks that stall the host (shapes still checked)
    pad_multiple: int = 1  # round padded batch frames up to a multiple (masked; bounds compiled shapes)
    text_pad_multiple: int = 1  # the same for transcript tokens
    loader_negatives: bool = False  # draw the text_hinge negatives in the loader workers (inert otherwise)
    compile_dynamic: str = "batch"  # compile: blocks -- batch (only batch dim symbolic) or auto
    # Negatives. text_hinge: the transcript hinge above (contrastive_weight/margin; one more text encoding
    # and generator pass, ~20-25% compute). latent_delta: RobustSpeechFlow/ΔFM corrupted target latents
    # with the correct text, L_pos - λ_rand L_rand - λ_aug L_aug, target-only (dacvae_tts.negatives).
    contrastive_mode: str = "text_hinge"  # text_hinge | latent_delta | none
    contrastive_random_weight: float = 0.2  # latent_delta: λ_rand, another utterance's target latents
    contrastive_aug_weight: float = 0.2  # latent_delta: λ_aug, spans repeated or skipped in the target
    contrastive_span_min: int = 3  # latent_delta edit span in frames (paper: 0.1-5 s; 25 fps)
    contrastive_span_max: int = 125
    contrastive_repeat_coverage: tuple = (0.2, 0.4)  # share of the target a repeat negative overwrites
    contrastive_skip_coverage: tuple = (0.4, 0.8)  # share a skip negative removes (tail -> silence)
    contrastive_negative_cap: float = 0.0  # 0: plain subtraction; >0: distance <= cap x target gap
    # Training pairs (issue #11). All off by default, which reproduces the previous data stream exactly.
    # Cross-utterance prompts (VoiceStar, arXiv:2505.19462: continuation-only WER 8.49 -> mixed 6.42, SIM flat):
    # in `within` pairing a prompted item uses, with this probability, 1..max other utterances of its speaker
    # label (joined transcripts, at most max seconds) as the prompt and its whole utterance as the target.
    cross_prompt_prob: float = 0.0
    cross_prompt_max_utterances: int = 3
    cross_prompt_max_seconds: float = 12.0
    # Short targets: with this probability the within cut fraction is drawn from
    # [prompt_fraction_max, prompt_fraction_long_max] instead of [prompt_fraction_min, prompt_fraction_max].
    long_prompt_prob: float = 0.0
    prompt_fraction_long_max: float = 0.85
    # Tail silence (Irodori v4.1: fixing over-long durations CER 5.35 -> 4.69): append 1..max seconds of the
    # encoded-silence latent (<cache>/silence.pt, scripts/silence_latent.py) to targets; the loss covers it.
    tail_silence_prob: float = 0.0
    tail_silence_max_seconds: float = 0.8
    # `quiet` moves each within cut to the silence-closest frame within +-0.3 s (needs silence.pt), so
    # prompts end in a pause like inference prompts do instead of mid-word.
    prompt_cut: str = "random"
    # Prompt tempo perturbation (VoiceStar, arXiv:2505.19462: +cross prompts 6.42 -> 5.66 WER, SIM -0.004): with
    # this probability the prompt is the same speech at another tempo, from a tempo-variant store (WSOLA-stretched
    # audio re-encoded by scripts/build_tempo_variants.py; relative paths resolve against the cache). Training then
    # shows prompt/target rate mismatches, the break the length-aware RoPE prior meets whenever the duration rule
    # does not reproduce the prompt's rate. `tempo_prompt_factors` picks stored tempos (e.g. [0.8, 0.9, 1.0, 1.111,
    # 1.25]; [] = all; 1.0 is the re-encoded round trip); `tempo_prompt_pairs`: all (within cuts and cross prompts)
    # or cross (cross prompts only, as VoiceStar). Stretched prompts carry no REPA teacher frames.
    tempo_prompt_prob: float = 0.0
    tempo_variants: str = ""
    tempo_prompt_factors: tuple = ()
    tempo_prompt_pairs: str = "all"
    # Teacher-feature auxiliary losses (alignment.py, scripts/extract_teacher_features.py); stores are
    # sidecar directories, relative paths resolve against the cache directory.
    teacher_features: str = ""  # speech-REPA frame store (e.g. teacher/mhubert147-l12-pca256)
    repa_weight: float = 0.0  # A-DMA used 1.0; 0.5-1.0
    repa_stop_step: int = 0  # HASTE (arXiv:2505.16792): align only during the first N updates; 0 = always
    repa_frames: str = "all"  # all valid frames (prompt frames are real speech too) or target frames only
    speaker_embeddings: str = ""  # TLA-SA utterance speaker-embedding store (e.g. teacher/ecapa-speechbrain)
    tla_weight: float = 0.0  # TLA-SA used 0.5
    # Speaker-embedding condition (model.speaker_condition_dim > 0): utterance embeddings of a speaker store built by
    # scripts/extract_teacher_features.py speakers (train and val splits). `other` conditions a within-utterance item
    # on another utterance of its speaker label (the target never leaks into its own condition, and inference
    # embeds the prompt, not the target), `same` on its own; `min_cosine` falls back to the item's own embedding when
    # the other utterance is further away (diarization label noise). Must not be the TLA-SA store.
    speaker_condition: str = ""
    speaker_condition_source: str = "other"
    speaker_condition_min_cosine: float = 0.0
    # Speaker context (model.speaker_context): with this probability an item gets a context of other utterances of
    # its label, whole clips in random order up to a length drawn from [min, max] seconds (at most max_utterances;
    # below min: none), drawn independently of prompt dropout (prompt-free rows then learn from the context alone).
    # The other items keep an empty context, so single-clip inference stays in distribution (Irodori v4's single clip
    # fell from 0.678 to 0.661 when every item had a long reference).
    speaker_context_prob: float = 0.0
    speaker_context_min_seconds: float = 3.0
    speaker_context_max_seconds: float = 30.0
    speaker_context_max_utterances: int = 8
    tla_entropy: float = 0.01  # weight of the negative entropy of the time-dependent block weights
    # Quality condition (model.quality_condition): JSON {uid: [SIG, BAK, OVRL]} (relative paths resolve against the
    # cache, e.g. quality/dnsmos.json from scripts/trc/quality_store.py). Rows without scores and a `quality_dropout`
    # share of the others get the zero vector ("unknown quality", MOS 3), so unlabelled inference still works.
    quality_scores: str = ""
    quality_dropout: float = 0.1
    # Schedule and regularization options (issue #14); every default reproduces the original recipe exactly.
    # wsd: warmup, constant LR, then a decay over the last `decay_fraction` of the updates. A 20% 1-sqrt
    # cooldown matches cosine and any stable-phase checkpoint can branch into a cooldown (Hägele et al.,
    # arXiv:2405.18392); Echo-TTS and Irodori train with Muon + WSD.
    lr_schedule: str = "cosine"  # cosine (warmup + cosine to min_lr_ratio) or wsd
    decay_fraction: float = 0.2  # wsd: share of all updates in the final decay
    decay_shape: str = "1-sqrt"  # wsd: 1-sqrt or linear decay
    min_lr_ratio: float = 0.1  # final LR / peak LR of either schedule (0.1: the original cosine floor)
    # wsd: from the decay start on, train on this second merged cache (e.g. the hq subset), MiniCPM's
    # (arXiv:2404.06395) switch to high-quality data in the decay: the warm-started stages in one run.
    decay_cache: object = None
    # Time sampling from `final_time_sampling_start` on (a fraction of `steps`, or "decay": the wsd decay
    # start). BareWave (arXiv:2606.09048), logit-normal -> uniform late: SIM 0.522 -> 0.543,
    # UTMOS 3.70 -> 3.82, WER flat (2.86 -> 2.93).
    final_time_sampling: object = None  # null, uniform or logit_normal
    final_time_sampling_start: object = "decay"
    # More EMA tracks, saved next to `ema` (which keeps `ema_decay`) as `ema_<decay>`, validated and loadable
    # with load_model(..., ema=<decay>). The best EMA length depends on the run and on CFG (EDM2,
    # arXiv:2312.02696); 0.9999 is too slow for <15k-update fine-tunes. `ema_decay` itself may be listed.
    ema_decays: list = field(default_factory=list)
    # EMA warm-up: every track (`ema` and `ema_decays`) averages with min(decay, (1 + step) / (10 + step)),
    # so a track's own decay only applies from update ~9k (0.999), ~18k (0.9995) or ~90k (0.9999) on and
    # until then all tracks are identical; on --init-from it also discards the warm-started EMA within ~10
    # updates. false: every track uses exactly its decay from the first update, starting from the warm-start
    # checkpoint's EMA; needs --init-from (a random initialization would dominate the average).
    ema_warmup: bool = True
    # Model guidance (arXiv:2502.12154; on F5-TTS arXiv:2504.20334): the target becomes
    # v + w sg(v_cond - v_null) from the model's own predictions; sample without CFG (--guidance 1). The fixed
    # point bakes in CFG scale 1 / (1 - w) (w 0.5 ~ 2, 0.7 ~ 3.3); w >= 1 diverges. One extra no-grad forward
    # per update (~+30%); meant for fine-tuning a trained checkpoint with --init-from.
    model_guidance_weight: float = 0.0

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
        if self.weight_decay_scope not in {"all", "matrices"}:
            raise ValueError("train.weight_decay_scope must be all or matrices")
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
        if self.compile not in {False, True, "model", "blocks"}:
            raise ValueError("compile must be false, true, model or blocks")
        if self.contrastive_weight < 0 or self.contrastive_margin < 0:
            raise ValueError("contrastive settings must be nonnegative")
        if self.batch_expansion < 1 or self.keep_every < 0 or self.ctc_weight < 0:
            raise ValueError("batch_expansion must be positive and keep_every nonnegative")
        interval = isinstance(self.grad_checkpoint, int) and not isinstance(self.grad_checkpoint, bool)
        if not (
            isinstance(self.grad_checkpoint, bool)
            or self.grad_checkpoint == "selective"
            or (interval and self.grad_checkpoint >= 1)
        ):
            raise ValueError("grad_checkpoint must be false, true, selective or a positive block interval")
        multiples = (self.pad_multiple, self.text_pad_multiple)
        if any(not isinstance(m, int) or isinstance(m, bool) or m < 1 for m in multiples):
            raise ValueError("pad_multiple and text_pad_multiple must be positive integers")
        if self.compile_dynamic not in {"batch", "auto"}:
            raise ValueError("compile_dynamic must be batch or auto")
        if self.contrastive_mode not in {"text_hinge", "latent_delta", "none"}:
            raise ValueError("contrastive_mode must be text_hinge, latent_delta or none")
        weights = (self.contrastive_random_weight, self.contrastive_aug_weight)
        if min(*weights, self.contrastive_negative_cap) < 0:
            raise ValueError("latent negative weights and cap must be nonnegative")
        if sum(weights) >= 1 and not self.contrastive_negative_cap:
            # Below 1 each frame's objective is a convex quadratic in the prediction: bounded below.
            raise ValueError("Uncapped latent negative weights must sum below 1")
        if not 1 <= self.contrastive_span_min <= self.contrastive_span_max:
            raise ValueError("latent negative spans need 1 <= contrastive_span_min <= contrastive_span_max")
        # YAML gives lists, defaults are tuples: normalize so resume's config comparison matches.
        self.contrastive_repeat_coverage = tuple(self.contrastive_repeat_coverage)
        self.contrastive_skip_coverage = tuple(self.contrastive_skip_coverage)
        for pair in (self.contrastive_repeat_coverage, self.contrastive_skip_coverage):
            if len(pair) != 2 or not 0 < pair[0] <= pair[1] < 1:
                raise ValueError("latent negative coverages must be [low, high] with 0 < low <= high < 1")
        if not all(
            0 <= p <= 1 for p in (self.cross_prompt_prob, self.long_prompt_prob, self.tail_silence_prob)
        ):
            raise ValueError("cross_prompt_prob, long_prompt_prob and tail_silence_prob must lie in [0,1]")
        if self.cross_prompt_max_utterances < 1 or self.cross_prompt_max_seconds <= 0:
            raise ValueError("cross_prompt_max_utterances and cross_prompt_max_seconds must be positive")
        if self.long_prompt_prob and not self.prompt_fraction_max <= self.prompt_fraction_long_max < 1:
            raise ValueError("Need prompt_fraction_max <= prompt_fraction_long_max < 1")
        if self.tail_silence_max_seconds <= 0 or self.prompt_cut not in {"random", "quiet"}:
            raise ValueError("tail_silence_max_seconds must be positive; prompt_cut random or quiet")
        if self.pairing != "within" and (
            self.cross_prompt_prob or self.long_prompt_prob or self.prompt_cut != "random"
        ):
            raise ValueError("cross_prompt_prob, long_prompt_prob and prompt_cut: quiet need within pairing")
        self.tempo_prompt_factors = tuple(self.tempo_prompt_factors)
        if not 0 <= self.tempo_prompt_prob <= 1 or self.tempo_prompt_pairs not in {"all", "cross"}:
            raise ValueError("tempo_prompt_prob must lie in [0,1]; tempo_prompt_pairs all or cross")
        if not all(isinstance(f, (int, float)) and not isinstance(f, bool) and 0.5 <= f <= 2.0
                   for f in self.tempo_prompt_factors) or len(set(self.tempo_prompt_factors)) != len(
                self.tempo_prompt_factors):
            raise ValueError("tempo_prompt_factors must be distinct tempo factors in [0.5, 2.0]")
        if self.tempo_prompt_prob and (not self.tempo_variants or self.pairing != "within"):
            raise ValueError("tempo_prompt_prob needs a tempo_variants store and within pairing")
        if self.tempo_prompt_prob and self.tempo_prompt_pairs == "cross" and not self.cross_prompt_prob:
            raise ValueError("tempo_prompt_pairs: cross stretches cross prompts only; set cross_prompt_prob > 0")
        if min(self.repa_weight, self.repa_stop_step, self.tla_weight, self.tla_entropy) < 0:
            raise ValueError("Teacher loss weights and repa_stop_step must be nonnegative")
        if not 0 <= self.speaker_context_prob <= 1 or self.speaker_context_max_utterances < 1 or not (
            0 < self.speaker_context_min_seconds <= self.speaker_context_max_seconds
        ):
            raise ValueError("speaker_context_prob in [0,1], positive max_utterances, 0 < min_seconds <= max_seconds")
        if not 0 <= self.quality_dropout <= 1:
            raise ValueError("quality_dropout must lie in [0,1]")
        if self.speaker_context_prob and self.pairing != "within":
            raise ValueError("speaker_context_prob draws contexts for within pairing")
        if self.speaker_condition_source not in {"other", "same"} or not -1 <= self.speaker_condition_min_cosine <= 1:
            raise ValueError("speaker_condition_source must be other or same; min_cosine in [-1,1]")
        if self.speaker_condition and self.tla_weight and self.speaker_condition == self.speaker_embeddings:
            # The TLA heads could read the injected vector back from the hidden states and satisfy their loss.
            raise ValueError("speaker_condition must be another store than the TLA-SA speaker_embeddings")
        if self.repa_frames not in {"all", "target"}:
            raise ValueError("repa_frames must be all or target")
        if (self.repa_weight and not self.teacher_features) or (
            self.tla_weight and not self.speaker_embeddings
        ):
            raise ValueError("repa_weight needs teacher_features and tla_weight needs speaker_embeddings")
        if self.lr_schedule not in {"cosine", "wsd"} or self.decay_shape not in {"linear", "1-sqrt"}:
            raise ValueError("lr_schedule must be cosine or wsd; decay_shape linear or 1-sqrt")
        if not 0 < self.decay_fraction <= 1 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("decay_fraction must lie in (0,1] and min_lr_ratio in [0,1]")
        if self.lr_schedule == "wsd" and self.steps - round(self.steps * self.decay_fraction) < self.warmup:
            raise ValueError("The wsd decay would start inside the warmup; lower decay_fraction or warmup")
        if self.decay_cache is not None and (
            not isinstance(self.decay_cache, str) or self.lr_schedule != "wsd"
        ):
            raise ValueError("decay_cache is a cache path and needs lr_schedule: wsd")
        start = self.final_time_sampling_start
        if self.final_time_sampling not in {None, "uniform", "logit_normal"} or not (
            start == "decay"
            or (isinstance(start, (int, float)) and not isinstance(start, bool) and 0 < start < 1)
        ):
            raise ValueError("final_time_sampling: null, uniform or logit_normal; start: decay or in (0,1)")
        if self.final_time_sampling is not None and start == "decay" and self.lr_schedule != "wsd":
            raise ValueError("final_time_sampling_start: decay needs lr_schedule: wsd; give a fraction")
        if (
            not isinstance(self.ema_decays, list)
            or len(set(self.ema_decays)) != len(self.ema_decays)
            or not all(isinstance(d, float) and 0 <= d < 1 for d in self.ema_decays)
        ):
            raise ValueError("ema_decays must be a list of distinct decays in [0,1)")
        if not isinstance(self.ema_warmup, bool):
            raise ValueError("ema_warmup must be true or false")
        if not 0 <= self.model_guidance_weight < 1:
            raise ValueError("model_guidance_weight must lie in [0,1); w >= 1 diverges")
        if self.model_guidance_weight and self.contrastive_mode == "text_hinge" and self.contrastive_weight:
            # The hinge would compare the loss against the guided target with a plain-target negative (whose
            # guided target would need one more null pass under the wrong text). latent_delta composes with
            # model guidance instead: its negative targets get the same guidance offset (training.Objective).
            raise ValueError("model_guidance_weight cannot be combined with the text_hinge contrastive_weight")


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
        if (self.train.repa_weight and not self.model.repa_layer) or (
            self.train.tla_weight and not self.model.tla_layers
        ):
            raise ValueError("repa_weight needs model.repa_layer and tla_weight needs model.tla_layers")
        if (self.model.speaker_context != "none") != bool(self.train.speaker_context_prob):
            raise ValueError("model.speaker_context and train.speaker_context_prob > 0 go together")
        if self.model.quality_condition != bool(self.train.quality_scores):
            raise ValueError("model.quality_condition and train.quality_scores (a score store) go together")
        if bool(self.model.speaker_condition_dim) != bool(self.train.speaker_condition):
            raise ValueError("model.speaker_condition_dim and train.speaker_condition (a speaker store) go together")
        if self.train.model_guidance_weight and not self.model.cond_dropout:
            raise ValueError("Model guidance needs model.cond_dropout > 0 to learn the null prediction")

    @classmethod
    def from_dict(cls, obj):
        return cls(ModelConfig(**obj["model"]), TrainConfig(**obj.get("train", {})))

    def to_dict(self):
        return asdict(self)
