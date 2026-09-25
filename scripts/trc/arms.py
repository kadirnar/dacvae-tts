"""The A/B arms of the tr-combined study: one change each against run C's recipe with the #7 speed options on.

`BASE` is configs/nano_tr_w512_fast.yaml (run C's model, objective, pairs and schedule; regional compilation,
selective checkpointing, masked padding, sync-free checks). `base-eager` is run C's exact execution
(configs/nano_tr_w512.yaml) for the #7 parity check. Every other arm is BASE plus the overrides below, taken from the
single-variable configs in configs/experiments/ (#8-#11, #14 and the second-round options).
"""

BASE = "configs/nano_tr_w512_fast.yaml"
EAGER = "configs/nano_tr_w512.yaml"
# Execution on a 32 GB RTX 5090 (#7 benchmark, outputs/trc/RESULTS.md): compiled blocks, selective checkpointing and
# symbolic lengths (0.14 s/update, ~8 GB, 4 compiled graphs; run C's eager execution with full checkpointing: 0.27 s).
# Padding to multiples of 8 only: the compiled memory-efficient attention needs mask lengths aligned to 4 (unpadded
# text crashed: "attn_bias is not correctly aligned"), while pad_multiple 64 spent 20% of every frame budget on padding
# with tr-combined's short clips (13.8k vs 17.3k target frames per update; at equal frames seen the padded and eager
# runs had the same WER/CER, so the loss was data per update, not numerics). Only the execution differs.
EXECUTION = ["train.grad_checkpoint=selective", "train.compile_dynamic=auto", "train.pad_multiple=8",
             "train.text_pad_multiple=8"]

LATENT_NEGATIVES = [
    "train.contrastive_mode=latent_delta", "train.contrastive_random_weight=0.2", "train.contrastive_aug_weight=0.2",
    "train.contrastive_span_min=3", "train.contrastive_span_max=125", "train.contrastive_repeat_coverage=[0.2, 0.4]",
    "train.contrastive_skip_coverage=[0.4, 0.8]", "train.contrastive_negative_cap=0.0",
]
CROSS = ["train.cross_prompt_prob=0.4", "train.cross_prompt_max_utterances=3", "train.cross_prompt_max_seconds=12.0"]
TAIL = ["train.tail_silence_prob=0.3", "train.tail_silence_max_seconds=0.8", "train.long_prompt_prob=0.25",
        "train.prompt_fraction_long_max=0.85", "train.prompt_cut=quiet"]
REPA = ["model.repa_layer=10", "model.repa_dim=256", "train.teacher_features=teacher/mhubert147-l12-pca256",
        "train.repa_weight=1.0", "train.repa_stop_step=0", "train.repa_frames=all"]
TLA = ["model.tla_layers=all", "model.tla_dim=192", "model.tla_hidden=256",
       "train.speaker_embeddings=teacher/ecapa-speechbrain", "train.tla_weight=0.5", "train.tla_entropy=0.01"]

# name: (issue, config, overrides, files the arm needs next to the cache); BASE arms get EXECUTION prepended.
ARMS = {
    "base-s42": ("#7 baseline", BASE, [], []),
    "base-eager": ("#7", EAGER, [], []),
    "base-s43": ("seed noise floor", BASE, ["train.seed=43"], []),
    "pairs-cross": ("#11", BASE, CROSS, []),
    "latent-negatives": ("#8", BASE, LATENT_NEGATIVES, ["silence.pt"]),
    "no-negatives": ("#8", BASE, ["train.contrastive_mode=none"], []),
    "pairs-tail": ("#11", BASE, TAIL, ["silence.pt"]),
    "pairs-char-ctc": ("#11", BASE, ["model.ctc_targets=chars"], []),
    "char-units": ("round 2", BASE, ["model.text_units=chars"], []),
    "long-skip": ("#9", BASE, ["model.long_skip=true"], []),
    "value-residual": ("#9", BASE, ["model.value_residual=true"], []),
    "ffn-conv": ("#9", BASE, ["model.ffn_conv_kernel=5"], []),
    "attn-gate": ("#9", BASE, ["model.attn_gate=head"], []),
    "swiglu": ("#9", BASE, ["model.ffn_activation=swiglu"], []),
    "final-adaln": ("#9", BASE, ["model.final_adaln=true"], []),
    "cond-text-pool": ("#9", BASE, ["model.cond_text_pool=true"], []),
    "regularized": ("#14", BASE, ["model.dropout=0.1", "train.weight_decay=0.05"], []),
    "speaker-context": ("round 2", BASE, ["model.speaker_context=vector", "train.speaker_context_prob=0.5"], []),
    "repa": ("#10", BASE, REPA, ["teacher/mhubert147-l12-pca256"]),
    "tla": ("#10", BASE, TLA, ["teacher/ecapa-speechbrain"]),
    "repa-tla": ("#10", BASE, REPA + TLA, ["teacher/mhubert147-l12-pca256", "teacher/ecapa-speechbrain"]),
    "speaker-condition": ("round 2", BASE, ["model.speaker_condition_dim=192",
                                            "train.speaker_condition=teacher/ecapa-speechbrain"],
                          ["teacher/ecapa-speechbrain"]),
    "quality-cond": ("new: quality condition", BASE, ["model.quality_condition=true",
                                                     "train.quality_scores=quality/dnsmos.json"],
                     ["quality/dnsmos.json"]),
    "tempo-prompts": ("round 2", BASE, ["train.tempo_prompt_prob=0.3", "train.tempo_variants=tempo/wsola-v1"],
                      ["tempo/wsola-v1"]),
}
ARMS = {name: (issue, config, (EXECUTION + overrides) if config == BASE else overrides, needs)
        for name, (issue, config, overrides, needs) in ARMS.items()}
