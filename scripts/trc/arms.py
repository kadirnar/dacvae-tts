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

# Round 1c: after pairs-cross won by a wide margin (20k WER 12.0 vs 30.3 for the plain recipe), every further option is
# tested on top of it: "x-<arm>" = BASE + EXECUTION + CROSS + the arm's own change, against pairs-cross (seed 42) and
# x-s43 (its seed-43 twin, the noise floor of the new base).
ROUND2 = ("latent-negatives", "quality-cond", "no-negatives", "pairs-tail", "pairs-char-ctc", "char-units", "long-skip",
          "value-residual", "ffn-conv", "attn-gate", "swiglu", "final-adaln", "cond-text-pool", "regularized",
          "speaker-context", "repa", "tla", "repa-tla", "speaker-condition")
ARMS["x-s43"] = ("noise floor of base+cross", BASE, EXECUTION + CROSS + ["train.seed=43"], [])
for _name in ROUND2:
    _issue, _config, _overrides, _needs = ARMS[_name]
    _own = [o for o in _overrides if o not in EXECUTION]
    ARMS[f"x-{_name}"] = (f"{_issue}, on base+cross", BASE, EXECUTION + CROSS + _own, _needs)

# Full-length model candidates (60k updates, the whole LR schedule): the answer to "is it better than run C?".
ARMS["full-cross"] = ("model candidate: run C recipe + cross prompts, 60k", BASE, EXECUTION + CROSS, [])

# #14 (c) model-guidance fine-tune of the full-length model: 8k updates from full-cross 60k (LR 2e-4, EMA 0.9995 without
# warm-up, no text hinge, which model guidance cannot combine with). The w=0 control is the same fine-tune without
# guidance, sampled with CFG 5, so extra updates cannot pass for a model-guidance gain. Configs: derive_config.py of
# configs/experiments/tr_w512_model_guidance_ft.yaml + EXECUTION + CROSS (runs/configs/mg-ft-w*.yaml).
FINETUNE = {
    "ft-mg-w07": dict(init="/workspace/runs/trc-full-cross/step-0060000.pt", guidance="1.0", steps="8000"),
    "ft-w0": dict(init="/workspace/runs/trc-full-cross/step-0060000.pt", guidance="5.0", steps="8000"),
}
ARMS["ft-mg-w07"] = ("#14 model guidance w=0.7 (sampled at g=1)", "/workspace/runs/configs/mg-ft-w0.7.yaml", [], [])
ARMS["ft-w0"] = ("#14 control: same fine-tune, w=0 (sampled at g=5)", "/workspace/runs/configs/mg-ft-w0.0.yaml", [], [])

# Post-training jobs run through the queue as exclusive jobs (their own script; ready once their input exists).
POSTTRAIN = {"grpo": dict(script="scripts/trc/run_grpo.sh", init="/workspace/runs/trc-full-cross/step-0060000.pt")}
ARMS["grpo"] = ("#16 Flow-GRPO from full-cross 60k", "", [], [])

# Second training seed of the best x-arm (character units beat both base seeds at 20k): the strict test of #4's rule.
ARMS["x-char-units-s43"] = ("round 2 char units, training seed 43", BASE,
                            EXECUTION + CROSS + ["model.text_units=chars", "train.seed=43"], [])

# Second full-length model candidate: every option with a clear 20k win on top of cross prompts. Quality condition
# (requested quality 4.0/4.5/3.8: DNSMOS +0.135 at equal WER), speech-REPA (WER 10.5 at 10k vs 29.6-36.7), character
# units (11.1/7.4 at 20k, below both base seeds). Its 20k snapshot is the interaction check against the single arms.
QUALITY = ["model.quality_condition=true", "train.quality_scores=quality/dnsmos.json", "model.quality_target=[4.0, 4.5, 3.8]"]
ARMS["full-v2"] = ("model candidate v2: cross + quality + REPA + char units, 60k", BASE,
                   EXECUTION + CROSS + QUALITY + REPA + ["model.text_units=chars"],
                   ["quality/dnsmos.json", "teacher/mhubert147-l12-pca256"])

# Round 3 (26 September): A/Bs on the full-v2 recipe. The architecture audit showed that on base+cross a zero-init
# option (identical initial function and data order) still moved the 20k WER by ~5 points: that WER mostly records
# when alignment happened. With speech-REPA the text aligns within ~5k updates (quick WER 11 at 5k vs ~95), so the
# 20k scores should measure the options instead. "y-<option>" = the full-v2 recipe + the option, stopped at 20k of
# the same 60k schedule; y-base is full-v2's own 20k snapshot (identical training up to there), y-s43 its seed twin.
V2 = EXECUTION + CROSS + QUALITY + REPA + ["model.text_units=chars"]
V2_NEEDS = ["quality/dnsmos.json", "teacher/mhubert147-l12-pca256"]
ARMS["y-s43"] = ("noise floor of the v2 recipe", BASE, V2 + ["train.seed=43"], V2_NEEDS)
for _name in ("long-skip", "value-residual", "ffn-conv", "final-adaln", "cond-text-pool", "swiglu", "attn-gate",
              "speaker-condition"):
    _issue, _config, _overrides, _needs = ARMS[_name]
    _own = [o for o in _overrides if o not in EXECUTION]
    ARMS[f"y-{_name}"] = (f"{_issue}, on v2", BASE, V2 + _own, V2_NEEDS + [n for n in _needs if n not in V2_NEEDS])
# Weight decay only on matrices (audit: 'all' also shrinks norm gains, biases and gates), alone and as #14's
# regularization (dropout 0.1, weight decay 0.05 on matrices).
ARMS["y-decay-matrices"] = ("audit: weight decay on matrices only, on v2", BASE,
                            V2 + ["train.weight_decay_scope=matrices"], V2_NEEDS)
ARMS["y-regularized"] = ("#14 dropout 0.1 + weight decay 0.05 on matrices, on v2", BASE,
                         V2 + ["model.dropout=0.1", "train.weight_decay=0.05", "train.weight_decay_scope=matrices"],
                         V2_NEEDS)

# Third full-length model candidate: the v2 recipe plus the round-3 winner (value residual: DNSMOS/UTMOS above both
# v2 seeds at 24 parameters) at width 640 / 10 heads (104M parameters vs 67M). The other round-3 options gave no gain
# or traded quality for SIM-o (ffn-conv, speaker condition). Width: DiTTo-TTS scaling; tr-combined shows no
# overfitting at 67M (full-v2's validation flow still falls at 60k).
ARMS["full-v3"] = ("model candidate v3: v2 + value residual, width 640 (104M), 60k", BASE,
                   V2 + ["model.value_residual=true", "model.width=640", "model.heads=10"], V2_NEEDS)
