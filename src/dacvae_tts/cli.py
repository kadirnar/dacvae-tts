import argparse
import json


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main():
    parser = argparse.ArgumentParser(description="Compact DACVAE flow TTS")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect", help="Count trainable TTS parameters; excludes frozen codec")
    p.add_argument("--config", default="configs/tiny.yaml")

    p = sub.add_parser("prepare", help="Stream JSONL/Parquet audio into a latent-cache partition")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--codec", default="facebook/dacvae-watermarked")
    add_codec_args(p)
    p.add_argument("--device", default="cuda")
    p.add_argument("--text-column", default="text")
    p.add_argument("--audio-column", default="audio")
    p.add_argument("--speaker-column", default="speaker_id")
    p.add_argument(
        "--text-normalization",
        choices=["unicode-v1", "english-explicit-v2", "turkish-v1", "turkish-v2"],
        default="unicode-v1",
        help="Transcript normalization stored in the cache; turkish-v2 = turkish-v1 plus 'dört' softening "
        "(4'e -> dörde) and T.C./A.Ş./Ltd. Şti./Dr./vb. expansion (scripts/renormalize_cache.py converts a cache)",
    )
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--max-seconds", type=float, default=15.0)
    p.add_argument(
        "--loudness",
        type=float,
        help="Normalize every waveform to this integrated loudness in LUFS (DACVAE's own API uses -16)",
    )
    p.add_argument("--quality-column", help="Numeric per-row quality column used by --min-quality")
    p.add_argument("--min-quality", type=float, help="Reject rows whose quality column is below this value")
    p.add_argument("--reject-digits", action="store_true", help="Reject transcripts that contain digits")
    p.add_argument(
        "--languages",
        default="en",
        help="Comma-separated accepted `language` tags (default English variants), or `any`",
    )
    p.add_argument(
        "--workers", type=int, default=4, help="CPU audio-loading workers per codec process; 0 is serial"
    )
    p.add_argument("--worker-backend", choices=["thread", "process"], default="thread")
    p.add_argument(
        "--worker-threads", type=positive_int, default=1, help="Torch threads per spawned CPU worker"
    )
    p.add_argument("--prefetch", type=positive_int, default=16, help="Maximum queued CPU loading tasks")
    p.add_argument("--batch-size", type=positive_int, default=8, help="Maximum recordings per codec forward")
    p.add_argument(
        "--bucket-size",
        type=positive_int,
        default=256,
        help="Bounded lookahead window for equal hop-rounded length batches",
    )
    p.add_argument(
        "--batch-seconds",
        type=float,
        default=120.0,
        help="Maximum total hop-padded audio seconds in one encoder batch",
    )
    p.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="BF16 is experimental; compare codec reconstruction before corpus use",
    )
    p.add_argument(
        "--no-fold-weight-norm",
        action="store_true",
        help="Keep the original encoder weight-normalization hooks for comparison",
    )
    add_split_key_arg(p)
    add_partition_args(p)

    p = sub.add_parser("merge", help="Merge partitions, deduplicate, check splits and calculate statistics")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--drop-uids", help="File with uids (JSON list or one per line) to exclude from the merged cache")
    p.add_argument(
        "--drop-conflicting-duplicates",
        action="store_true",
        help="Drop (instead of failing on) identical audio that reappears with different labels",
    )
    p.add_argument(
        "--keep-singletons",
        action="store_true",
        help="Keep speakers with one recording (usable only with within-utterance pairing)",
    )
    add_split_key_arg(p, " Re-splits every row while merging (no re-encoding).")
    p.add_argument(
        "--split-map",
        help="JSON {speaker label or split key: train|val|test} applied while merging, e.g. split_map.json "
        "of scripts/speaker_clusters.py (one split per cross-episode voice cluster); labels without an "
        "entry fall back to --split-key, else keep their split",
    )

    p = sub.add_parser("train", help="Pretrain from scratch; launch with torchrun for DDP")
    p.add_argument("--config", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p.add_argument("--init-from", help="Warm-start model/EMA weights from a checkpoint (fresh schedule)")
    p.add_argument("--wandb-project", help="Mirror training/validation logs to this Weights & Biases project")
    p.add_argument("--wandb-group", help="Optional W&B group name")
    p.add_argument("--wandb-id", help="W&B run id to continue (default: the output directory name)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu"])
    p.add_argument("--steps", type=positive_int)
    p.add_argument("--batch-size", type=positive_int)
    p.add_argument("--accumulation", type=positive_int)
    p.add_argument("--workers", type=int)
    add_loader_args(p)
    p.add_argument("--cuda-prefetch", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--precision", choices=["fp32", "bf16"])
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--optimizer", choices=["muon", "adamw"], help="Default comes from the config (muon)")
    p.add_argument(
        "--frame-budget",
        type=int,
        default=0,
        help="Maximum padded prompt+target frames per rank/microbatch; 0 disables",
    )
    p.add_argument(
        "--compile",
        nargs="?",
        const="objective",
        choices=["objective", "model", "blocks"],
        help="Compile the whole objective, only the generator (model) or each generator block (blocks)",
    )
    p.add_argument("--no-validation", action="store_true", help="For smoke tests only")
    p.add_argument(
        "--stop-after", type=positive_int, help="Gracefully checkpoint early without changing LR schedule"
    )

    p = sub.add_parser("infer", help="Synthesize using a complete reference utterance and transcript")
    add_inference_args(p)
    p.add_argument("--reference", "--ref-audio", dest="reference", required=True)
    p.add_argument(
        "--reference-text", help="Optional; omitted transcripts use ASR, not a transcript-free TTS model"
    )
    p.add_argument("--asr-model", default="small.en")
    p.add_argument("--asr-device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--asr-language", default="en", help="Whisper language code of the reference audio")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--text", required=True)
    p.add_argument("--seconds", type=float)
    p.add_argument("--duration-scale", type=float, default=1.0)
    p.add_argument(
        "--duration-mode",
        choices=["rule", "clamp", "syllable", "predictor", "auto", "articulation"],
        default="rule",
        help="Target length for rule-duration models: prompt rate per byte, the same with fast prompts slowed, "
        "per syllable, the fitted duration predictor, auto (rule/clamp/predictor by prompt rate), or articulation "
        "(syllables per second of the prompt's speaking time without edge silence and pauses >= 200 ms, plus "
        "0.15 s per internal comma and 0.3 s per internal sentence end; see dacvae_tts/duration.py)",
    )
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser(
        "audit-cache",
        help="Read-only audit of speaker balance, splits, supplied sessions/intervals and optional latents",
    )
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scan-latents", action="store_true")

    p = sub.add_parser(
        "codec-reconstruct", help="Reconstruct original audio through the production codec and normalization"
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=positive_int, default=32)
    p.add_argument("--cache-precision", choices=["float16", "float32"], default="float16")
    add_codec_args(p)

    p = sub.add_parser(
        "make-cases", help="Freeze distinct original-audio reference/target pairs for evaluation"
    )
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--limit", type=positive_int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cross-session", action="store_true")
    p.add_argument(
        "--exclude-speakers",
        help="Speaker labels never used in cases (JSON list, JSON object keys such as leakage.json of "
        "scripts/speaker_clusters.py, or one per line), so cases use only truly unseen speakers",
    )

    p = sub.add_parser(
        "run-eval", help="Run a bounded, configured duration/sampler/profile comparison on frozen cases"
    )
    p.add_argument("--config", required=True)

    p = sub.add_parser("candidates", help="Generate a pool for evaluation or offline preferences")
    add_inference_args(p)
    add_partition_args(p)
    p.add_argument("--cache", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--limit", type=positive_int, default=1000)
    p.add_argument("--candidates", type=positive_int, default=4)
    p.add_argument("--duration-scale", type=float, default=1.0)

    p = sub.add_parser("evaluate", help="Score synthesized audio; optional external frozen judges")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--asr-model", default="large-v3")
    p.add_argument("--language", default="en", help="Whisper language code for scoring")
    p.add_argument("--dnsmos-model", help="Path to official non-personalized sig_bak_ovr.onnx")
    p.add_argument("--speaker-model", default="microsoft/wavlm-base-plus-sv")
    p.add_argument("--no-speaker", action="store_true")
    p.add_argument(
        "--metric-normalization",
        choices=["english-unicode-v2", "legacy-ascii-v1", "turkish-v1", "turkish-v2"],
        help="WER/CER text normalization (default: turkish-v1 for --language tr, else english-unicode-v2); "
        "turkish-v2 also reads Roman numerals, clock times, units and acronyms and folds â/î/û",
    )
    p.add_argument(
        "--reference-kind",
        choices=["original", "codec"],
        help="Protocol v2: what the manifest's reference_audio is (original recording -> sim_o, codec "
        "resynthesis -> sim_r); rows may instead carry original_prompt_audio / codec_prompt_audio",
    )
    from .eval_protocol import add_protocol_args

    add_protocol_args(p)

    p = sub.add_parser(
        "compare", help="Paired before/after metrics with speaker-clustered bootstrap intervals"
    )
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap", type=positive_int, default=5000)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("rank-pairs", help="Select non-regressing metric-ranked training pairs")
    p.add_argument("--scores", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-margin", type=float, default=0.05)
    p.add_argument("--max-wer", type=float, default=0.1)
    p.add_argument("--max-cer", type=float, default=0.05)
    p.add_argument("--min-similarity", type=float, default=0.6)

    p = sub.add_parser("distill-cache", help="Save coarse segments from an in-domain teacher's trajectories")
    add_inference_args(p, steps=False)
    add_partition_args(p)
    p.add_argument("--cache", required=True)
    p.add_argument("--limit", type=positive_int, default=1000)
    p.add_argument("--teacher-steps", type=positive_int, default=32)
    p.add_argument("--student-steps", type=positive_int, default=8)

    p = sub.add_parser(
        "post-train", help="Experimental preference learning or trajectory distillation with DDP, or online Flow-GRPO"
    )
    p.add_argument("--mode", choices=["preference", "distill", "grpo"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--data", help="Pair or trajectory JSONL (required for preference/distill)")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu"])
    p.add_argument("--steps", type=positive_int, default=1000)
    p.add_argument("--batch-size", type=positive_int, default=2)
    p.add_argument("--accumulation", type=positive_int, default=2)
    p.add_argument("--workers", type=int, default=2)
    add_loader_args(p)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument(
        "--optimizer", choices=["muon", "adamw"], help="Default: the optimizer recorded in the checkpoint"
    )
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--anchor", type=float, default=0.1)
    p.add_argument("--replay-weight", type=float, default=1.0)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--save-every", type=positive_int, default=100)
    p.add_argument("--seed", type=int, default=42)
    add_grpo_args(p)

    args = parser.parse_args()
    if args.command == "post-train" and args.mode != "grpo" and not args.data:
        parser.error("--data is required for --mode preference/distill")
    if getattr(args, "languages", None) is not None:
        from .prepare import ENGLISH_TAGS

        args.languages = ENGLISH_TAGS if args.languages == "en" else set(args.languages.split(","))
    if hasattr(args, "num_shards") and not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be between zero and num-shards - 1")
    if args.command == "inspect":
        from .config import Config
        from .model import FlowTTS

        cfg = Config.load(args.config)
        model = FlowTTS(cfg.model)
        print(
            json.dumps(
                {
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "config": cfg.to_dict(),
                    "codec_included": False,
                },
                indent=2,
            )
        )
    elif args.command in {"prepare", "merge"}:
        from . import prepare

        getattr(prepare, args.command)(args)
    elif args.command == "train":
        from .training import train

        train(args)
    elif args.command == "infer":
        from .inference import infer

        infer(args)
    elif args.command == "evaluate":
        from .metrics import evaluate

        evaluate(args)
    elif args.command == "compare":
        from .comparison import compare

        compare(args)
    elif args.command == "post-train" and args.mode == "grpo":
        from .grpo import grpo_train

        grpo_train(args)
    elif args.command in {"audit-cache", "codec-reconstruct", "make-cases", "run-eval"}:
        from . import experiments

        getattr(experiments, args.command.replace("-", "_"))(args)
    else:
        from . import posttrain

        getattr(posttrain, args.command.replace("-", "_"))(args)


def add_partition_args(parser):
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=42)


def add_split_key_arg(parser, note=""):
    parser.add_argument(
        "--split-key",
        help="Regex extracting the key the split hash uses from the speaker label (named group `key`, else "
        "group 1, else the whole match), e.g. '^(.+)_speaker_\\d+$' holds whole episodes out; labels it does "
        "not match are errors." + note,
    )


def add_codec_args(parser):
    parser.add_argument("--codec-backend", choices=["reference", "fast"], default="reference")
    parser.add_argument("--codec-compile", action="store_true", help="Compile the exact fast codec trunk")
    parser.add_argument("--codec-graphs", action="store_true", help="Bounded repeated-shape CUDA graphs")
    parser.add_argument("--codec-graph-max-shapes", type=positive_int, default=4)
    parser.add_argument(
        "--codec-layout",
        choices=["native", "channels_last"],
        default="native",
        help="channels_last is experimental and can be slower in strict FP32",
    )


def add_loader_args(parser):
    parser.add_argument("--worker-threads", type=positive_int)
    parser.add_argument("--prefetch-factor", type=positive_int)
    parser.add_argument("--loader-start-method", choices=["spawn", "forkserver"])


def add_grpo_args(parser, training=True):
    """Options of `post-train --mode grpo`; training=False keeps the sampler/reward part (best-of-N oracle).

    Only read in grpo mode (see dacvae_tts/grpo.py for the math and the defaults' sources).
    """
    add_codec_args(parser)
    g = parser.add_argument_group("GRPO sampler and reward")
    g.add_argument("--sample-steps", type=positive_int, default=16, help="Euler steps of a rollout (FlowTTS-GRPO: 16)")
    g.add_argument("--guidance", type=float, default=5.0, help="CFG of the sampled policy (deployed default 5)")
    g.add_argument("--sway", type=float, default=-1.0)
    g.add_argument("--guidance-from", type=float, default=0.0)
    g.add_argument("--guidance-until", type=float, default=1.0)
    g.add_argument(
        "--duration-mode", choices=["gt", "rule", "clamp", "syllable", "predictor", "auto"], default="auto",
        help="Target length of a prompt: the deployed rules, or gt = the recorded length",
    )
    g.add_argument("--duration-scale", type=float, default=1.0)
    g.add_argument("--max-frames", type=positive_int, default=1000, help="Skip longer prompts (prompt+target frames)")
    g.add_argument("--sde-window", type=positive_int, default=2, help="Consecutive stochastic (SDE) steps")
    g.add_argument("--sde-sigma", type=float, default=0.5, help="Constant sigma, or `a` of the flow schedule")
    g.add_argument("--sigma-schedule", choices=["constant", "flow"], default="constant")
    g.add_argument("--window-max", type=float, default=0.5, help="The window ends within this fraction of the steps")
    g.add_argument(
        "--shared-noise", action=argparse.BooleanOptionalAction, default=True,
        help="Samples of a group share the initial noise (differences come from the SDE window only)",
    )
    g.add_argument(
        "--reward-weights", default="cer=1.0,sim=0.5,dnsmos=0.4,utmos=0.4",
        help="name=weight list: cer, wer, sim, dnsmos, utmos, distillmos or module:attr",
    )
    g.add_argument("--reward-floors", default="", help="name=value overrides of the minimum group spread per term")
    g.add_argument("--advantage", choices=["standardized", "weighted"], default="standardized")
    g.add_argument("--allow-single-reward", action="store_true", help="Permit one reward term (hacking risk)")
    g.add_argument("--asr-model", default="openai/whisper-large-v3-turbo", help="Reward ASR (transformers Whisper)")
    g.add_argument("--speaker-model", default="microsoft/unispeech-sat-base-plus-sv", help="Reward SV model")
    g.add_argument("--dnsmos-model", help="Official sig_bak_ovr.onnx (needed by the dnsmos term)")
    g.add_argument("--utmos-repo", default="tarepan/SpeechMOS:v1.2.0", help="torch.hub repo of UTMOS22-strong")
    g.add_argument("--language", default="tr", help="Whisper language of the judges")
    g.add_argument(
        "--metric-normalization", choices=["english-unicode-v2", "legacy-ascii-v1", "turkish-v1", "turkish-v2"]
    )
    g.add_argument("--reward-device", help="Device of the codec and judges (default: the policy device)")
    if not training:
        return
    t = parser.add_argument_group("GRPO optimization and held-out monitor")
    t.add_argument("--group-size", type=positive_int, default=8, help="Samples per prompt (FlowTTS-GRPO: 10)")
    t.add_argument("--prompts-per-step", type=positive_int, default=4, help="Groups per rollout step")
    t.add_argument("--micro-batch", type=positive_int, default=8, help="Samples per gradient forward")
    t.add_argument("--clip", type=float, default=1e-4, help="PPO ratio clip (per-element mean log-ratio)")
    t.add_argument("--kl", type=float, default=0.04, help="Weight of the closed-form KL to the start model")
    t.add_argument("--ppo-epochs", type=positive_int, default=1)
    t.add_argument("--updates-per-rollout", type=positive_int, default=1)
    t.add_argument("--logprob-reduction", choices=["mean", "sum"], default="mean")
    t.add_argument("--max-grad-norm", type=float, default=1.0)
    t.add_argument("--monitor-every", type=int, default=50, help="Held-out monitor interval in steps; 0 disables")
    t.add_argument("--monitor-prompts", type=int, default=32)
    t.add_argument("--monitor-steps", type=positive_int, default=32, help="Steps of the deployed ODE sampler")
    t.add_argument("--monitor-terms", default="cer,wer,sim,dnsmos", help="Held-out metrics (add distillmos)")
    t.add_argument("--monitor-asr-model", default="large-v3")
    t.add_argument("--monitor-speaker-model", default="microsoft/wavlm-base-plus-sv")
    t.add_argument("--monitor-audio", type=int, default=4, help="Monitor samples saved per evaluation")


def add_inference_args(parser, steps=True):
    add_codec_args(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    if steps:
        parser.add_argument("--steps", type=positive_int, default=16)
    parser.add_argument(
        "--guidance",
        type=float,
        default=1.5,
        help="1 disables CFG; use 1 for a distilled model with baked-in guidance",
    )
    parser.add_argument("--sway", type=float, default=-1.0)
    parser.add_argument(
        "--guidance-until", type=float, default=1.0, help="Apply CFG only while t < this (t=0 noise); 0.5 = noisy half"
    )
    parser.add_argument("--noise-scale", type=float, default=1.0, help="Scale of the initial noise (Echo: 0.8-0.9)")
    parser.add_argument("--guidance-from", type=float, default=0.0, help="Apply CFG only while t >= this")
    parser.add_argument("--cfg-rescale", type=float, default=0.0, help="CFG rescale phi in [0,1] (against over-saturation)")
    parser.add_argument("--apg-eta", type=float, default=1.0, help="APG weight of the parallel guidance component")
    parser.add_argument("--apg-norm", type=float, default=0.0, help="APG cap on the per-element RMS of the guidance")
    parser.add_argument("--apg-momentum", type=float, default=0.0, help="APG (reverse) momentum, e.g. -0.3")
    parser.add_argument(
        "--speaker-guidance", type=float, help="Independent speaker guidance scale (text guidance = --guidance)"
    )
    add_output_quality_args(parser)


def pre_tanh_gain(value):
    from .codec import parse_pre_tanh_gain

    try:
        return parse_pre_tanh_gain(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def add_output_quality_args(parser):
    """Late guidance window, latent moment matching and the decoder's pre-tanh gain (all off by default)."""
    parser.add_argument(
        "--guidance-split",
        type=float,
        help="Split the guided interval at this t: [from, split) uses --guidance/--apg-*/--cfg-rescale, "
        "[split, until) the *-late values (e.g. 0.5 with --apg-eta-late 0.5: CFG early, APG late)",
    )
    parser.add_argument("--guidance-late", type=float, help="Late-window CFG scale (default: --guidance)")
    parser.add_argument("--apg-eta-late", type=float, help="Late-window APG eta (default: --apg-eta)")
    parser.add_argument("--apg-norm-late", type=float, help="Late-window APG norm cap (default: --apg-norm)")
    parser.add_argument("--apg-momentum-late", type=float, help="Late-window APG momentum (default: --apg-momentum)")
    parser.add_argument("--cfg-rescale-late", type=float, help="Late-window CFG rescale phi (default: --cfg-rescale)")
    parser.add_argument(
        "--moment-match",
        choices=["std", "meanstd"],
        help="Rescale each generated latent channel to the voice prompt's std (meanstd: also its mean) before decoding",
    )
    parser.add_argument(
        "--pre-tanh-gain",
        type=pre_tanh_gain,
        help="Scale the codec decoder's output-tanh input against saturation: a number, auto (99.9th percentile of "
        "|input| to atanh 0.95) or auto:<output ceiling>",
    )


if __name__ == "__main__":
    main()
