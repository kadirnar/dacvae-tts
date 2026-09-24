"""Best-of-N oracle under the GRPO composite reward: the ceiling that reweighting the model's own samples can reach.

Run it before GRPO. For each held-out prompt N candidates are sampled and scored with the same reward terms, group
standardization and weights as `post-train --mode grpo` (dacvae_tts/grpo.py); summary.json reports per metric the first
candidate (a plain sample), the candidate mean, the composite-selected best, the per-metric best and the mean
within-group spread. `--sampler sde` samples the GRPO rollout policy instead of the deployed ODE sampler: its spread is
the signal the policy gradient gets (raise --sde-sigma / move the window if groups are degenerate or floored).

  python scripts/oracle_best_of_n.py --checkpoint runs/tr-w512-clean/step-0060000.pt --cache data/tr55/clean \\
      --output outputs/oracle-c60k --split val --limit 64 --candidates 8 --sample-steps 32 --guidance 5 \\
      --dnsmos-model models/sig_bak_ovr.onnx --save-audio 8
"""

import argparse

from dacvae_tts.cli import add_grpo_args, positive_int
from dacvae_tts.grpo import oracle_best_of_n


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True, help="New directory for oracle.jsonl, summary.json and audio")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--limit", type=positive_int, default=64, help="Number of prompts")
    parser.add_argument("--candidates", type=positive_int, default=8, help="N of best-of-N")
    parser.add_argument("--sampler", choices=["ode", "sde"], default="ode")
    parser.add_argument("--save-audio", type=int, default=0, help="Write prompt/first/best WAVs of this many prompts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    add_grpo_args(parser, training=False)
    return parser


def main():
    oracle_best_of_n(build_parser().parse_args())


if __name__ == "__main__":
    main()
