# GPU runbook: the experiments that remain

Everything that could be settled without a GPU is merged (see
[docs/review-and-architecture-2026-09-24.md](../../docs/review-and-architecture-2026-09-24.md)). What remains needs
synthesis with run C, new training runs or both. These scripts run all of it on one Linux GPU machine, one job per
GPU at a time. None of them has been run end to end yet: they were syntax-checked, their derived configs load and
their job queue was dry-run with a stub job, but no step has touched a GPU.

## Requirements

- Linux with NVIDIA GPUs, a CUDA 12 driver, bash 4.4+, `curl`, `git` and network access to Hugging Face and GitHub.
- The reference GPU is an RTX 4090 (24 GB): run C used frame budget 6000 on one of them at 0.565 s per update, so a
  20k-update arm takes about 3.1 hours plus its evaluation. Larger GPUs work unchanged.
- Disk, roughly: 10 GB for the raw corpus and Common Voice test split, 2–3 GB per latent cache (merged, clean, hq),
  about 5x the clean cache for the tempo store, a few GB for the teacher stores, and 2–3 GB per training run. Plan
  on 150 GB free.
- A `$WORK/.env` file (default `WORK=/workspace`) with the secrets; it is sourced by every step:

  ```bash
  HF_TOKEN=hf_...          # Vyvo/tr-dataset-12 and the original prompt recordings
  WANDB_API_KEY=...        # optional, together with WANDB_PROJECT=dacvae-tts-tr
  ```

## Running

```bash
git clone https://github.com/kadirnar/dacvae-tts /workspace/dacvae-tts && cd /workspace/dacvae-tts
bash scripts/gpu/00_setup.sh            # .venv, SpeechBrain, DNSMOS, Freya-TR-Eval, run C, Common Voice test
bash scripts/gpu/10_data.sh             # caches, corpus scores, clean/hq filters, leakage report, prompt set
bash scripts/gpu/15_stores.sh           # teacher and tempo stores for the arms that read them
bash scripts/gpu/20_eval_inference.sh   # run C: evaluation protocol, duration, sampler and output options
bash scripts/gpu/30_ab_training.sh      # 24 training arms at 20k updates, evaluated and compared
bash scripts/gpu/40_posttrain.sh        # model-guidance fine-tune, WSD 60k run, Flow-GRPO
```

Every step is idempotent: re-running skips finished work, an interrupted training arm resumes from its `last.pt`,
and one job can be re-run by name (`bash scripts/gpu/30_ab_training.sh swiglu`). Steps 20 and 30 are independent
of each other once step 10 is done. Long steps belong in `tmux` or behind `nohup ... > log 2>&1 &`.

Settings live in [env.sh](env.sh) and can be overridden per call, for example
`GPUS="0 1 2 3" FRAME_BUDGET=6000 bash scripts/gpu/30_ab_training.sh`. The defaults put data in `$WORK/data`,
models in `$WORK/models`, training runs in `$WORK/runs` and evaluations in `$WORK/outputs`.

## What each step answers

| Step, job | Issue | Question |
|---|---|---|
| 10: `speaker-clusters/leakage.json` | #6 | Which held-out podcast labels share a voice with train labels |
| 10: `cv-tr-prompts` | #3, #6 | 48 Common Voice test speakers: prompt voices that never occur in training |
| 20: `podcast-base` vs `podcast-noleak` | #6 | How much the old prompt set flattered SIM and WER |
| 20: `ceiling-codec`, `ceiling-real` | #3 | SIM-o, WER and UTMOS of codec resynthesis and of the real recordings (200 pairs) |
| 20: `clamp`, `auto`, `predictor`, `articulation` | #12 | Duration rules |
| 20: `bo3`, `bo3-duration`, `bo3-composite` | #12, #13 | Best-of-3 with the ASR selector, duration-diverse candidates, a composite selector |
| 20: `until05*`, `late-apg`, `late-g2`, `speaker-g3` | #13 | Guidance windows and scales |
| 20: `pretanh-auto`, `moment-std` | #13 | Decoder saturation and latent moment matching |
| 20: `oracle-ode`, `oracle-sde` | #16 | The ceiling that reweighting run C's own samples can reach |
| 30: `benchmark` | #7 | Step time of the speed options (run alone on an idle machine) |
| 30: `speed` | #7 | The speed options must leave quality unchanged |
| 30: `latent-negatives`, `no-negatives` | #8 | Latent repeat/skip negatives vs the transcript hinge vs none |
| 30: `long-skip` … `cond-text-pool` | #9 | The seven DiT block options, one at a time |
| 30: `repa`, `tla`, `repa-tla` | #10 | Speech-REPA and TLA-SA teacher alignment |
| 30: `pairs`, `pairs-cross`, `pairs-tail`, `pairs-char-ctc` | #11 | Training pairs, together and one group at a time |
| 30: `regularized` | #14 | Dropout 0.1 and weight decay 0.05 |
| 30: `char-units`, `tempo-prompts`, `speaker-condition`, `speaker-context` | — | The four second-round options |
| 40: `mg-w07`, `mg-w05` | #14 | Model-guidance fine-tune of run C, sampled without CFG |
| 40: `wsd` | #14 | One 60k WSD run with an hq cooldown vs run C + stage 2 |
| 40: `grpo` | #16 | Flow-GRPO under a composite reward |

Issue #15 (more Turkish data) is a data-pipeline step with its own scripts in `scripts/data/`
(`prepare_yodas.py`, `prepare_common_voice.py`, `prepare_issai_tsc.py`, `manifest_pipeline.py`); it is left out
here because its value can only be judged after the A/Bs have fixed the recipe.

## Protocol and decision rule

- **One variable per arm.** Every arm is run C's recipe (`configs/nano_tr_w512.yaml`) with one change, trained on
  the clean cache with frame budget 6000 on one GPU, on the unchanged 60k LR schedule and stopped at 20k updates
  (`--steps 60000 --stop-after 20000`). Arms that share a comparison must share the GPU model, the cache, the frame
  budget and the schedule.
- **Two baseline seeds.** `base-s42` is the reference; `base-s43`'s row in `compare.md` is the seed noise floor.
- **One evaluation.** Freya-TR-Eval (495 sentences) with the 48 leak-free Common Voice voices, guidance 5, 32 steps,
  protocol v2 (deterministic Whisper large-v3, SIM-o with WavLM-large ECAPA plus a SpeechBrain SV model, UTMOS,
  DNSMOS, clipping, loudness, bandwidth), turkish-v2 metric normalization and the Freya-convention WER/CER columns.
- **Decision.** `compare.md` gives paired differences with speaker-clustered jackknife-t intervals (48 clusters).
  An interval that contains 0 is a tie. An option is kept when its CER gain is larger than the base-s42/base-s43
  difference and WER, SIM-o and UTMOS do not get worse beyond the same margin. The winners are then combined into one
  full 60k run (`configs/experiments/tr_w512_combined.yaml` is the template).

## Known limits

- **Best-of-N.** The `bo3*` selectors use Whisper and the judge is Whisper, so their gains are an upper bound. An
  independent confirmation needs a judge from another model family, which the repository does not have yet.
- **Pre-tanh gain.** It changes loudness, and UTMOS moves with loudness. Compare `pretanh-auto` with `base` at equal
  `median_lufs` / `loudness_lufs` (`summary.tsv`), not on UTMOS alone.
- **Speaker context and speaker condition.** `eval_sentences.py` uses the prompt as the only context clip. The
  10 s / 30 s context sweep of `speaker-context` and the 3 s vs 10 s prompt test of `speaker-condition` need
  `dacvae-tts infer --context-audio ...` runs by hand.
- **Step time.** `training.tsv` reports seconds per update from the training logs; arms that ran next to each other
  competed for CPU cores. Use `30_ab_training.sh benchmark` on an otherwise idle machine for #7's numbers.
- **Text normalization.** New caches use `TEXT_NORM=turkish-v2`; run C was trained on turkish-v1 (138 of 42,591
  transcripts differ). The 20k arms are comparable among themselves; step 40 compares against run C itself.
- **Encoder GPUs.** `scripts/prepare_local_2gpu.sh` (step 10) uses GPUs `0 .. NGPU-1` regardless of `GPUS`.
