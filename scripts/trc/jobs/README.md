# tr-combined job scripts

The one-off jobs behind the numbers in `docs/tr-combined-experiments-2026-09-25.md` and `docs/experiments-log.md`,
kept as they ran on the RTX 5090 machine (paths under `/workspace`, credentials from `/workspace/.env`). The A/B arms
themselves run through `scripts/trc/arm_queue.py` / `run_arm.py`; these scripts cover everything around them.

| script | what it did |
|---|---|
| `build_stores.sh` | teacher stores from the original audio (mHuBERT-147 L12 PCA 256 for speech-REPA, ECAPA for TLA-SA / speaker condition) |
| `runc_steps.sh` | run C at 20k and 40k updates on the leak-free protocol (step vs data effect) |
| `inference_runc.sh`, `inference_runc_2.sh` | run C inference options (#12 duration, #13 sampler and output) |
| `quality_sweep.sh` | quality-condition controllability: one checkpoint asked for different DNSMOS targets |
| `system_evals.sh` | new model vs run C with the winning inference settings |
| `duration_evals.sh`, `duration_evals_old.sh` | full-cross and run C with the duration predictor refit on tr-combined |
| `v2_evals.sh` | full-v2 with the refit duration predictor (the headline numbers) |
| `y_base.sh` | y-base: full-v2's 20k snapshot evaluated in full as the round-3 baseline |
| `rescore_fix.sh` | rescoring of evaluation rows that failed with CUDA OOM |
| `migrate_pending.sh` | moving runs from the former per-run Hub repos into `VoiceHub/dacvae-tts-tr-combined` |
| `launch_v3.sh` | queueing full-v3 behind round 3 (stopped by the owner before it trained) |
| `status.sh` | one-screen status |
| `grpo_trend.py` | Flow-GRPO reward trend (#16) |
