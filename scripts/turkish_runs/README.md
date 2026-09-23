# Launchers of the Turkish runs (Vast.ai instance, 2x RTX 4090)

The exact commands behind the runs in `docs/turkce-arastirma-2026-09-22.md` (paths refer to `/workspace`; credentials come
from `/workspace/.env`, which is not part of the repository):

- `run_pilot.sh`, `run_pilot_monitor.sh`: 4-shard pilot, monitor, codec/ASR ceiling and corpus re-transcription.
- `run_full.sh`, `run_ab.sh`: full-data runs A (`nano_tr.yaml`) and B (`nano_tr_ke4.yaml`) with their monitors.
- `run_round2.sh`: width-512 run C on the clean cache and the warm-started stage 2.
- `chain_c_stage2.sh`: stage 2 of run C on the hq cache, started when C finishes.
- `sweep_sampler.sh`, `sweep2.sh`: sampler sweeps (guidance, guidance interval, noise scale, sway, duration scale, steps).
- `final_eval.sh`, `finish_run.sh`: guidance sweep with DNSMOS, 5 custom sentences, Freya-TR-Eval and publishing to VoiceHub.

The demo experiments (duration modes, guidance variants, best-of-N) are in `demo/tools/`.
