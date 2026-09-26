# Gradio demo (Hugging Face Space)

`space-v2/` is the demo of the tr-combined models, https://huggingface.co/spaces/Vyvo/dacvae-tts-tr-v2-demo: full-v2
(default), full-cross and run C from `VoiceHub/dacvae-tts-tr-combined`, one sample per sentence and the duration
predictor refit on tr-combined by default, one sentence per chunk (see the engine's `plan_text`). It reuses the v1
interface; `tools/build_space_v2.sh` assembles it with the v1 example prompts and the current package.

The v1 demo of run C:

`space/` is the code of https://huggingface.co/spaces/Vyvo/dacvae-tts-tr-demo: `app.py` (interface), `engine.py` (models,
ZeroGPU jobs, post-processing), `requirements.txt`, Space card and the example prompts. The Space additionally contains a
copy of `src/dacvae_tts/` and the pre-generated `samples/`.

`tools/` are the scripts used on the training instance (paths refer to `/workspace`): `sync_package.sh` copies the package
into the Space and model-repo bundles, `deploy_space.py` uploads and waits for the Space, `make_samples.py` renders the
showcase samples through a running app, `demo_experiments.sh`/`demo_run.sh` run the Freya-TR-Eval experiments behind the
demo defaults, `hybrid_sim.py`/`compare_runs.py` analyse them, `push_experiments.py` publishes them and
`update_model_repo.py` refreshes `VoiceHub/dacvae-tts-tr-w512`. Results: `docs/turkish-experiments-2026-09-22.md` §8.
