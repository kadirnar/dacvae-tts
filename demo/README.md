# Gradio demo (Hugging Face Space)

`space/` is the code of https://huggingface.co/spaces/Vyvo/dacvae-tts-tr-demo: `app.py` (interface), `engine.py` (models,
ZeroGPU jobs, post-processing), `requirements.txt`, Space card and the example prompts. The Space additionally contains a
copy of `src/dacvae_tts/` and the pre-generated `samples/`.

`tools/` are the scripts used on the training instance (paths refer to `/workspace`): `sync_package.sh` copies the package
into the Space and model-repo bundles, `deploy_space.py` uploads and waits for the Space, `make_samples.py` renders the
showcase samples through a running app, `demo_experiments.sh`/`demo_run.sh` run the Freya-TR-Eval experiments behind the
demo defaults, `hybrid_sim.py`/`compare_runs.py` analyse them, `push_experiments.py` publishes them and
`update_model_repo.py` refreshes `VoiceHub/dacvae-tts-tr-w512`. Results: `docs/turkce-arastirma-2026-09-22.md` §8.
