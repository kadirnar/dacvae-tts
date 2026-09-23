# Deploying this demo as a Hugging Face Space

The live demo is https://huggingface.co/spaces/Vyvo/dacvae-tts-tr-demo (ZeroGPU). To run a copy:

```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import HfApi, snapshot_download
api = HfApi()  # your token (huggingface-cli login)
src = snapshot_download("VoiceHub/dacvae-tts-tr-w512", allow_patterns=["space/*", "space/**", "dacvae_tts/*"])
repo = "YOUR_ORG/dacvae-tts-tr-demo"
api.create_repo(repo, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g")  # or "t4-small", "cpu-basic"
api.upload_folder(folder_path=f"{src}/space", repo_id=repo, repo_type="space")
api.upload_folder(folder_path=f"{src}/dacvae_tts", path_in_repo="dacvae_tts", repo_id=repo, repo_type="space")
print("https://huggingface.co/spaces/" + repo)
PY
```

- `app.py` is the interface, `engine.py` the models and GPU jobs. On ZeroGPU the codec, the published model, Whisper
  (`openai/whisper-large-v3-turbo`) and WavLM-SV are created at import time with `.to("cuda")`; ZeroGPU packs them once and
  moves them to the GPU per worker, so requests do not reload anything. Other checkpoints are loaded inside the worker
  and cached while it lives. GPU durations are computed per request (`spaces.GPU(duration=callable)`).
- No Space secret is needed: every model is public; private checkpoints are read with the visitor's own OAuth token.
- Local run: `pip install -r requirements.txt "gradio[oauth]==6.28.0" && python app.py` (any CUDA GPU or CPU).
