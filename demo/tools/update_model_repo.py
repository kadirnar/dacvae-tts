"""Update VoiceHub/dacvae-tts-tr-w512: model card, bundled dacvae_tts package and the demo code under space/.

  python update_model_repo.py
"""

import os

from huggingface_hub import HfApi

REPO = "VoiceHub/dacvae-tts-tr-w512"


def main():
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_file(path_or_fileobj="/workspace/release/model/README.md", path_in_repo="README.md", repo_id=REPO,
                    commit_message="Model card: Freya-validated inference settings (auto duration, best-of-3), new demo")
    api.upload_folder(folder_path="/workspace/release/model/dacvae_tts", path_in_repo="dacvae_tts", repo_id=REPO,
                      ignore_patterns=["__pycache__/*", "*.pyc"], delete_patterns=["*.py", "*.json"],
                      commit_message="dacvae_tts: Turkish frontend, duration modes and predictor, sampler options, batched synthesis")
    api.upload_folder(folder_path="/workspace/release/space", path_in_repo="space", repo_id=REPO,
                      ignore_patterns=["__pycache__/*", "*.pyc", "dacvae_tts/*", "models/*", "flagged/*"],
                      delete_patterns=["*.py", "*.md", "*.txt", "examples/*", "samples/*"],
                      commit_message="space/: rewritten demo (engine + interface)")
    print(f"https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
