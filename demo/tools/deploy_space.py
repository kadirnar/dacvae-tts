"""Upload /workspace/release/space to a Hugging Face Space and wait until it runs.

  python deploy_space.py --repo Vyvo/dacvae-tts-tr-demo-dev --private     # test copy
  python deploy_space.py --repo Vyvo/dacvae-tts-tr-demo --delete-secret HF_TOKEN
"""

import argparse
import sys
import time

from huggingface_hub import HfApi

SPACE = "/workspace/release/space"
IGNORE = ["__pycache__/*", "**/__pycache__/*", "*.pyc", "SPACE.md", "models/*", "flagged/*", ".ruff_cache/*", ".ruff_cache/**"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--hardware", default="zero-a10g")
    parser.add_argument("--delete-secret", action="append", default=[])
    parser.add_argument("--message", default="Update demo")
    parser.add_argument("--wait", type=int, default=1500)
    args = parser.parse_args()
    api = HfApi()
    api.create_repo(args.repo, repo_type="space", space_sdk="gradio", space_hardware=args.hardware,
                    private=args.private, exist_ok=True)
    api.upload_folder(folder_path=SPACE, repo_id=args.repo, repo_type="space", ignore_patterns=IGNORE,
                      delete_patterns=["dacvae_tts/*.py", "dacvae_tts/*.json"], commit_message=args.message)
    for name in args.delete_secret:
        try:
            api.delete_space_secret(args.repo, name)
            print(f"deleted secret {name}", flush=True)
        except Exception as error:
            print(f"secret {name}: {error}", flush=True)
    print(f"uploaded {args.repo}; waiting", flush=True)
    time.sleep(30)
    deadline = time.time() + args.wait
    stage = None
    while time.time() < deadline:
        runtime = api.get_space_runtime(args.repo)
        if runtime.stage != stage:
            stage = runtime.stage
            print(f"{time.strftime('%H:%M:%S')} {stage} {runtime.hardware}", flush=True)
        if stage in {"RUNNING"}:
            return
        if stage and "ERROR" in stage:
            sys.exit(1)
        time.sleep(15)
    sys.exit(2)


if __name__ == "__main__":
    main()
