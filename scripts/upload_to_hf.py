"""Upload the merged LLaMA model and GGUF variant to Hugging Face Hub.

Reads HUGGINGFACE_API_KEY from .env in the project root.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MERGED_DIR = PROJECT_ROOT / "outputs" / "llama-3.2-1b-instruct" / "sar-to-non" / "final"
GGUF_FILE = PROJECT_ROOT / "outputs" / "llama-3.2-1b-instruct-gguf" / "llmao-llama-1b-sarcasm.gguf"

REPO_ID = "SeeYangZhi/Llama-3.2-1B-Sarcasm-Rewriter"


def load_token() -> str:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("HUGGINGFACE_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_API_KEY not found in .env or environment")
    return token


def main():
    token = load_token()
    api = HfApi(token=token)

    print(f"Creating/updating repo {REPO_ID}...")
    create_repo(REPO_ID, token=token, exist_ok=True, repo_type="model")

    print(f"Uploading merged safetensors from {MERGED_DIR}...")
    api.upload_folder(
        folder_path=str(MERGED_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload fine-tuned LLaMA 3.2 1B for sarcasm rewriting",
        ignore_patterns=["*.bin", "checkpoint-*"],
    )
    print("  Merged model uploaded.")

    if GGUF_FILE.exists():
        print(f"Uploading GGUF file from {GGUF_FILE}...")
        api.upload_file(
            path_or_fileobj=str(GGUF_FILE),
            path_in_repo=GGUF_FILE.name,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Add GGUF export for LMStudio / llama.cpp",
        )
        print("  GGUF uploaded.")
    else:
        print(f"  Skipping GGUF (not found at {GGUF_FILE})")

    print(f"\nDone. Model available at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
