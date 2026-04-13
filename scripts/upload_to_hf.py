"""Upload the merged LLaMA model(s) to Hugging Face Hub.

Uploads both the headline-only variant and the context-enhanced variant.
Each variant's README.md must already exist in its `final/` folder.

Reads HUGGINGFACE_API_KEY from .env in the project root.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VARIANTS = [
    {
        "repo_id": "SeeYangZhi/Llama-3.2-1B-Sarcasm-Rewriter",
        "folder": PROJECT_ROOT / "outputs" / "llama-3.2-1b-instruct" / "sar-to-non" / "final",
        "commit_message": "Upload fine-tuned LLaMA 3.2 1B for sarcasm rewriting",
        "gguf_file": PROJECT_ROOT / "outputs" / "llama-3.2-1b-instruct-gguf" / "llmao-llama-1b-sarcasm.gguf",
    },
    {
        "repo_id": "SeeYangZhi/Llama-3.2-1B-Sarcasm-Rewriter-Context",
        "folder": PROJECT_ROOT / "outputs" / "llama-3.2-1b-instruct-context" / "sar-to-non" / "final",
        "commit_message": "Upload context-enhanced LLaMA 3.2 1B for sarcasm rewriting",
        "gguf_file": None,
    },
]


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


def upload_variant(api: HfApi, token: str, variant: dict) -> None:
    repo_id = variant["repo_id"]
    folder = variant["folder"]

    if not folder.exists():
        print(f"[skip] {repo_id}: folder missing at {folder}")
        return

    print(f"\n=== {repo_id} ===")
    print(f"  Source: {folder}")
    print("  Creating/updating repo...")
    create_repo(repo_id, token=token, exist_ok=True, repo_type="model")

    print("  Uploading folder...")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        commit_message=variant["commit_message"],
        ignore_patterns=["*.bin", "checkpoint-*", "training_args.bin", "*.pt"],
    )
    print("  Merged model uploaded.")

    gguf = variant.get("gguf_file")
    if gguf and gguf.exists():
        print(f"  Uploading GGUF file from {gguf}...")
        api.upload_file(
            path_or_fileobj=str(gguf),
            path_in_repo=gguf.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add GGUF export for LMStudio / llama.cpp",
        )
        print("  GGUF uploaded.")

    print(f"  Done -> https://huggingface.co/{repo_id}")


def main():
    token = load_token()
    api = HfApi(token=token)

    for variant in VARIANTS:
        upload_variant(api, token, variant)

    print("\nAll LLaMA variants uploaded.")


if __name__ == "__main__":
    main()
