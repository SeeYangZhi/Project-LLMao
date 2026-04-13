"""Convert the merged LLaMA model to GGUF format for LMStudio.

The final/ directory already contains the merged model (merge_and_unload was
called during training). This script converts it to GGUF so LMStudio can load it.

Usage:
    # Install llama-cpp-python first:
    #   pip install llama-cpp-python
    #
    # Then clone llama.cpp and use its converter:
    #   git clone https://github.com/ggml-org/llama.cpp
    #   cd llama.cpp && pip install -r requirements.txt
    #
    # Convert:
    python llama.cpp/convert_hf_to_gguf.py \
        outputs/llama-3.2-1b-instruct/sar-to-non/final/ \
        --outfile outputs/llama-3.2-1b-instruct-gguf/llmao-llama-1b-sarcasm.gguf \
        --outtype f16

    # Then load the .gguf file in LMStudio.

Alternatively, if you want to quantize (smaller + faster):
    python llama.cpp/convert_hf_to_gguf.py \
        outputs/llama-3.2-1b-instruct/sar-to-non/final/ \
        --outfile outputs/llama-3.2-1b-instruct-gguf/llmao-llama-1b-sarcasm-q8.gguf \
        --outtype q8_0
"""

from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "outputs" / "llama-3.2-1b-instruct" / "sar-to-non" / "final"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "llama-3.2-1b-instruct-gguf"
LLAMA_CPP_DIR = PROJECT_ROOT / "llama.cpp"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outfile = OUTPUT_DIR / "llmao-llama-1b-sarcasm.gguf"

    converter = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not converter.exists():
        print(f"llama.cpp not found at {LLAMA_CPP_DIR}")
        print("Clone it first:")
        print(f"  cd {PROJECT_ROOT} && git clone https://github.com/ggml-org/llama.cpp")
        print(f"  pip install -r {LLAMA_CPP_DIR}/requirements.txt")
        sys.exit(1)

    cmd = [
        sys.executable, str(converter),
        str(MODEL_DIR),
        "--outfile", str(outfile),
        "--outtype", "f16",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"\nGGUF saved to: {outfile}")
    print("Load this file in LMStudio to use the fine-tuned model.")


if __name__ == "__main__":
    main()
