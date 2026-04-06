"""
Batch run eval pipeline on all cleaned model outputs.
Usage: python scripts/batch_eval.py [--skip_judge] [--gemini_key KEY]
"""
import subprocess
import os
import sys

CLEAN_DIR = "model_outputs_clean"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Collect args to forward
extra_args = sys.argv[1:]

models = [f.replace(".csv", "") for f in os.listdir(CLEAN_DIR) if f.endswith(".csv")]
models.sort()

print(f"Found {len(models)} models to evaluate: {models}\n")

for name in models:
    input_path = os.path.join(CLEAN_DIR, f"{name}.csv")
    output_path = os.path.join(RESULTS_DIR, f"{name}_results.csv")

    cmd = [
        sys.executable, "scripts/eval_pipeline.py",
        "--input", input_path,
        "--output", output_path,
    ] + extra_args

    print("=" * 60)
    print(f"EVALUATING: {name}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"ERROR: {name} failed with return code {result.returncode}")
    print()

print("=" * 60)
print("ALL DONE. Results in:", RESULTS_DIR)
print("=" * 60)
