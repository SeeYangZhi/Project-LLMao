"""
Batch preprocess all model output CSVs for evaluation.
- Strips prompt prefix from input column
- For joint model: extracts rewrite from "strategy: X rewrite: Y" output
- Saves cleaned files to cleaned/ folder

Usage: python scripts/batch_preprocess.py
"""
import pandas as pd
import os
import re

INPUT_DIR = "model_outputs_raw"
OUTPUT_DIR = "model_outputs_clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PREFIXES = [
    "rewrite to non-sarcastic and predict strategy: ",
    "rewrite to non-sarcastic: ",
]

def strip_prefix(text):
    text = str(text)
    for prefix in PREFIXES:
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix):]
    return text

def parse_joint_output(text):
    """Extract just the rewrite from 'strategy: X rewrite: Y' format."""
    text = str(text)
    match = re.search(r"rewrite:\s*(.*)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text

files = {
    # Ablation models (T5 trained without one subtype)
    "ablation_without_irony":              "initial2_2026_04_05_ablation_without_irony_test_outputs.csv",
    "ablation_without_overstatement":      "initial2_2026_04_05_ablation_without_overstatement_test_outputs.csv",
    "ablation_without_rhetorical_question":"initial2_2026_04_05_ablation_without_rhetorical_question_test_outputs.csv",
    "ablation_without_sarcasm":            "initial2_2026_04_05_ablation_without_sarcasm_test_outputs.csv",
    "ablation_without_satire":             "initial2_2026_04_05_ablation_without_satire_test_outputs.csv",
    "ablation_without_understatement":     "initial2_2026_04_05_ablation_without_understatement_test_outputs.csv",
    # Joint model (classify + rewrite)
    "joint":                               "initial2_2026_04_05_joint_test_outputs.csv",
    # Control model (rewrite only, baseline)
    "t5_control":                          "t5_control_t5-small_20260406_040406control_test_outputs.csv",
}

for name, filename in files.items():
    path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(path):
        print(f"SKIP (not found): {path}")
        continue

    df = pd.read_csv(path)

    # Strip prompt prefix from input
    df["input"] = df["input"].apply(strip_prefix)

    # For joint model: parse output to extract just the rewrite
    if name == "joint":
        df["predicted_strategy"] = df["output"].apply(
            lambda x: re.search(r"strategy:\s*(\w+)", str(x)).group(1)
            if re.search(r"strategy:\s*(\w+)", str(x)) else ""
        )
        df["output"] = df["output"].apply(parse_joint_output)

    # Re-index id
    df["id"] = range(1, len(df) + 1)

    out_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(out_path, index=False)
    print(f"OK  {name}: {len(df)} rows -> {out_path}")

print(f"\nDone. All cleaned files in {OUTPUT_DIR}/")
print(f"\nNext: run eval on each:")
for name in files:
    print(f"  python scripts/eval_pipeline.py --input {OUTPUT_DIR}/{name}.csv --output results/{name}_results.csv --skip_judge")
