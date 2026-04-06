"""
Batch preprocess ALL model output CSVs (round 2).
Handles:
- Rui Bin's T5 files: has prefix + subtype
- Yang Zhi's BART/LLaMA files: no prefix, no subtype (need to join from test set)
- Camille's t5_base joint: has prefix + joint output format + subtype

Usage: python scripts/batch_preprocess_v2.py
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
    text = str(text)
    match = re.search(r"rewrite:\s*(.*)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text

# ── Build subtype lookup from any file that has it ──
# Use t5_control or any ablation file as source of truth for id -> subtype mapping
subtype_source = None
for candidate in [
    os.path.join(OUTPUT_DIR, "t5_control.csv"),
    os.path.join(OUTPUT_DIR, "ablation_without_irony.csv"),
]:
    if os.path.exists(candidate):
        subtype_source = candidate
        break

subtype_map = {}
if subtype_source:
    src_df = pd.read_csv(subtype_source)
    # Build lookup by input text (more reliable than id since ids differ between files)
    for _, row in src_df.iterrows():
        subtype_map[str(row["input"]).strip().lower()] = row["subtype"]
    print(f"Loaded {len(subtype_map)} subtype mappings from {subtype_source}")
else:
    print("WARNING: No subtype source found. Run round 1 preprocessing first.")

# ── Files to process ──
files = {
    # Yang Zhi's models (no prefix, no subtype)
    "bart_base":        {"file": "bart-base.csv",             "has_prefix": False, "has_subtype": False, "joint": False},
    "bart_base_ce":     {"file": "bart-base-ce.csv",          "has_prefix": False, "has_subtype": False, "joint": False},
    "bart_base_ce_rl":  {"file": "bart-base-ce-rl.csv",       "has_prefix": False, "has_subtype": False, "joint": False},
    "bart_base_rl":     {"file": "bart-base-rl.csv",          "has_prefix": False, "has_subtype": False, "joint": False},
    "llama_3_2_1b":     {"file": "llama-3.2-1b-instruct.csv", "has_prefix": False, "has_subtype": False, "joint": False},
    # Camille's t5-base joint
    "t5_base_joint":    {"file": "t5_base_20260406_1026_joint_test_outputs.csv", "has_prefix": True, "has_subtype": True, "joint": True},
}

for name, config in files.items():
    path = os.path.join(INPUT_DIR, config["file"])
    if not os.path.exists(path):
        print(f"SKIP (not found): {path}")
        continue

    df = pd.read_csv(path)

    # Strip prefix if needed
    if config["has_prefix"]:
        df["input"] = df["input"].apply(strip_prefix)

    # Parse joint output if needed
    if config["joint"]:
        df["predicted_strategy"] = df["output"].apply(
            lambda x: re.search(r"strategy:\s*(\w+)", str(x)).group(1)
            if re.search(r"strategy:\s*(\w+)", str(x)) else ""
        )
        df["output"] = df["output"].apply(parse_joint_output)

    # Join subtype if missing
    if not config["has_subtype"] or "subtype" not in df.columns:
        if subtype_map:
            df["subtype"] = df["input"].apply(
                lambda x: subtype_map.get(str(x).strip().lower(), "unknown")
            )
            matched = (df["subtype"] != "unknown").sum()
            print(f"  Subtype joined: {matched}/{len(df)} matched")
        else:
            df["subtype"] = "unknown"

    # Re-index
    df["id"] = range(1, len(df) + 1)

    out_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(out_path, index=False)
    print(f"OK  {name}: {len(df)} rows -> {out_path}")

print(f"\nDone. All cleaned files in {OUTPUT_DIR}/")
print(f"\nNow run:")
print(f"  python scripts/batch_eval.py --skip_judge")
print(f"  or")
print(f"  python scripts/batch_eval.py --gemini_key YOUR_KEY")
