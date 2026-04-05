


"""
convert_isarcasm.py
Downloads iSarcasmEval and converts to [id, input, output, subtype] format
for testing the evaluation pipeline with REAL data.
Usage:
    python scripts/convert_isarcasm.py
    python scripts/convert_isarcasm.py --output data/isarcasm_test.csv
"""
import argparse
import pandas as pd
def main():
    parser = argparse.ArgumentParser(description="Convert iSarcasmEval to eval pipeline format")
    parser.add_argument("--output", type=str, default="isarcasm_test.csv",
                        help="Output CSV path (default: isarcasm_test.csv)")
    args = parser.parse_args()
    # Download directly from GitHub
    url = "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/train/train.En.csv"
    print("Downloading iSarcasmEval...")
    df = pd.read_csv(url)
    print(f"Raw dataset: {len(df)} rows")
    # Keep only sarcastic tweets (sarcastic == 1) that have a rephrase
    df = df[df["sarcastic"] == 1].copy()
    df = df[df["rephrase"].notna()].copy()
    print(f"After filtering sarcastic + has rephrase: {len(df)} rows")
    # Determine subtype: whichever of the 6 columns has value 1.0
    subtype_cols = ["sarcasm", "irony", "satire", "understatement", "overstatement", "rhetorical_question"]
    def get_subtype(row):
        for col in subtype_cols:
            if row[col] == 1.0:
                return col
        return "sarcasm"  # default fallback
    df["subtype"] = df.apply(get_subtype, axis=1)
    # Build final dataframe
    result = pd.DataFrame({
        "id":      range(len(df)),
        "input":   df["tweet"].values,       # sarcastic tweet
        "output":  df["rephrase"].values,     # non-sarcastic rephrase
        "subtype": df["subtype"].values
    })
    result.to_csv(args.output, index=False)
    print(f"\nSaved {args.output} with {len(result)} samples")
    print(f"\nSubtype distribution:")
    print(result["subtype"].value_counts())
    print(f"\nSample rows:")
    print(result.head(3).to_string(index=False))
    print(f"\nNow run:")
    print(f"  python scripts/eval_pipeline.py --input {args.output} --output isarcasm_results.csv --skip_judge")
if __name__ == "__main__":
    main()