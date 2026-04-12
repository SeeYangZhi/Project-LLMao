"""
Batch run eval pipeline on all cleaned model outputs.

Usage: 
    python scripts/batch_eval.py [--skip_judge] [--gemini_key KEY]
    python scripts/batch_eval.py --multi_classifier --skip_judge

Options:
    --skip_judge        Skip LLM judge (faster)
    --multi_classifier  Run all 3 sarcasm classifiers
    --gemini_key KEY    Gemini API key for LLM judge
"""
import subprocess
import os
import sys
import pandas as pd
from pathlib import Path

CLEAN_DIR = "model_outputs_clean"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Collect args to forward
extra_args = sys.argv[1:]

models = [f.replace(".csv", "") for f in os.listdir(CLEAN_DIR) if f.endswith(".csv")]
models.sort()

print(f"Found {len(models)} models to evaluate:")
for m in models:
    print(f"  - {m}")
print()

# Check if multi_classifier mode
is_multi = "--multi_classifier" in extra_args

if is_multi:
    print("=" * 70)
    print("MULTI-CLASSIFIER MODE: Running all 3 classifiers on each model")
    print("=" * 70)
    print()

all_summaries = []

for name in models:
    input_path = os.path.join(CLEAN_DIR, f"{name}.csv")
    output_path = os.path.join(RESULTS_DIR, f"{name}_results.csv")

    cmd = [
        sys.executable, "scripts/eval_pipeline.py",
        "--input", input_path,
        "--output", output_path,
    ] + extra_args

    print("=" * 70)
    print(f"EVALUATING: {name}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print("=" * 70)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"ERROR: {name} failed with return code {result.returncode}")
    else:
        # Collect summary if multi_classifier
        if is_multi:
            multi_path = output_path.replace("_results.csv", "_results_multi_classifier.csv")
            if os.path.exists(multi_path):
                multi_df = pd.read_csv(multi_path)
                multi_df['model'] = name
                all_summaries.append(multi_df)
    print()

# Generate final summary for multi-classifier mode
if is_multi and all_summaries:
    print("\n" + "=" * 80)
    print("MULTI-CLASSIFIER SUMMARY: ALL MODELS × ALL CLASSIFIERS")
    print("=" * 80)
    
    summary_df = pd.concat(all_summaries, ignore_index=True)
    
    # Pivot table: models as rows, classifiers as columns
    print("\n--- FLIP RATE BY MODEL × CLASSIFIER ---\n")
    
    pivot = summary_df.pivot(index='model', columns='classifier', values='flip_rate')
    pivot = pivot * 100  # Convert to percentage
    
    # Sort by model name
    pivot = pivot.sort_index()
    
    # Print nicely
    print(f"{'Model':<35}", end="")
    for clf in pivot.columns:
        print(f"{clf:>20}", end="")
    print()
    print("-" * (35 + 20 * len(pivot.columns)))
    
    for model, row in pivot.iterrows():
        print(f"{model:<35}", end="")
        for val in row:
            print(f"{val:>19.1f}%", end="")
        print()
    
    # Averages
    print("-" * (35 + 20 * len(pivot.columns)))
    print(f"{'AVERAGE':<35}", end="")
    for col in pivot.columns:
        print(f"{pivot[col].mean():>19.1f}%", end="")
    print()
    
    # Save combined summary
    summary_path = os.path.join(RESULTS_DIR, "all_models_multi_classifier.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved combined summary to: {summary_path}")
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    # Average flip rate by classifier
    clf_avg = summary_df.groupby('classifier')['flip_rate'].mean() * 100
    print("\nAverage flip rate by classifier:")
    for clf, rate in clf_avg.sort_values(ascending=False).items():
        print(f"  {clf:<25} {rate:>6.1f}%")
    
    # Range of flip rates
    print(f"\nFlip rate range:")
    print(f"  Min: {summary_df['flip_rate'].min()*100:.1f}% ({summary_df.loc[summary_df['flip_rate'].idxmin(), 'model']})")
    print(f"  Max: {summary_df['flip_rate'].max()*100:.1f}% ({summary_df.loc[summary_df['flip_rate'].idxmax(), 'model']})")
    
    # Classifier agreement check
    print("\nClassifier disagreement (same model, different flip rates):")
    for model in pivot.index:
        rates = pivot.loc[model].values
        spread = rates.max() - rates.min()
        if spread > 30:  # More than 30 percentage points difference
            print(f"  {model}: {spread:.1f} pp spread ({rates.min():.1f}% - {rates.max():.1f}%)")

print("\n" + "=" * 70)
print("ALL DONE. Results in:", RESULTS_DIR)
print("=" * 70)