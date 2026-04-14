"""
Verify perplexity calculations for all models.
Shows both FILTERED and UNFILTERED means to identify the discrepancy.

Usage:
    python verify_perplexity.py
"""
import pandas as pd
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import os

CLEAN_DIR = "model_outputs_clean"

def compute_perplexity_both(outputs, lm_model, lm_tokenizer, device):
    """Compute both filtered and unfiltered perplexity."""
    per_sample = []
    lm_model.eval()
    
    for text in tqdm(outputs, desc="Computing PPL"):
        encodings = lm_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        input_ids = encodings.input_ids.to(device)
        with torch.no_grad():
            outputs_lm = lm_model(input_ids, labels=input_ids)
            loss = outputs_lm.loss
            ppl = torch.exp(loss).item()
        per_sample.append(ppl)
    
    # Unfiltered (all valid values)
    valid = [p for p in per_sample if not np.isnan(p) and not np.isinf(p)]
    unfiltered_mean = np.mean(valid) if valid else float('nan')
    
    # Filtered (< 10000, same as eval_pipeline.py)
    filtered = [p for p in per_sample if p < 10000 and not np.isnan(p)]
    filtered_mean = np.mean(filtered) if filtered else float('nan')
    
    # Stats
    outliers = len(valid) - len(filtered)
    max_ppl = max(valid) if valid else 0
    
    return {
        'unfiltered_mean': unfiltered_mean,
        'filtered_mean': filtered_mean,
        'n_samples': len(valid),
        'n_outliers': outliers,
        'max_ppl': max_ppl,
    }

def main():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print("Loading GPT-2...")
    lm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    lm_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    print("GPT-2 loaded.\n")
    
    # Find all models
    models = [f.replace(".csv", "") for f in os.listdir(CLEAN_DIR) if f.endswith(".csv")]
    models.sort()
    
    print(f"Found {len(models)} models to verify.\n")
    print("=" * 100)
    print(f"{'Model':<35} {'Unfiltered':>12} {'Filtered':>12} {'Ratio':>8} {'Outliers':>10} {'Max PPL':>12}")
    print("=" * 100)
    
    results = []
    
    for model_name in models:
        input_path = os.path.join(CLEAN_DIR, f"{model_name}.csv")
        df = pd.read_csv(input_path)
        outputs = df['output'].tolist()
        
        ppl_stats = compute_perplexity_both(outputs, lm_model, lm_tokenizer, device)
        
        ratio = ppl_stats['unfiltered_mean'] / ppl_stats['filtered_mean'] if ppl_stats['filtered_mean'] > 0 else 0
        
        print(f"{model_name:<35} {ppl_stats['unfiltered_mean']:>12.1f} {ppl_stats['filtered_mean']:>12.1f} {ratio:>7.2f}x {ppl_stats['n_outliers']:>10} {ppl_stats['max_ppl']:>12.1f}")
        
        results.append({
            'model': model_name,
            'unfiltered_ppl': ppl_stats['unfiltered_mean'],
            'filtered_ppl': ppl_stats['filtered_mean'],
            'ratio': ratio,
            'n_outliers': ppl_stats['n_outliers'],
            'max_ppl': ppl_stats['max_ppl'],
        })
    
    print("=" * 100)
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    results_df = pd.DataFrame(results)
    
    print(f"\nModels with ratio > 1.5 (significant filtering effect):")
    high_ratio = results_df[results_df['ratio'] > 1.5]
    for _, row in high_ratio.iterrows():
        print(f"  {row['model']:<35} ratio={row['ratio']:.2f}x, outliers={row['n_outliers']}, max={row['max_ppl']:.0f}")
    
    print(f"\nModels with ratio <= 1.5 (minimal filtering effect):")
    low_ratio = results_df[results_df['ratio'] <= 1.5]
    for _, row in low_ratio.iterrows():
        print(f"  {row['model']:<35} ratio={row['ratio']:.2f}x, outliers={row['n_outliers']}, max={row['max_ppl']:.0f}")
    
    # Save results
    results_df.to_csv("perplexity_verification.csv", index=False)
    print(f"\nResults saved to: perplexity_verification.csv")
    
    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)
    print("""
    If EVALUATION.md shows values close to 'Filtered' column → it uses filtered PPL
    If Dashboard shows values close to 'Unfiltered' column → it uses unfiltered PPL
    
    The 'Ratio' column shows how much the filtering affects each model.
    High ratio = many outlier samples with extreme perplexity.
    """)

if __name__ == "__main__":
    main()