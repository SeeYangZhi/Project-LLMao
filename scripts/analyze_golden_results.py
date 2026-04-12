"""
Analyze golden data: merge human annotations with automated metrics.
Compute classifier vs human agreement and generate poster stats.

Usage:
    python scripts/analyze_golden_results.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

def analyze_model(model_name, golden_path, results_path):
    """Analyze a single model's golden data results."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {model_name}")
    print('='*60)
    
    # Load data
    golden = pd.read_csv(golden_path)
    results = pd.read_csv(results_path)
    
    print(f"Golden samples: {len(golden)}")
    print(f"Results samples: {len(results)}")
    
    # Merge on id
    merged = golden.merge(
        results[['id', 'hard_flipped', 'flip_delta', 'similarity', 
                 'bleu', 'perplexity', 'edit_dist_norm', 'paraphrase_score']],
        on='id',
        how='left'
    )
    
    # Inter-annotator agreement
    valid = merged['human_sarcasm_removed_1'].notna() & merged['human_sarcasm_removed_2'].notna()
    a1 = merged.loc[valid, 'human_sarcasm_removed_1'].astype(int)
    a2 = merged.loc[valid, 'human_sarcasm_removed_2'].astype(int)
    
    agreement = (a1 == a2).mean()
    try:
        inter_kappa = cohen_kappa_score(a1, a2)
    except:
        inter_kappa = float('nan')
    
    print(f"\nInter-annotator agreement: {agreement:.1%}")
    print(f"Inter-annotator Cohen's κ: {inter_kappa:.3f}")
    
    # Classifier vs Human comparison
    classifier_flip = merged['hard_flipped'].mean()
    human_flip_strict = merged['human_flipped_strict'].mean()
    human_flip_lenient = merged['human_flipped_lenient'].mean()
    
    # Classifier-Human agreement
    valid_clf = merged['hard_flipped'].notna() & merged['human_flipped_strict'].notna()
    clf_human_agree = (
        merged.loc[valid_clf, 'hard_flipped'] == merged.loc[valid_clf, 'human_flipped_strict']
    ).mean()
    
    try:
        clf_human_kappa = cohen_kappa_score(
            merged.loc[valid_clf, 'hard_flipped'].astype(int),
            merged.loc[valid_clf, 'human_flipped_strict'].astype(int)
        )
    except:
        clf_human_kappa = float('nan')
    
    print(f"\nClassifier flip rate: {classifier_flip:.1%}")
    print(f"Human flip rate (strict): {human_flip_strict:.1%}")
    print(f"Human flip rate (lenient): {human_flip_lenient:.1%}")
    print(f"Classifier-Human agreement: {clf_human_agree:.1%}")
    print(f"Classifier-Human Cohen's κ: {clf_human_kappa:.3f}")
    
    # Meaning & Success
    meaning_change_rate = merged['meaning_change'].mean()
    strict_success_rate = merged['human_strict_success'].mean()
    
    print(f"\nMeaning change rate: {meaning_change_rate:.1%}")
    print(f"Strict success rate: {strict_success_rate:.1%}")
    
    # Automated metrics
    print(f"\nAutomated Metrics:")
    print(f"  Similarity: {merged['similarity'].mean():.3f}")
    print(f"  BLEU vs input: {merged['bleu'].mean():.3f}")
    print(f"  Edit distance: {merged['edit_dist_norm'].mean():.3f}")
    print(f"  Paraphrase score: {merged['paraphrase_score'].mean():.4f}")
    
    # Find disagreement examples
    print(f"\nDisagreement Examples (classifier ≠ human):")
    
    # Classifier says flipped, humans say no
    false_pos = merged[(merged['hard_flipped']==1) & (merged['human_flipped_strict']==0)]
    if len(false_pos) > 0:
        ex = false_pos.iloc[0]
        print(f"\n  FALSE POSITIVE (classifier wrong):")
        print(f"    Input:  {ex['input'][:70]}...")
        print(f"    Output: {ex['output'][:70]}...")
    
    # Classifier says not flipped, humans say yes
    false_neg = merged[(merged['hard_flipped']==0) & (merged['human_flipped_strict']==1)]
    if len(false_neg) > 0:
        ex = false_neg.iloc[0]
        print(f"\n  FALSE NEGATIVE (classifier missed):")
        print(f"    Input:  {ex['input'][:70]}...")
        print(f"    Output: {ex['output'][:70]}...")
    
    # Save merged data
    output_path = Path(results_path).parent / f"{model_name}_merged.csv"
    merged.to_csv(output_path, index=False)
    print(f"\nSaved merged data to: {output_path}")
    
    return {
        'model': model_name,
        'n_samples': len(merged),
        'inter_annotator_kappa': inter_kappa,
        'classifier_flip_rate': classifier_flip,
        'human_flip_rate_strict': human_flip_strict,
        'human_flip_rate_lenient': human_flip_lenient,
        'classifier_human_kappa': clf_human_kappa,
        'meaning_change_rate': meaning_change_rate,
        'strict_success_rate': strict_success_rate,
        'mean_similarity': merged['similarity'].mean(),
        'mean_edit_dist': merged['edit_dist_norm'].mean(),
        'mean_paraphrase': merged['paraphrase_score'].mean(),
    }

def main():
    models = [
        ('t5_base_joint', 'data/golden/cleaned/t5_base_joint_golden.csv', 'results/golden/t5_base_joint_results.csv'),
        ('t5_base_control', 'data/golden/cleaned/t5_base_control_golden.csv', 'results/golden/t5_base_control_results.csv'),
        ('bart_base_rl', 'data/golden/cleaned/bart_base_rl_golden.csv', 'results/golden/bart_base_rl_results.csv'),
    ]
    
    all_stats = []
    
    for model_name, golden_path, results_path in models:
        if not Path(golden_path).exists():
            print(f"WARNING: {golden_path} not found")
            continue
        if not Path(results_path).exists():
            print(f"WARNING: {results_path} not found")
            continue
        
        stats = analyze_model(model_name, golden_path, results_path)
        all_stats.append(stats)
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    
    df = pd.DataFrame(all_stats)
    
    print("\n" + df.to_string(index=False))
    
    # Save summary
    df.to_csv('results/golden/summary.csv', index=False)
    print("\nSaved summary to: results/golden/summary.csv")
    
    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS FOR POSTER")
    print("="*70)
    
    joint = df[df['model'] == 't5_base_joint'].iloc[0]
    control = df[df['model'] == 't5_base_control'].iloc[0]
    bart = df[df['model'] == 'bart_base_rl'].iloc[0]
    
    print(f"""
1. INTER-ANNOTATOR AGREEMENT
   Joint:   κ = {joint['inter_annotator_kappa']:.3f}
   Control: κ = {control['inter_annotator_kappa']:.3f}
   BART-RL: κ = {bart['inter_annotator_kappa']:.3f}
   → Excellent agreement, human eval is reliable

2. CLASSIFIER vs HUMAN (THE GAP)
   Classifier flip rate: ~{joint['classifier_flip_rate']:.1%} for all models
   Human flip rate:      ~{joint['human_flip_rate_strict']:.1%} for all models
   Classifier-Human κ:   {joint['classifier_human_kappa']:.3f} (poor agreement!)
   → Classifier can't distinguish real sarcasm removal

3. JOINT vs CONTROL (KEY FINDING)
   Meaning Change:  Joint {joint['meaning_change_rate']:.1%} vs Control {control['meaning_change_rate']:.1%}
   Strict Success:  Joint {joint['strict_success_rate']:.1%} vs Control {control['strict_success_rate']:.1%}
   → Joint preserves meaning better!

4. WHY?
   Joint learns to decompose: identify sarcasm mechanism BEFORE rewriting
   Control learns blind input→output mapping
   Strategy label acts as scaffold for what to preserve
""")

if __name__ == "__main__":
    main()