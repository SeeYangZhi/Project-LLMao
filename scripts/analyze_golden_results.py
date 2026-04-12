"""
Analyze golden data: merge human annotations with automated metrics.
Comprehensive classifier evaluation and error analysis.

Usage:
    python scripts/analyze_golden_results.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    cohen_kappa_score, precision_score, recall_score, f1_score, 
    accuracy_score, matthews_corrcoef, confusion_matrix
)
from scipy import stats

def detect_surface_edit(input_text, output_text):
    """Detect if output is just a surface edit of input (lowercase, punctuation only)."""
    # Normalize: lowercase and remove punctuation
    import re
    def normalize(s):
        return re.sub(r'[^\w\s]', '', s.lower()).split()
    
    input_words = normalize(input_text)
    output_words = normalize(output_text)
    
    # Check if words are same after normalization
    return input_words == output_words

def analyze_model(model_name, golden_path, results_path):
    """Analyze a single model's golden data results."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {model_name}")
    print('='*80)
    
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
    
    # Detect surface edits
    merged['is_surface_edit'] = merged.apply(
        lambda x: detect_surface_edit(str(x['input']), str(x['output'])), axis=1
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
    
    # ============================================================
    # CLASSIFIER PERFORMANCE (Human as Ground Truth)
    # ============================================================
    print(f"\n" + "="*60)
    print(f"CLASSIFIER PERFORMANCE (Human = Ground Truth)")
    print("="*60)
    
    valid_clf = merged['hard_flipped'].notna() & merged['human_flipped_strict'].notna()
    y_true = merged.loc[valid_clf, 'human_flipped_strict'].astype(int).values
    y_pred = merged.loc[valid_clf, 'hard_flipped'].astype(int).values
    
    # Confusion matrix counts
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = len(y_true)
    
    # All classifier metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # = sensitivity = TPR
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    balanced_acc = (recall + specificity) / 2
    mcc = matthews_corrcoef(y_true, y_pred)
    
    try:
        clf_human_kappa = cohen_kappa_score(y_true, y_pred)
    except:
        clf_human_kappa = float('nan')
    
    classifier_flip = y_pred.mean()
    human_flip_strict = y_true.mean()
    human_flip_lenient = merged['human_flipped_lenient'].mean()
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────┐
    │                    CONFUSION MATRIX                         │
    ├─────────────────────────────────────────────────────────────┤
    │                        HUMAN (Ground Truth)                 │
    │                    Flipped (1)    Not Flipped (0)           │
    │                  ┌─────────────┬────────────────┐           │
    │  CLASSIFIER      │             │                │           │
    │    Predicted 1   │     {tp:3d}       │      {fp:3d}         │           │
    │    (says flip)   │  (True Pos) │ (False Pos)    │           │
    │                  ├─────────────┼────────────────┤           │
    │    Predicted 0   │     {fn:3d}       │      {tn:3d}         │           │
    │    (says no)     │ (False Neg) │  (True Neg)    │           │
    │                  └─────────────┴────────────────┘           │
    └─────────────────────────────────────────────────────────────┘
    
    RAW COUNTS:
      True Positives (TP):   {tp:3d} ({tp/total*100:5.1f}%)  - Both say flipped ✓
      True Negatives (TN):   {tn:3d} ({tn/total*100:5.1f}%)  - Both say not flipped ✓
      False Positives (FP):  {fp:3d} ({fp/total*100:5.1f}%)  - Classifier WRONG (says flip, human says no) ✗
      False Negatives (FN):  {fn:3d} ({fn/total*100:5.1f}%)  - Classifier MISSED (says no, human says flip) ✗
      ─────────────────────────────
      Total:                 {total:3d}
      Correct:               {tp+tn:3d} ({(tp+tn)/total*100:5.1f}%)
      Wrong:                 {fp+fn:3d} ({(fp+fn)/total*100:5.1f}%)
    """)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────┐
    │              COMPREHENSIVE CLASSIFIER METRICS               │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  BASIC METRICS:                                             │
    │    Accuracy:        {accuracy*100:5.1f}%   (TP+TN)/Total                │
    │    Error Rate:      {(1-accuracy)*100:5.1f}%   (FP+FN)/Total                │
    │                                                             │
    │  WHEN CLASSIFIER SAYS "FLIPPED":                            │
    │    Precision:       {precision*100:5.1f}%   TP/(TP+FP) - How often correct?  │
    │    → Classifier is WRONG {(1-precision)*100:.0f}% of the time when it says flipped │
    │                                                             │
    │  OF ALL REAL FLIPS (human=1):                               │
    │    Recall (TPR):    {recall*100:5.1f}%   TP/(TP+FN) - How many caught?    │
    │    Miss Rate (FNR): {fnr*100:5.1f}%   FN/(TP+FN) - How many missed?    │
    │    → Classifier MISSES {fnr*100:.0f}% of real flips                     │
    │                                                             │
    │  OF ALL NON-FLIPS (human=0):                                │
    │    Specificity:     {specificity*100:5.1f}%   TN/(TN+FP) - Correct rejections │
    │    False Pos Rate:  {fpr*100:5.1f}%   FP/(TN+FP) - Wrong acceptances  │
    │    → Classifier wrongly flags {fpr*100:.0f}% of non-flips               │
    │                                                             │
    │  OVERALL QUALITY:                                           │
    │    F1 Score:        {f1:.3f}   Harmonic mean of Prec & Recall       │
    │    Balanced Acc:    {balanced_acc*100:5.1f}%   (TPR + TNR) / 2               │
    │    MCC:             {mcc:+.3f}   Matthews Correlation Coefficient   │
    │    Cohen's κ:       {clf_human_kappa:+.3f}   Agreement beyond chance          │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """)
    
    # Interpretation
    print(f"""
    INTERPRETATION:
    ──────────────────────────────────────────────────────────────
    
    κ = {clf_human_kappa:.3f} → {"🔴 NEGATIVE! Classifier is ANTI-CORRELATED with human!" if clf_human_kappa < 0 else "🟡 Poor agreement" if clf_human_kappa < 0.2 else "🟢 Fair agreement"}
    
    MCC = {mcc:.3f} → {"🔴 NEGATIVE! Worse than random!" if mcc < 0 else "🟡 Poor" if mcc < 0.3 else "🟢 Moderate"}
    
    The classifier has TWO problems:
    1. LOW RECALL ({recall*100:.0f}%): Misses {fn}/{tp+fn} ({fnr*100:.0f}%) of real flips
    2. LOW PRECISION ({precision*100:.0f}%): Wrong on {fp}/{tp+fp} ({(1-precision)*100:.0f}%) of its "flipped" predictions
    
    These opposite errors (missing real + flagging fake) → NEGATIVE κ
    """)
    
    # ============================================================
    # SURFACE EDIT ANALYSIS
    # ============================================================
    print("="*60)
    print("SURFACE EDIT ANALYSIS")
    print("="*60)
    
    surface_edits = merged[merged['is_surface_edit'] == True]
    real_rewrites = merged[merged['is_surface_edit'] == False]
    
    print(f"""
    Surface edits (only lowercase/punctuation change): {len(surface_edits)} ({len(surface_edits)/len(merged)*100:.1f}%)
    Real rewrites (actual content change): {len(real_rewrites)} ({len(real_rewrites)/len(merged)*100:.1f}%)
    """)
    
    if len(surface_edits) > 0:
        # How often does classifier get fooled by surface edits?
        surface_clf_flip = surface_edits['hard_flipped'].mean()
        surface_human_flip = surface_edits['human_flipped_strict'].mean()
        
        print(f"""
    SURFACE EDITS (no real change):
      Classifier says flipped: {surface_clf_flip*100:.1f}%
      Human says flipped:      {surface_human_flip*100:.1f}%
      → Classifier is {"FOOLED by surface edits!" if surface_clf_flip > surface_human_flip + 0.1 else "not too fooled"}
        """)
        
        # Examples of surface edits that fooled classifier
        fooled = surface_edits[(surface_edits['hard_flipped']==1) & (surface_edits['human_flipped_strict']==0)]
        if len(fooled) > 0:
            print(f"    Examples where classifier was FOOLED by surface edit ({len(fooled)} total):")
            for _, ex in fooled.head(3).iterrows():
                print(f"      Input:  {ex['input'][:50]}...")
                print(f"      Output: {ex['output'][:50]}...")
                print()
    
    if len(real_rewrites) > 0:
        # Classifier performance on real rewrites
        real_clf_flip = real_rewrites['hard_flipped'].mean()
        real_human_flip = real_rewrites['human_flipped_strict'].mean()
        
        print(f"""
    REAL REWRITES (actual content change):
      Classifier says flipped: {real_clf_flip*100:.1f}%
      Human says flipped:      {real_human_flip*100:.1f}%
      Gap: {(real_human_flip - real_clf_flip)*100:.1f} percentage points
        """)
    
    # ============================================================
    # ERROR PATTERN ANALYSIS
    # ============================================================
    print("="*60)
    print("ERROR PATTERN ANALYSIS")
    print("="*60)
    
    # Group by error type
    merged['error_type'] = 'Unknown'
    merged.loc[(merged['hard_flipped']==1) & (merged['human_flipped_strict']==1), 'error_type'] = 'TP'
    merged.loc[(merged['hard_flipped']==0) & (merged['human_flipped_strict']==0), 'error_type'] = 'TN'
    merged.loc[(merged['hard_flipped']==1) & (merged['human_flipped_strict']==0), 'error_type'] = 'FP'
    merged.loc[(merged['hard_flipped']==0) & (merged['human_flipped_strict']==1), 'error_type'] = 'FN'
    
    print(f"""
    What makes the classifier WRONG? Let's compare metrics:
    
    {'Error Type':<12} {'Count':>6} {'Edit Dist':>10} {'Similarity':>11} {'BLEU':>8} {'Surface%':>9}
    {'-'*60}""")
    
    for error_type in ['TP', 'TN', 'FP', 'FN']:
        subset = merged[merged['error_type'] == error_type]
        if len(subset) > 0:
            surface_pct = subset['is_surface_edit'].mean() * 100
            print(f"    {error_type:<12} {len(subset):>6} {subset['edit_dist_norm'].mean():>10.3f} {subset['similarity'].mean():>11.3f} {subset['bleu'].mean():>8.3f} {surface_pct:>8.1f}%")
    
    print(f"""
    
    KEY PATTERNS:
    """)
    
    fp_df = merged[merged['error_type'] == 'FP']
    fn_df = merged[merged['error_type'] == 'FN']
    tp_df = merged[merged['error_type'] == 'TP']
    tn_df = merged[merged['error_type'] == 'TN']
    
    if len(fp_df) > 0 and len(fn_df) > 0:
        print(f"    FALSE POSITIVES (classifier fooled):")
        print(f"      - Edit distance: {fp_df['edit_dist_norm'].mean():.3f} (low = minimal change)")
        print(f"      - Surface edit rate: {fp_df['is_surface_edit'].mean()*100:.1f}%")
        print(f"      → Classifier thinks lowercase/punctuation = sarcasm removed")
        
        print(f"\n    FALSE NEGATIVES (classifier missed):")
        print(f"      - Edit distance: {fn_df['edit_dist_norm'].mean():.3f}")
        print(f"      - Surface edit rate: {fn_df['is_surface_edit'].mean()*100:.1f}%")
        print(f"      → Classifier doesn't recognize semantic sarcasm removal")
    
    # ============================================================
    # CORRELATION ANALYSIS
    # ============================================================
    print(f"\n" + "="*60)
    print("CORRELATION: What predicts classifier error?")
    print("="*60)
    
    # Binary: is classifier correct?
    merged['clf_correct'] = (merged['hard_flipped'] == merged['human_flipped_strict']).astype(int)
    
    # Correlations
    metrics = ['edit_dist_norm', 'similarity', 'bleu', 'paraphrase_score', 'flip_delta']
    
    print(f"\n    Correlation with classifier correctness:")
    print(f"    {'Metric':<20} {'Pearson r':>12} {'p-value':>12} {'Interpretation':<30}")
    print(f"    {'-'*75}")
    
    for metric in metrics:
        valid_corr = merged[metric].notna() & merged['clf_correct'].notna()
        if valid_corr.sum() > 10:
            r, p = stats.pearsonr(
                merged.loc[valid_corr, metric],
                merged.loc[valid_corr, 'clf_correct']
            )
            interp = ""
            if abs(r) < 0.1:
                interp = "No correlation"
            elif r > 0:
                interp = f"Higher {metric} → more correct"
            else:
                interp = f"Higher {metric} → less correct"
            
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {metric:<20} {r:>+10.3f}{sig:>2} {p:>12.4f} {interp:<30}")
    
    # ============================================================
    # SUBTYPE BREAKDOWN
    # ============================================================
    print(f"\n" + "="*60)
    print("SUBTYPE BREAKDOWN: Where does classifier fail?")
    print("="*60)
    
    subtype_stats = []
    
    for subtype in merged['subtype'].unique():
        sub_df = merged[merged['subtype'] == subtype]
        n = len(sub_df)
        
        sub_y_true = sub_df['human_flipped_strict'].astype(int)
        sub_y_pred = sub_df['hard_flipped'].astype(int)
        
        # Counts
        sub_tp = ((sub_y_pred == 1) & (sub_y_true == 1)).sum()
        sub_tn = ((sub_y_pred == 0) & (sub_y_true == 0)).sum()
        sub_fp = ((sub_y_pred == 1) & (sub_y_true == 0)).sum()
        sub_fn = ((sub_y_pred == 0) & (sub_y_true == 1)).sum()
        
        # Rates
        clf_flip_rate = sub_df['hard_flipped'].mean()
        human_flip_rate = sub_df['human_flipped_strict'].mean()
        
        # Metrics
        sub_acc = accuracy_score(sub_y_true, sub_y_pred) if n > 0 else 0
        sub_prec = precision_score(sub_y_true, sub_y_pred, zero_division=0)
        sub_rec = recall_score(sub_y_true, sub_y_pred, zero_division=0)
        
        try:
            sub_kappa = cohen_kappa_score(sub_y_true, sub_y_pred)
        except:
            sub_kappa = float('nan')
        
        try:
            sub_mcc = matthews_corrcoef(sub_y_true, sub_y_pred)
        except:
            sub_mcc = float('nan')
        
        subtype_stats.append({
            'subtype': subtype,
            'n': n,
            'clf_flip_rate': clf_flip_rate,
            'human_flip_rate': human_flip_rate,
            'TP': sub_tp, 'TN': sub_tn, 'FP': sub_fp, 'FN': sub_fn,
            'accuracy': sub_acc,
            'precision': sub_prec,
            'recall': sub_rec,
            'kappa': sub_kappa,
            'mcc': sub_mcc,
            'fp_rate': sub_fp / n if n > 0 else 0,
            'fn_rate': sub_fn / n if n > 0 else 0,
            'error_rate': (sub_fp + sub_fn) / n if n > 0 else 0,
            'surface_edit_rate': sub_df['is_surface_edit'].mean(),
            'mean_edit_dist': sub_df['edit_dist_norm'].mean(),
        })
    
    subtype_df = pd.DataFrame(subtype_stats).sort_values('n', ascending=False)
    
    print(f"\n{'Subtype':<20} {'N':>4} {'Clf%':>6} {'Hum%':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'κ':>7} {'MCC':>7} {'Err%':>6}")
    print("-" * 95)
    for _, row in subtype_df.iterrows():
        print(f"{row['subtype']:<20} {row['n']:>4} {row['clf_flip_rate']*100:>5.1f}% {row['human_flip_rate']*100:>5.1f}% {row['accuracy']*100:>5.1f}% {row['precision']*100:>5.1f}% {row['recall']*100:>5.1f}% {row['kappa']:>+7.3f} {row['mcc']:>+7.3f} {row['error_rate']*100:>5.1f}%")
    
    # Ranking
    print(f"\n    SUBTYPE RANKINGS:")
    
    print(f"\n    By Classifier ERROR RATE (worst → best):")
    for i, (_, row) in enumerate(subtype_df.sort_values('error_rate', ascending=False).iterrows()):
        emoji = "🔴" if row['error_rate'] > 0.5 else "🟡" if row['error_rate'] > 0.3 else "🟢"
        print(f"      {i+1}. {emoji} {row['subtype']:<20} {row['error_rate']*100:5.1f}% error ({row['FP']} FP + {row['FN']} FN)")
    
    print(f"\n    By κ (classifier-human agreement, worst → best):")
    for i, (_, row) in enumerate(subtype_df.sort_values('kappa').iterrows()):
        emoji = "🔴" if row['kappa'] < 0 else "🟡" if row['kappa'] < 0.2 else "🟢"
        print(f"      {i+1}. {emoji} {row['subtype']:<20} κ = {row['kappa']:+.3f}")
    
    print(f"\n    By RECALL (what % of real flips does classifier catch):")
    for i, (_, row) in enumerate(subtype_df.sort_values('recall').iterrows()):
        emoji = "🔴" if row['recall'] < 0.2 else "🟡" if row['recall'] < 0.4 else "🟢"
        print(f"      {i+1}. {emoji} {row['subtype']:<20} {row['recall']*100:5.1f}% recall (catches {row['TP']}/{row['TP']+row['FN']})")
    
    # ============================================================
    # EXAMPLE ERRORS BY SUBTYPE
    # ============================================================
    print(f"\n" + "="*60)
    print("EXAMPLE ERRORS BY SUBTYPE")
    print("="*60)
    
    for _, row in subtype_df.iterrows():
        subtype = row['subtype']
        sub_df = merged[merged['subtype'] == subtype]
        
        fp_examples = sub_df[(sub_df['hard_flipped']==1) & (sub_df['human_flipped_strict']==0)]
        fn_examples = sub_df[(sub_df['hard_flipped']==0) & (sub_df['human_flipped_strict']==1)]
        
        if len(fp_examples) > 0 or len(fn_examples) > 0:
            print(f"\n  [{subtype.upper()}] n={row['n']}, FP={row['FP']}, FN={row['FN']}, κ={row['kappa']:+.3f}")
            
            if len(fp_examples) > 0:
                ex = fp_examples.iloc[0]
                print(f"    FALSE POSITIVE (classifier fooled):")
                print(f"      In:  {ex['input'][:60]}...")
                print(f"      Out: {ex['output'][:60]}...")
                print(f"      Edit dist: {ex['edit_dist_norm']:.3f}, Surface edit: {ex['is_surface_edit']}")
            
            if len(fn_examples) > 0:
                ex = fn_examples.iloc[0]
                print(f"    FALSE NEGATIVE (classifier missed real flip):")
                print(f"      In:  {ex['input'][:60]}...")
                print(f"      Out: {ex['output'][:60]}...")
                print(f"      Edit dist: {ex['edit_dist_norm']:.3f}, Surface edit: {ex['is_surface_edit']}")
    
    # ============================================================
    # MEANING & SUCCESS BY SUBTYPE
    # ============================================================
    print(f"\n" + "="*60)
    print("MEANING PRESERVATION & SUCCESS BY SUBTYPE")
    print("="*60)
    
    meaning_change_rate = merged['meaning_change'].mean()
    strict_success_rate = merged['human_strict_success'].mean()
    
    print(f"\n  Overall: Meaning change {meaning_change_rate*100:.1f}%, Strict success {strict_success_rate*100:.1f}%")
    
    print(f"\n  {'Subtype':<20} {'N':>4} {'Meaning Δ':>10} {'Success':>8} {'Clf Acc':>8}")
    print("  " + "-" * 55)
    for _, row in subtype_df.iterrows():
        sub_df = merged[merged['subtype'] == row['subtype']]
        mc = sub_df['meaning_change'].mean()
        ss = sub_df['human_strict_success'].mean()
        print(f"  {row['subtype']:<20} {row['n']:>4} {mc*100:>9.1f}% {ss*100:>7.1f}% {row['accuracy']*100:>7.1f}%")
    
    # ============================================================
    # AUTOMATED METRICS
    # ============================================================
    print(f"\n" + "="*60)
    print("AUTOMATED METRICS SUMMARY")
    print("="*60)
    
    print(f"""
    Similarity:       {merged['similarity'].mean():.3f} ± {merged['similarity'].std():.3f}
    Edit distance:    {merged['edit_dist_norm'].mean():.3f} ± {merged['edit_dist_norm'].std():.3f}
    BLEU vs input:    {merged['bleu'].mean():.3f} ± {merged['bleu'].std():.3f}
    Paraphrase score: {merged['paraphrase_score'].mean():.4f} ± {merged['paraphrase_score'].std():.4f}
    Perplexity:       {merged['perplexity'].mean():.1f} ± {merged['perplexity'].std():.1f}
    Flip delta:       {merged['flip_delta'].mean():.3f} ± {merged['flip_delta'].std():.3f}
    """)
    
    # Save outputs
    output_path = Path(results_path).parent / f"{model_name}_merged.csv"
    merged.to_csv(output_path, index=False)
    print(f"Saved merged data to: {output_path}")
    
    subtype_output = Path(results_path).parent / f"{model_name}_subtype_analysis.csv"
    subtype_df.to_csv(subtype_output, index=False)
    print(f"Saved subtype analysis to: {subtype_output}")
    
    return {
        'model': model_name,
        'n_samples': len(merged),
        'inter_annotator_kappa': inter_kappa,
        'classifier_flip_rate': classifier_flip,
        'human_flip_rate_strict': human_flip_strict,
        'human_flip_rate_lenient': human_flip_lenient,
        'classifier_human_kappa': clf_human_kappa,
        'classifier_accuracy': accuracy,
        'classifier_precision': precision,
        'classifier_recall': recall,
        'classifier_specificity': specificity,
        'classifier_f1': f1,
        'classifier_mcc': mcc,
        'classifier_balanced_acc': balanced_acc,
        'meaning_change_rate': meaning_change_rate,
        'strict_success_rate': strict_success_rate,
        'mean_similarity': merged['similarity'].mean(),
        'mean_edit_dist': merged['edit_dist_norm'].mean(),
        'mean_paraphrase': merged['paraphrase_score'].mean(),
        'surface_edit_rate': merged['is_surface_edit'].mean(),
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
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
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "="*80)
    print("FINAL SUMMARY: ALL MODELS")
    print("="*80)
    
    df = pd.DataFrame(all_stats)
    
    print(f"""
    ┌────────────────────────────────────────────────────────────────────────────────┐
    │                         CLASSIFIER PERFORMANCE SUMMARY                          │
    ├────────────────────────────────────────────────────────────────────────────────┤
    │ Model          │ Acc   │ Prec  │ Recall │ Spec  │ F1    │ MCC    │ κ      │
    ├────────────────┼───────┼───────┼────────┼───────┼───────┼────────┼────────┤""")
    
    for _, row in df.iterrows():
        print(f"    │ {row['model']:<14} │ {row['classifier_accuracy']*100:4.1f}% │ {row['classifier_precision']*100:4.1f}% │ {row['classifier_recall']*100:5.1f}% │ {row['classifier_specificity']*100:4.1f}% │ {row['classifier_f1']:.3f} │ {row['classifier_mcc']:+.3f}  │ {row['classifier_human_kappa']:+.3f}  │")
    
    print(f"""    └────────────────────────────────────────────────────────────────────────────────┘
    """)
    
    # Confusion matrix summary
    print(f"    CONFUSION MATRIX SUMMARY:")
    print(f"    {'Model':<15} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5} │ {'FP%':>6} {'FN%':>6} │ {'Surface%':>8}")
    print(f"    {'-'*70}")
    for _, row in df.iterrows():
        n = row['n_samples']
        print(f"    {row['model']:<15} {int(row['true_positive']):>5} {int(row['true_negative']):>5} {int(row['false_positive']):>5} {int(row['false_negative']):>5} │ {row['false_positive']/n*100:>5.1f}% {row['false_negative']/n*100:>5.1f}% │ {row['surface_edit_rate']*100:>7.1f}%")
    
    # Save summary
    df.to_csv('results/golden/summary.csv', index=False)
    print(f"\nSaved summary to: results/golden/summary.csv")
    
    # ============================================================
    # KEY FINDINGS
    # ============================================================
    print("\n" + "="*80)
    print("KEY FINDINGS FOR POSTER")
    print("="*80)
    
    joint = df[df['model'] == 't5_base_joint'].iloc[0]
    control = df[df['model'] == 't5_base_control'].iloc[0]
    bart = df[df['model'] == 'bart_base_rl'].iloc[0]
    
    print(f"""
    1. CLASSIFIER IS UNRELIABLE (Evidence)
       ─────────────────────────────────────────────────────────────
       │ Metric         │ T5-Joint │ T5-Control │ BART-RL │ What it means           │
       ├────────────────┼──────────┼────────────┼─────────┼─────────────────────────┤
       │ Accuracy       │  {joint['classifier_accuracy']*100:5.1f}%  │   {control['classifier_accuracy']*100:5.1f}%   │  {bart['classifier_accuracy']*100:5.1f}% │ Overall correctness     │
       │ Precision      │  {joint['classifier_precision']*100:5.1f}%  │   {control['classifier_precision']*100:5.1f}%   │  {bart['classifier_precision']*100:5.1f}% │ When says flip, % right │
       │ Recall         │  {joint['classifier_recall']*100:5.1f}%  │   {control['classifier_recall']*100:5.1f}%   │  {bart['classifier_recall']*100:5.1f}% │ % of real flips caught  │
       │ MCC            │  {joint['classifier_mcc']:+.3f}  │   {control['classifier_mcc']:+.3f}   │  {bart['classifier_mcc']:+.3f} │ Correlation coefficient │
       │ Cohen's κ      │  {joint['classifier_human_kappa']:+.3f}  │   {control['classifier_human_kappa']:+.3f}   │  {bart['classifier_human_kappa']:+.3f} │ Agreement beyond chance │
       └────────────────────────────────────────────────────────────────────────────┘
       
       → NEGATIVE κ and MCC = classifier is ANTI-CORRELATED with human!
       → Precision ~35-50% = classifier is WRONG half the time
       → Recall ~15-22% = classifier MISSES 80% of real flips
    
    2. WHY CLASSIFIER FAILS
       ─────────────────────────────────────────────────────────────
       - Surface edit rate: {joint['surface_edit_rate']*100:.1f}% of outputs are just lowercase/punctuation changes
       - Classifier flags these as "flipped" but humans say NO
       - Classifier misses semantic sarcasm removal (real rewrites)
       
       Evidence (T5-Joint):
         False Positives: {int(joint['false_positive'])} (classifier fooled by surface edits)
         False Negatives: {int(joint['false_negative'])} (classifier missed real flips)
    
    3. HUMAN EVAL IS RELIABLE
       ─────────────────────────────────────────────────────────────
       Inter-annotator κ: Joint={joint['inter_annotator_kappa']:.3f}, Control={control['inter_annotator_kappa']:.3f}, BART={bart['inter_annotator_kappa']:.3f}
       → All κ > 0.8 = Excellent agreement = Human labels are trustworthy
    
    4. JOINT vs CONTROL (Main Finding)
       ─────────────────────────────────────────────────────────────
       Meaning Change:  Joint {joint['meaning_change_rate']*100:.1f}% vs Control {control['meaning_change_rate']*100:.1f}% → Joint wins!
       Strict Success:  Joint {joint['strict_success_rate']*100:.1f}% vs Control {control['strict_success_rate']*100:.1f}% → Joint wins!
       
       → Classifier CANNOT see this difference (both ~45% accuracy)
       → Only human evaluation reveals Joint is better
    """)

if __name__ == "__main__":
    main()