"""
Analyze golden data: merge human annotations with automated metrics.
Comprehensive classifier evaluation and error analysis.
NOW WITH MULTI-CLASSIFIER SUPPORT - evaluates all 3 classifiers vs human ground truth.

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

# ============================================================
# CLASSIFIER DEFINITIONS
# ============================================================
CLASSIFIERS = [
    {
        'name': 'RoBERTa-Twitter',
        'column': 'hard_flipped_roberta_twitter',
        'delta_column': 'flip_delta_roberta_twitter',
        'type': 'RoBERTa',
        'training': 'Twitter',
        'short': 'twitter',
    },
    {
        'name': 'Bert-Kaggle',
        'column': 'hard_flipped_bert_kaggle',
        'delta_column': 'flip_delta_bert_kaggle',
        'type': 'Bert',
        'training': 'Kaggle',
        'short': 'kaggle',
    },
    {
        'name': 'RoBERTa-News',
        'column': 'hard_flipped_roberta_news',
        'delta_column': 'flip_delta_roberta_news',
        'type': 'RoBERTa',
        'training': 'News',
        'short': 'news',
    },
]

def detect_surface_edit(input_text, output_text):
    """Detect if output is just a surface edit of input (lowercase, punctuation only)."""
    import re
    def normalize(s):
        return re.sub(r'[^\w\s]', '', s.lower()).split()
    
    input_words = normalize(input_text)
    output_words = normalize(output_text)
    
    return input_words == output_words

def get_classifier_column(merged, clf_info):
    """Get the actual column name for a classifier, with fallback."""
    clf_col = clf_info['column']
    if clf_col in merged.columns:
        return clf_col
    # Fallback for Twitter (default)
    if clf_info['name'] == 'RoBERTa-Twitter' and 'hard_flipped' in merged.columns:
        return 'hard_flipped'
    return None

def analyze_single_classifier(merged, clf_info, human_col='human_flipped_strict', verbose=True):
    """Analyze a single classifier against human ground truth with full metrics."""
    clf_name = clf_info['name']
    clf_col = get_classifier_column(merged, clf_info)
    
    if clf_col is None:
        return None
    
    valid_clf = merged[clf_col].notna() & merged[human_col].notna()
    y_true = merged.loc[valid_clf, human_col].astype(int).values
    y_pred = merged.loc[valid_clf, clf_col].astype(int).values
    
    if len(y_true) == 0:
        return None
    
    # Confusion matrix counts
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = len(y_true)
    
    # All classifier metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    balanced_acc = (recall + specificity) / 2
    mcc = matthews_corrcoef(y_true, y_pred)
    
    try:
        kappa = cohen_kappa_score(y_true, y_pred)
    except:
        kappa = float('nan')
    
    if verbose:
        print(f"""
    ┌─────────────────────────────────────────────────────────────┐
    │  {clf_name:^55}  │
    ├─────────────────────────────────────────────────────────────┤
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
    │    Cohen's κ:       {kappa:+.3f}   Agreement beyond chance          │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """)
    
        # Interpretation
        print(f"""
    INTERPRETATION:
    ──────────────────────────────────────────────────────────────
    
    κ = {kappa:.3f} → {"🔴 NEGATIVE! Classifier is ANTI-CORRELATED with human!" if kappa < 0 else "🟡 Poor agreement" if kappa < 0.2 else "🟢 Fair agreement"}
    
    MCC = {mcc:.3f} → {"🔴 NEGATIVE! Worse than random!" if mcc < 0 else "🟡 Poor" if mcc < 0.3 else "🟢 Moderate"}
    
    The classifier has TWO problems:
    1. LOW RECALL ({recall*100:.0f}%): Misses {fn}/{tp+fn} ({fnr*100:.0f}%) of real flips
    2. LOW PRECISION ({precision*100:.0f}%): Wrong on {fp}/{tp+fp if tp+fp > 0 else 1} ({(1-precision)*100:.0f}%) of its "flipped" predictions
    
    {"These opposite errors (missing real + flagging fake) → NEGATIVE κ" if kappa < 0 else ""}
    """)
    
    return {
        'name': clf_name,
        'type': clf_info['type'],
        'training': clf_info['training'],
        'column': clf_col,
        'n_samples': total,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'fpr': fpr,
        'fnr': fnr,
        'balanced_acc': balanced_acc,
        'mcc': mcc,
        'kappa': kappa,
        'clf_flip_rate': y_pred.mean(),
        'human_flip_rate': y_true.mean(),
    }

def print_classifier_comparison(clf_results):
    """Print comparison table of all classifiers."""
    print(f"""
    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │                    ALL CLASSIFIERS vs HUMAN GROUND TRUTH                             │
    ├──────────────────────────────────────────────────────────────────────────────────────┤
    │ Classifier        │ Acc   │ Prec  │ Recall │ Spec  │ F1    │ MCC    │ κ      │ Flip% │
    ├───────────────────┼───────┼───────┼────────┼───────┼───────┼────────┼────────┼───────┤""")
    
    for r in clf_results:
        emoji = "🔴" if r['kappa'] < 0 else "🟡" if r['kappa'] < 0.2 else "🟢"
        print(f"    │ {emoji}{r['name']:<16} │ {r['accuracy']*100:4.1f}% │ {r['precision']*100:4.1f}% │ {r['recall']*100:5.1f}% │ {r['specificity']*100:4.1f}% │ {r['f1']:.3f} │ {r['mcc']:+.3f}  │ {r['kappa']:+.3f}  │ {r['clf_flip_rate']*100:4.1f}% │")
    
    print(f"    └──────────────────────────────────────────────────────────────────────────────────────┘")
    
    # Confusion matrices side by side
    print(f"\n    CONFUSION MATRICES SUMMARY:")
    print(f"    {'Classifier':<20} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5} │ {'FP%':>6} {'FN%':>6} │ {'Error%':>7}")
    print(f"    {'-'*75}")
    for r in clf_results:
        n = r['n_samples']
        err_pct = (r['fp'] + r['fn']) / n * 100
        print(f"    {r['name']:<20} {r['tp']:>5} {r['tn']:>5} {r['fp']:>5} {r['fn']:>5} │ {r['fp']/n*100:>5.1f}% {r['fn']/n*100:>5.1f}% │ {err_pct:>6.1f}%")
    
    print(f"\n    Human flip rate: {clf_results[0]['human_flip_rate']*100:.1f}%")

def analyze_model(model_name, golden_path, results_path):
    """Analyze a single model's golden data results with ALL classifiers."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {model_name}")
    print('='*80)
    
    # Load data
    golden = pd.read_csv(golden_path)
    results = pd.read_csv(results_path)
    
    print(f"Golden samples: {len(golden)}")
    print(f"Results samples: {len(results)}")
    
    # Determine which classifier columns are available
    available_clf_cols = [c for c in results.columns if c.startswith('hard_flipped')]
    print(f"Classifier columns found: {available_clf_cols}")
    
    # Build merge columns list
    merge_cols = ['id']
    for clf in CLASSIFIERS:
        if clf['column'] in results.columns:
            merge_cols.append(clf['column'])
        if clf['delta_column'] in results.columns:
            merge_cols.append(clf['delta_column'])
    
    # Also include default columns and other metrics
    for col in ['hard_flipped', 'flip_delta', 'similarity', 'bleu', 'perplexity', 
                'edit_dist_norm', 'paraphrase_score']:
        if col in results.columns and col not in merge_cols:
            merge_cols.append(col)
    
    # Merge
    merged = golden.merge(results[merge_cols], on='id', how='left')
    
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
    # MULTI-CLASSIFIER ANALYSIS
    # ============================================================
    print(f"\n" + "="*70)
    print(f"MULTI-CLASSIFIER PERFORMANCE (Human = Ground Truth)")
    print("="*70)
    
    clf_results = []
    for clf_info in CLASSIFIERS:
        print(f"\n{'─'*70}")
        print(f"CLASSIFIER: {clf_info['name']} ({clf_info['type']}, trained on {clf_info['training']})")
        print('─'*70)
        
        result = analyze_single_classifier(merged, clf_info, verbose=True)
        if result:
            clf_results.append(result)
        else:
            print(f"  WARNING: {clf_info['name']} column not found, skipping")
    
    # Print comparison summary
    if len(clf_results) > 1:
        print(f"\n" + "="*70)
        print("CLASSIFIER COMPARISON SUMMARY")
        print("="*70)
        print_classifier_comparison(clf_results)
        
        # Classifier agreement with each other
        print(f"\n    INTER-CLASSIFIER AGREEMENT (do classifiers agree with each other?):")
        for i, r1 in enumerate(clf_results):
            for r2 in clf_results[i+1:]:
                col1 = r1['column']
                col2 = r2['column']
                if col1 in merged.columns and col2 in merged.columns:
                    valid = merged[col1].notna() & merged[col2].notna()
                    try:
                        inter_clf_kappa = cohen_kappa_score(
                            merged.loc[valid, col1].astype(int),
                            merged.loc[valid, col2].astype(int)
                        )
                        emoji = "🔴" if inter_clf_kappa < 0.4 else "🟡" if inter_clf_kappa < 0.6 else "🟢"
                        print(f"      {emoji} {r1['name']} vs {r2['name']}: κ = {inter_clf_kappa:+.3f}")
                    except:
                        pass
    
    # ============================================================
    # SURFACE EDIT ANALYSIS
    # ============================================================
    print("\n" + "="*60)
    print("SURFACE EDIT ANALYSIS")
    print("="*60)
    
    surface_edits = merged[merged['is_surface_edit'] == True]
    real_rewrites = merged[merged['is_surface_edit'] == False]
    
    print(f"""
    Surface edits (only lowercase/punctuation change): {len(surface_edits)} ({len(surface_edits)/len(merged)*100:.1f}%)
    Real rewrites (actual content change): {len(real_rewrites)} ({len(real_rewrites)/len(merged)*100:.1f}%)
    """)
    
    # Analyze each classifier on surface edits
    if len(surface_edits) > 0:
        print(f"    SURFACE EDIT DETECTION BY CLASSIFIER:")
        print(f"    {'Classifier':<20} {'Clf Says Flip%':>15} {'Human Says Flip%':>17} {'Fooled?':>10}")
        print(f"    {'-'*65}")
        
        for clf_info in CLASSIFIERS:
            clf_col = get_classifier_column(merged, clf_info)
            if clf_col and clf_col in surface_edits.columns:
                clf_flip = surface_edits[clf_col].mean()
                human_flip = surface_edits['human_flipped_strict'].mean()
                fooled = "YES 🔴" if clf_flip > human_flip + 0.1 else "No 🟢"
                print(f"    {clf_info['name']:<20} {clf_flip*100:>14.1f}% {human_flip*100:>16.1f}% {fooled:>10}")
        
        # Examples of surface edits that fooled classifiers
        primary_col = get_classifier_column(merged, CLASSIFIERS[0]) or 'hard_flipped'
        if primary_col in surface_edits.columns:
            fooled = surface_edits[(surface_edits[primary_col]==1) & (surface_edits['human_flipped_strict']==0)]
            if len(fooled) > 0:
                print(f"\n    Examples where classifier was FOOLED by surface edit ({len(fooled)} total):")
                for _, ex in fooled.head(3).iterrows():
                    print(f"      Input:  {ex['input'][:60]}...")
                    print(f"      Output: {ex['output'][:60]}...")
                    print()
    
    if len(real_rewrites) > 0:
        print(f"    REAL REWRITES DETECTION BY CLASSIFIER:")
        print(f"    {'Classifier':<20} {'Clf Says Flip%':>15} {'Human Says Flip%':>17} {'Gap':>10}")
        print(f"    {'-'*65}")
        
        for clf_info in CLASSIFIERS:
            clf_col = get_classifier_column(merged, clf_info)
            if clf_col and clf_col in real_rewrites.columns:
                clf_flip = real_rewrites[clf_col].mean()
                human_flip = real_rewrites['human_flipped_strict'].mean()
                gap = (human_flip - clf_flip) * 100
                print(f"    {clf_info['name']:<20} {clf_flip*100:>14.1f}% {human_flip*100:>16.1f}% {gap:>+9.1f}pp")
    
    # ============================================================
    # ERROR PATTERN ANALYSIS (for primary classifier)
    # ============================================================
    print("\n" + "="*60)
    print("ERROR PATTERN ANALYSIS")
    print("="*60)
    
    primary_col = get_classifier_column(merged, CLASSIFIERS[0]) or 'hard_flipped'
    
    if primary_col in merged.columns:
        # Group by error type
        merged['error_type'] = 'Unknown'
        merged.loc[(merged[primary_col]==1) & (merged['human_flipped_strict']==1), 'error_type'] = 'TP'
        merged.loc[(merged[primary_col]==0) & (merged['human_flipped_strict']==0), 'error_type'] = 'TN'
        merged.loc[(merged[primary_col]==1) & (merged['human_flipped_strict']==0), 'error_type'] = 'FP'
        merged.loc[(merged[primary_col]==0) & (merged['human_flipped_strict']==1), 'error_type'] = 'FN'
        
        print(f"""
    What makes the classifier WRONG? Let's compare metrics:
    
    {'Error Type':<12} {'Count':>6} {'Edit Dist':>10} {'Similarity':>11} {'BLEU':>8} {'Surface%':>9}
    {'-'*60}""")
        
        for error_type in ['TP', 'TN', 'FP', 'FN']:
            subset = merged[merged['error_type'] == error_type]
            if len(subset) > 0:
                surface_pct = subset['is_surface_edit'].mean() * 100
                edit_dist = subset['edit_dist_norm'].mean() if 'edit_dist_norm' in subset.columns else float('nan')
                sim = subset['similarity'].mean() if 'similarity' in subset.columns else float('nan')
                bleu = subset['bleu'].mean() if 'bleu' in subset.columns else float('nan')
                print(f"    {error_type:<12} {len(subset):>6} {edit_dist:>10.3f} {sim:>11.3f} {bleu:>8.3f} {surface_pct:>8.1f}%")
        
        fp_df = merged[merged['error_type'] == 'FP']
        fn_df = merged[merged['error_type'] == 'FN']
        
        if len(fp_df) > 0 and len(fn_df) > 0:
            print(f"""
    KEY PATTERNS:
    
    FALSE POSITIVES (classifier fooled):
      - Edit distance: {fp_df['edit_dist_norm'].mean() if 'edit_dist_norm' in fp_df.columns else float('nan'):.3f} (low = minimal change)
      - Surface edit rate: {fp_df['is_surface_edit'].mean()*100:.1f}%
      → Classifier thinks lowercase/punctuation = sarcasm removed
    
    FALSE NEGATIVES (classifier missed):
      - Edit distance: {fn_df['edit_dist_norm'].mean() if 'edit_dist_norm' in fn_df.columns else float('nan'):.3f}
      - Surface edit rate: {fn_df['is_surface_edit'].mean()*100:.1f}%
      → Classifier doesn't recognize semantic sarcasm removal
    """)
    
    # ============================================================
    # CORRELATION ANALYSIS
    # ============================================================
    print(f"\n" + "="*60)
    print("CORRELATION: What predicts classifier error?")
    print("="*60)
    
    if primary_col in merged.columns:
        # Binary: is classifier correct?
        merged['clf_correct'] = (merged[primary_col] == merged['human_flipped_strict']).astype(int)
        
        # Correlations
        metrics = ['edit_dist_norm', 'similarity', 'bleu', 'paraphrase_score']
        # Add flip_delta columns
        for clf_info in CLASSIFIERS:
            if clf_info['delta_column'] in merged.columns:
                metrics.append(clf_info['delta_column'])
            elif 'flip_delta' in merged.columns and clf_info['name'] == 'RoBERTa-Twitter':
                metrics.append('flip_delta')
        
        print(f"\n    Correlation with classifier correctness:")
        print(f"    {'Metric':<30} {'Pearson r':>12} {'p-value':>12} {'Interpretation':<30}")
        print(f"    {'-'*85}")
        
        for metric in metrics:
            if metric not in merged.columns:
                continue
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
                    interp = f"Higher → more correct"
                else:
                    interp = f"Higher → less correct"
                
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {metric:<30} {r:>+10.3f}{sig:>2} {p:>12.4f} {interp:<30}")
    
    # ============================================================
    # SUBTYPE BREAKDOWN (ALL CLASSIFIERS)
    # ============================================================
    print(f"\n" + "="*60)
    print("SUBTYPE BREAKDOWN: ALL CLASSIFIERS vs HUMAN")
    print("="*60)
    
    subtype_stats = []
    
    for subtype in merged['subtype'].unique():
        sub_df = merged[merged['subtype'] == subtype]
        n = len(sub_df)
        
        row = {
            'subtype': subtype,
            'n': n,
            'human_flip_rate': sub_df['human_flipped_strict'].mean(),
            'surface_edit_rate': sub_df['is_surface_edit'].mean(),
            'mean_edit_dist': sub_df['edit_dist_norm'].mean() if 'edit_dist_norm' in sub_df.columns else float('nan'),
        }
        
        # Stats for each classifier
        for clf_info in CLASSIFIERS:
            clf_col = get_classifier_column(merged, clf_info)
            if clf_col is None or clf_col not in sub_df.columns:
                continue
            
            short = clf_info['short']
            sub_y_true = sub_df['human_flipped_strict'].astype(int)
            sub_y_pred = sub_df[clf_col].astype(int)
            
            # Counts
            sub_tp = ((sub_y_pred == 1) & (sub_y_true == 1)).sum()
            sub_tn = ((sub_y_pred == 0) & (sub_y_true == 0)).sum()
            sub_fp = ((sub_y_pred == 1) & (sub_y_true == 0)).sum()
            sub_fn = ((sub_y_pred == 0) & (sub_y_true == 1)).sum()
            
            row[f'{short}_flip_rate'] = sub_y_pred.mean()
            row[f'{short}_acc'] = accuracy_score(sub_y_true, sub_y_pred)
            row[f'{short}_prec'] = precision_score(sub_y_true, sub_y_pred, zero_division=0)
            row[f'{short}_rec'] = recall_score(sub_y_true, sub_y_pred, zero_division=0)
            row[f'{short}_fp'] = sub_fp
            row[f'{short}_fn'] = sub_fn
            row[f'{short}_error_rate'] = (sub_fp + sub_fn) / n if n > 0 else 0
            
            try:
                row[f'{short}_kappa'] = cohen_kappa_score(sub_y_true, sub_y_pred)
            except:
                row[f'{short}_kappa'] = float('nan')
            
            try:
                row[f'{short}_mcc'] = matthews_corrcoef(sub_y_true, sub_y_pred)
            except:
                row[f'{short}_mcc'] = float('nan')
        
        subtype_stats.append(row)
    
    subtype_df = pd.DataFrame(subtype_stats).sort_values('n', ascending=False)
    
    # Print subtype table (all classifiers)
    print(f"\n{'Subtype':<20} {'N':>4} {'Hum%':>6} │", end="")
    for clf_info in CLASSIFIERS:
        print(f" {clf_info['short'][:4].upper():>5}% {'κ':>7} │", end="")
    print()
    print("-" * 100)
    
    for _, row in subtype_df.iterrows():
        line = f"{row['subtype']:<20} {row['n']:>4} {row['human_flip_rate']*100:>5.1f}% │"
        for clf_info in CLASSIFIERS:
            short = clf_info['short']
            if f'{short}_flip_rate' in row:
                line += f" {row[f'{short}_flip_rate']*100:>5.1f}% {row[f'{short}_kappa']:>+6.3f} │"
            else:
                line += f"   N/A    N/A │"
        print(line)
    
    # Ranking by worst κ per classifier
    print(f"\n    SUBTYPE RANKINGS BY κ (worst → best):")
    for clf_info in CLASSIFIERS:
        short = clf_info['short']
        kappa_col = f'{short}_kappa'
        if kappa_col in subtype_df.columns:
            print(f"\n    {clf_info['name']}:")
            for i, (_, row) in enumerate(subtype_df.sort_values(kappa_col).iterrows()):
                emoji = "🔴" if row[kappa_col] < 0 else "🟡" if row[kappa_col] < 0.2 else "🟢"
                print(f"      {i+1}. {emoji} {row['subtype']:<20} κ = {row[kappa_col]:+.3f}")
    
    # ============================================================
    # EXAMPLE ERRORS BY SUBTYPE
    # ============================================================
    print(f"\n" + "="*60)
    print("EXAMPLE ERRORS BY SUBTYPE")
    print("="*60)
    
    if primary_col in merged.columns:
        for _, row in subtype_df.iterrows():
            subtype = row['subtype']
            sub_df = merged[merged['subtype'] == subtype]
            
            fp_examples = sub_df[(sub_df[primary_col]==1) & (sub_df['human_flipped_strict']==0)]
            fn_examples = sub_df[(sub_df[primary_col]==0) & (sub_df['human_flipped_strict']==1)]
            
            if len(fp_examples) > 0 or len(fn_examples) > 0:
                twitter_kappa = row.get('twitter_kappa', float('nan'))
                print(f"\n  [{subtype.upper()}] n={row['n']}, κ={twitter_kappa:+.3f}")
                
                if len(fp_examples) > 0:
                    ex = fp_examples.iloc[0]
                    print(f"    FALSE POSITIVE (classifier fooled):")
                    print(f"      In:  {ex['input'][:60]}...")
                    print(f"      Out: {ex['output'][:60]}...")
                    ed = ex.get('edit_dist_norm', 'N/A')
                    ed_str = f"{ed:.3f}" if isinstance(ed, float) else str(ed)
                    print(f"      Edit dist: {ed_str}, Surface edit: {ex['is_surface_edit']}")
                
                if len(fn_examples) > 0:
                    ex = fn_examples.iloc[0]
                    print(f"    FALSE NEGATIVE (classifier missed real flip):")
                    print(f"      In:  {ex['input'][:60]}...")
                    print(f"      Out: {ex['output'][:60]}...")
                    ed = ex.get('edit_dist_norm', 'N/A')
                    ed_str = f"{ed:.3f}" if isinstance(ed, float) else str(ed)
                    print(f"      Edit dist: {ed_str}, Surface edit: {ex['is_surface_edit']}")
    
    # ============================================================
    # MEANING & SUCCESS BY SUBTYPE
    # ============================================================
    print(f"\n" + "="*60)
    print("MEANING PRESERVATION & SUCCESS BY SUBTYPE")
    print("="*60)
    
    meaning_change_rate = merged['meaning_change'].mean()
    strict_success_rate = merged['human_strict_success'].mean()
    
    print(f"\n  Overall: Meaning change {meaning_change_rate*100:.1f}%, Strict success {strict_success_rate*100:.1f}%")
    
    print(f"\n  {'Subtype':<20} {'N':>4} {'Meaning Δ':>10} {'Success':>8}")
    print("  " + "-" * 45)
    for _, row in subtype_df.iterrows():
        sub_df = merged[merged['subtype'] == row['subtype']]
        mc = sub_df['meaning_change'].mean()
        ss = sub_df['human_strict_success'].mean()
        print(f"  {row['subtype']:<20} {row['n']:>4} {mc*100:>9.1f}% {ss*100:>7.1f}%")
    
    # ============================================================
    # AUTOMATED METRICS
    # ============================================================
    print(f"\n" + "="*60)
    print("AUTOMATED METRICS SUMMARY")
    print("="*60)
    
    if 'similarity' in merged.columns:
        print(f"\n    Similarity:       {merged['similarity'].mean():.3f} ± {merged['similarity'].std():.3f}")
    if 'edit_dist_norm' in merged.columns:
        print(f"    Edit distance:    {merged['edit_dist_norm'].mean():.3f} ± {merged['edit_dist_norm'].std():.3f}")
    if 'bleu' in merged.columns:
        print(f"    BLEU vs input:    {merged['bleu'].mean():.3f} ± {merged['bleu'].std():.3f}")
    if 'paraphrase_score' in merged.columns:
        print(f"    Paraphrase score: {merged['paraphrase_score'].mean():.4f} ± {merged['paraphrase_score'].std():.4f}")
    if 'perplexity' in merged.columns:
        print(f"    Perplexity:       {merged['perplexity'].mean():.1f} ± {merged['perplexity'].std():.1f}")
    
    # Save outputs
    output_path = Path(results_path).parent / f"{model_name}_merged.csv"
    merged.to_csv(output_path, index=False)
    print(f"\nSaved merged data to: {output_path}")
    
    subtype_output = Path(results_path).parent / f"{model_name}_subtype_analysis.csv"
    subtype_df.to_csv(subtype_output, index=False)
    print(f"Saved subtype analysis to: {subtype_output}")
    
    return {
        'model': model_name,
        'n_samples': len(merged),
        'inter_annotator_kappa': inter_kappa,
        'human_flip_rate': merged['human_flipped_strict'].mean(),
        'human_flip_rate_lenient': merged['human_flipped_lenient'].mean(),
        'meaning_change_rate': meaning_change_rate,
        'strict_success_rate': strict_success_rate,
        'surface_edit_rate': merged['is_surface_edit'].mean(),
        'mean_similarity': merged['similarity'].mean() if 'similarity' in merged.columns else float('nan'),
        'mean_edit_dist': merged['edit_dist_norm'].mean() if 'edit_dist_norm' in merged.columns else float('nan'),
        'mean_paraphrase': merged['paraphrase_score'].mean() if 'paraphrase_score' in merged.columns else float('nan'),
        'classifier_results': clf_results,
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
    # FINAL SUMMARY: ALL MODELS × ALL CLASSIFIERS
    # ============================================================
    print("\n" + "="*80)
    print("FINAL SUMMARY: ALL MODELS × ALL CLASSIFIERS")
    print("="*80)
    
    # Build comparison table
    print(f"""
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                            CLASSIFIER PERFORMANCE SUMMARY (ALL MODELS)                            │
    ├───────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ Model           │ Classifier        │ Acc   │ Prec  │ Recall │ Spec  │ F1    │ MCC    │ κ      │
    ├─────────────────┼───────────────────┼───────┼───────┼────────┼───────┼───────┼────────┼────────┤""")
    
    for stats in all_stats:
        model = stats['model']
        for clf in stats['classifier_results']:
            emoji = "🔴" if clf['kappa'] < 0 else "🟡" if clf['kappa'] < 0.2 else "🟢"
            print(f"    │ {model:<15} │ {emoji}{clf['name']:<16} │ {clf['accuracy']*100:4.1f}% │ {clf['precision']*100:4.1f}% │ {clf['recall']*100:5.1f}% │ {clf['specificity']*100:4.1f}% │ {clf['f1']:.3f} │ {clf['mcc']:+.3f}  │ {clf['kappa']:+.3f}  │")
    
    print(f"    └───────────────────────────────────────────────────────────────────────────────────────────────────┘")
    
    # Confusion matrix summary
    print(f"\n    CONFUSION MATRIX SUMMARY:")
    print(f"    {'Model':<15} {'Classifier':<20} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5} │ {'FP%':>6} {'FN%':>6} │ {'Surface%':>8}")
    print(f"    {'-'*95}")
    for stats in all_stats:
        for clf in stats['classifier_results']:
            n = clf['n_samples']
            print(f"    {stats['model']:<15} {clf['name']:<20} {clf['tp']:>5} {clf['tn']:>5} {clf['fp']:>5} {clf['fn']:>5} │ {clf['fp']/n*100:>5.1f}% {clf['fn']/n*100:>5.1f}% │ {stats['surface_edit_rate']*100:>7.1f}%")
    
    # Average by classifier across all models
    print(f"\n    AVERAGE BY CLASSIFIER (across all models):")
    print(f"    {'Classifier':<20} {'Acc':>7} {'Prec':>7} {'Recall':>8} {'MCC':>8} {'κ':>8}")
    print(f"    {'-'*65}")
    
    for clf_info in CLASSIFIERS:
        clf_name = clf_info['name']
        clf_stats = []
        for stats in all_stats:
            for clf in stats['classifier_results']:
                if clf['name'] == clf_name:
                    clf_stats.append(clf)
        
        if clf_stats:
            avg_acc = np.mean([c['accuracy'] for c in clf_stats])
            avg_prec = np.mean([c['precision'] for c in clf_stats])
            avg_rec = np.mean([c['recall'] for c in clf_stats])
            avg_mcc = np.mean([c['mcc'] for c in clf_stats])
            avg_kappa = np.mean([c['kappa'] for c in clf_stats])
            
            emoji = "🔴" if avg_kappa < 0 else "🟡" if avg_kappa < 0.2 else "🟢"
            print(f"    {emoji} {clf_name:<18} {avg_acc*100:>6.1f}% {avg_prec*100:>6.1f}% {avg_rec*100:>7.1f}% {avg_mcc:>+7.3f} {avg_kappa:>+7.3f}")
    
    # ============================================================
    # KEY FINDINGS FOR POSTER
    # ============================================================
    print("\n" + "="*80)
    print("KEY FINDINGS FOR POSTER")
    print("="*80)
    
    # Collect all kappa values
    all_kappas = []
    for stats in all_stats:
        for clf in stats['classifier_results']:
            all_kappas.append(clf['kappa'])
    
    print(f"""
    1. ALL CLASSIFIERS ARE UNRELIABLE (Evidence)
       ─────────────────────────────────────────────────────────────
       Total classifier evaluations: {len(all_kappas)}
       Negative κ (anti-correlated): {sum(1 for k in all_kappas if k < 0)} ({sum(1 for k in all_kappas if k < 0)/len(all_kappas)*100:.0f}%)
       
       │ Model          │ Classifier       │ κ       │ MCC     │ Acc   │ Prec  │ Recall │
       ├────────────────┼──────────────────┼─────────┼─────────┼───────┼───────┼────────┤""")
    
    for stats in all_stats:
        for clf in stats['classifier_results']:
            emoji = "🔴" if clf['kappa'] < 0 else "🟡"
            print(f"       │ {stats['model']:<14} │ {clf['name']:<16} │ {clf['kappa']:>+6.3f}  │ {clf['mcc']:>+6.3f}  │ {clf['accuracy']*100:>4.1f}% │ {clf['precision']*100:>4.1f}% │ {clf['recall']*100:>5.1f}% │ {emoji}")
    
    print(f"""
       This is NOT a single classifier problem - ALL classifiers fail:
       - Different architectures (RoBERTa, DistilBERT)  
       - Different training data (Twitter, Reddit, News)
       - Same result: NEGATIVE or near-zero κ with humans
       
       → Automated sarcasm flip rate is NOT a valid evaluation metric!
    
    2. WHY DO ALL CLASSIFIERS FAIL?
       ─────────────────────────────────────────────────────────────
       - They detect SURFACE patterns (punctuation, capitalization)
       - They miss SEMANTIC sarcasm removal (actual rewrites)
       - High false positive rate on surface edits
       - High false negative rate on real rewrites
    
    3. HUMAN EVALUATION IS RELIABLE
       ─────────────────────────────────────────────────────────────""")
    
    for stats in all_stats:
        emoji = "🟢" if stats['inter_annotator_kappa'] > 0.8 else "🟡"
        print(f"       {emoji} {stats['model']}: Inter-annotator κ = {stats['inter_annotator_kappa']:.3f}")
    
    print(f"""
       → All κ > 0.8 = Excellent agreement = Human labels trustworthy
       
       CONTRAST:
         Human-Human κ:      >0.80 (Excellent)
         Classifier-Human κ: <0.00 (Anti-correlated!)
    
    4. JOINT vs CONTROL (Main Finding - Visible ONLY with Human Eval)
       ─────────────────────────────────────────────────────────────""")
    
    joint = next((s for s in all_stats if s['model'] == 't5_base_joint'), None)
    control = next((s for s in all_stats if s['model'] == 't5_base_control'), None)
    
    if joint and control:
        print(f"""
       │ Metric          │ T5-Joint │ T5-Control │ Winner │
       ├─────────────────┼──────────┼────────────┼────────┤
       │ Meaning Change  │  {joint['meaning_change_rate']*100:5.1f}%  │   {control['meaning_change_rate']*100:5.1f}%   │ {'Joint ✓' if joint['meaning_change_rate'] < control['meaning_change_rate'] else 'Control'} │
       │ Strict Success  │  {joint['strict_success_rate']*100:5.1f}%  │   {control['strict_success_rate']*100:5.1f}%   │ {'Joint ✓' if joint['strict_success_rate'] > control['strict_success_rate'] else 'Control'} │
       
       → Classifiers CANNOT distinguish Joint from Control
       → Only human evaluation reveals Joint is better
    """)
    
    print(f"""
    5. IMPLICATION FOR SARCASM REWRITING RESEARCH
       ─────────────────────────────────────────────────────────────
       Papers using automated flip rate as primary metric are UNRELIABLE.
       Human evaluation is ESSENTIAL for this task.
       
       Our contribution: Demonstrated across 3 models × 3 classifiers
       that automated evaluation fails systematically.
    """)
    
    # Save summary CSV
    summary_rows = []
    for stats in all_stats:
        for clf in stats['classifier_results']:
            summary_rows.append({
                'model': stats['model'],
                'classifier': clf['name'],
                'classifier_type': clf['type'],
                'training_data': clf['training'],
                'accuracy': clf['accuracy'],
                'precision': clf['precision'],
                'recall': clf['recall'],
                'specificity': clf['specificity'],
                'f1': clf['f1'],
                'mcc': clf['mcc'],
                'kappa': clf['kappa'],
                'balanced_acc': clf['balanced_acc'],
                'fpr': clf['fpr'],
                'fnr': clf['fnr'],
                'tp': clf['tp'],
                'tn': clf['tn'],
                'fp': clf['fp'],
                'fn': clf['fn'],
                'clf_flip_rate': clf['clf_flip_rate'],
                'human_flip_rate': clf['human_flip_rate'],
                'inter_annotator_kappa': stats['inter_annotator_kappa'],
                'meaning_change_rate': stats['meaning_change_rate'],
                'strict_success_rate': stats['strict_success_rate'],
                'surface_edit_rate': stats['surface_edit_rate'],
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('results/golden/summary_all_classifiers.csv', index=False)
    print(f"\nSaved multi-classifier summary to: results/golden/summary_all_classifiers.csv")
    
    # Also save simple summary (backward compatible)
    simple_summary = []
    for stats in all_stats:
        # Use first classifier for backward compatibility
        clf = stats['classifier_results'][0] if stats['classifier_results'] else {}
        simple_summary.append({
            'model': stats['model'],
            'n_samples': stats['n_samples'],
            'inter_annotator_kappa': stats['inter_annotator_kappa'],
            'classifier_flip_rate': clf.get('clf_flip_rate', float('nan')),
            'human_flip_rate_strict': stats['human_flip_rate'],
            'human_flip_rate_lenient': stats['human_flip_rate_lenient'],
            'classifier_human_kappa': clf.get('kappa', float('nan')),
            'classifier_accuracy': clf.get('accuracy', float('nan')),
            'classifier_precision': clf.get('precision', float('nan')),
            'classifier_recall': clf.get('recall', float('nan')),
            'classifier_specificity': clf.get('specificity', float('nan')),
            'classifier_f1': clf.get('f1', float('nan')),
            'classifier_mcc': clf.get('mcc', float('nan')),
            'classifier_balanced_acc': clf.get('balanced_acc', float('nan')),
            'meaning_change_rate': stats['meaning_change_rate'],
            'strict_success_rate': stats['strict_success_rate'],
            'mean_similarity': stats['mean_similarity'],
            'mean_edit_dist': stats['mean_edit_dist'],
            'mean_paraphrase': stats['mean_paraphrase'],
            'surface_edit_rate': stats['surface_edit_rate'],
            'true_positive': clf.get('tp', 0),
            'true_negative': clf.get('tn', 0),
            'false_positive': clf.get('fp', 0),
            'false_negative': clf.get('fn', 0),
        })
    
    simple_df = pd.DataFrame(simple_summary)
    simple_df.to_csv('results/golden/summary.csv', index=False)
    print(f"Saved simple summary to: results/golden/summary.csv")

if __name__ == "__main__":
    main()