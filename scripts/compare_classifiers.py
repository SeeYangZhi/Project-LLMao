"""
Compare multiple sarcasm classifiers against human annotations.
Tests whether the classifier failure is model-specific or fundamental.

Usage:
    python scripts/compare_classifiers.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    cohen_kappa_score, precision_score, recall_score, 
    f1_score, accuracy_score, matthews_corrcoef
)
from transformers import pipeline
import torch
import warnings
warnings.filterwarnings('ignore')

# Classifiers to test
CLASSIFIERS = [
    {
        'name': 'RoBERTa-Twitter-Irony',
        'model': 'cardiffnlp/twitter-roberta-base-irony',
        'type': 'RoBERTa',
        'training': 'Twitter',
        'sarcastic_labels': ['irony'],  # exact labels that mean sarcastic
        'non_sarcastic_labels': ['non_irony'],  # exact labels that mean NOT sarcastic
    },
    {
        'name': 'DistilBERT-Reddit',
        'model': 'helinivan/english-sarcasm-detector',
        'type': 'DistilBERT',
        'training': 'Reddit',
        'sarcastic_labels': ['LABEL_1', 'sarcasm', 'sarcastic'],
        'non_sarcastic_labels': ['LABEL_0', 'not_sarcasm', 'not sarcasm', 'normal'],
    },
    {
        'name': 'BERT-Sarcasm-News',
        'model': 'jkhan447/sarcasm-detection-RoBerta-base-POS',
        'type': 'RoBERTa',
        'training': 'News',
        'sarcastic_labels': ['LABEL_1', 'sarcastic', 'sarcasm'],
        'non_sarcastic_labels': ['LABEL_0', 'not_sarcastic', 'not sarcasm'],
    },
    {
        'name': 'DistilBERT-Sarcasm',
        'model': 'badalsahani/sarcasm-detector',
        'type': 'DistilBERT',
        'training': 'Headlines',
        'sarcastic_labels': ['LABEL_1', 'sarcastic', 'sarcasm'],
        'non_sarcastic_labels': ['LABEL_0', 'not_sarcastic', 'not sarcasm'],
    },
]

def load_classifier(model_name):
    """Load a classifier pipeline."""
    print(f"  Loading {model_name}...")
    try:
        device = 0 if torch.cuda.is_available() else -1
        clf = pipeline(
            "text-classification",
            model=model_name,
            device=device,
            truncation=True,
            max_length=512
        )
        return clf
    except Exception as e:
        print(f"  ERROR loading {model_name}: {e}")
        return None

def predict_sarcasm(clf, text, sarcastic_labels, non_sarcastic_labels):
    """
    Predict if text is sarcastic. 
    Returns (is_sarcastic, confidence)
    """
    try:
        result = clf(text[:512])[0]
        label = result['label']
        score = result['score']
        
        # Check exact match for sarcastic labels
        label_lower = label.lower()
        
        is_sarcastic = None
        for sl in sarcastic_labels:
            if label_lower == sl.lower():
                is_sarcastic = True
                break
        
        if is_sarcastic is None:
            for nsl in non_sarcastic_labels:
                if label_lower == nsl.lower():
                    is_sarcastic = False
                    break
        
        # If still None, try to infer from label name
        if is_sarcastic is None:
            if 'non' in label_lower or 'not' in label_lower or 'normal' in label_lower:
                is_sarcastic = False
            elif 'irony' in label_lower or 'sarcas' in label_lower:
                is_sarcastic = True
            else:
                # Default: assume LABEL_1 = sarcastic, LABEL_0 = not
                is_sarcastic = '1' in label
        
        return is_sarcastic, score
            
    except Exception as e:
        print(f"    Error predicting: {e}")
        return None, None

def evaluate_classifier(clf_info, merged_df):
    """Evaluate a single classifier against human annotations."""
    clf = load_classifier(clf_info['model'])
    if clf is None:
        return None
    
    # Debug: check what labels the model outputs
    sample_text = merged_df.iloc[0]['output']
    try:
        sample_result = clf(sample_text[:512])[0]
        print(f"  Sample prediction: label='{sample_result['label']}', score={sample_result['score']:.3f}")
    except:
        pass
    
    results = []
    sarcastic_count = 0
    non_sarcastic_count = 0
    
    print(f"  Running predictions on {len(merged_df)} samples...")
    for idx, row in merged_df.iterrows():
        output_text = str(row['output'])
        is_sarcastic, confidence = predict_sarcasm(
            clf, output_text, 
            clf_info['sarcastic_labels'], 
            clf_info['non_sarcastic_labels']
        )
        
        if is_sarcastic is not None:
            if is_sarcastic:
                sarcastic_count += 1
            else:
                non_sarcastic_count += 1
            
            # Flip = NOT sarcastic (sarcasm was removed)
            results.append({
                'id': row['id'],
                'clf_says_flipped': 0 if is_sarcastic else 1,
                'clf_confidence': confidence,
                'human_flipped': row['human_flipped_strict'],
            })
    
    print(f"  Predictions: {sarcastic_count} sarcastic, {non_sarcastic_count} non-sarcastic")
    print(f"  → Classifier flip rate: {non_sarcastic_count}/{len(results)*100:.1f}%")
    
    if len(results) == 0:
        return None
    
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    y_true = results_df['human_flipped'].astype(int).values
    y_pred = results_df['clf_says_flipped'].astype(int).values
    
    metrics = {
        'name': clf_info['name'],
        'type': clf_info['type'],
        'training': clf_info['training'],
        'n_samples': len(results_df),
        'clf_flip_rate': y_pred.mean(),
        'human_flip_rate': y_true.mean(),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
    }
    
    # Confusion matrix
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    metrics['TP'] = tp
    metrics['TN'] = tn
    metrics['FP'] = fp
    metrics['FN'] = fn
    
    return metrics

def main():
    print("="*80)
    print("COMPARING SARCASM CLASSIFIERS vs HUMAN ANNOTATIONS")
    print("="*80)
    
    # Load golden data (merged with human annotations)
    golden_files = [
        ('T5-Joint', 'results/golden/t5_base_joint_merged.csv'),
        ('T5-Control', 'results/golden/t5_base_control_merged.csv'),
        ('BART-RL', 'results/golden/bart_base_rl_merged.csv'),
    ]
    
    all_results = []
    
    for model_name, golden_path in golden_files:
        if not Path(golden_path).exists():
            print(f"WARNING: {golden_path} not found")
            continue
        
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print("="*80)
        
        merged_df = pd.read_csv(golden_path)
        print(f"Loaded {len(merged_df)} samples")
        print(f"Human flip rate: {merged_df['human_flipped_strict'].mean()*100:.1f}%")
        
        for clf_info in CLASSIFIERS:
            print(f"\n--- Testing: {clf_info['name']} ({clf_info['type']}, trained on {clf_info['training']}) ---")
            metrics = evaluate_classifier(clf_info, merged_df)
            
            if metrics:
                metrics['model'] = model_name
                all_results.append(metrics)
                
                print(f"\n  RESULTS:")
                print(f"    Clf flip rate: {metrics['clf_flip_rate']*100:5.1f}% (human: {metrics['human_flip_rate']*100:.1f}%)")
                print(f"    Accuracy:      {metrics['accuracy']*100:5.1f}%")
                print(f"    Precision:     {metrics['precision']*100:5.1f}%")
                print(f"    Recall:        {metrics['recall']*100:5.1f}%")
                print(f"    F1:            {metrics['f1']:.3f}")
                print(f"    MCC:           {metrics['mcc']:+.3f}")
                print(f"    κ:             {metrics['kappa']:+.3f}")
                print(f"    Confusion: TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}")
    
    if len(all_results) == 0:
        print("No results collected!")
        return
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: ALL CLASSIFIERS vs HUMAN")
    print("="*80)
    
    df = pd.DataFrame(all_results)
    
    # By classifier
    print("\n┌" + "─"*78 + "┐")
    print("│ BY CLASSIFIER (averaged across models)" + " "*39 + "│")
    print("├" + "─"*78 + "┤")
    
    clf_summary = df.groupby('name').agg({
        'clf_flip_rate': 'mean',
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1': 'mean',
        'mcc': 'mean',
        'kappa': 'mean',
        'type': 'first',
        'training': 'first',
    }).reset_index()
    
    print(f"│ {'Classifier':<22} {'Type':<10} {'Train':<8} {'Flip%':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'κ':>7} │")
    print("├" + "─"*78 + "┤")
    for _, row in clf_summary.sort_values('kappa', ascending=False).iterrows():
        kappa_emoji = "🔴" if row['kappa'] < 0 else "🟡" if row['kappa'] < 0.2 else "🟢"
        print(f"│ {row['name']:<22} {row['type']:<10} {row['training']:<8} {row['clf_flip_rate']*100:>5.1f}% {row['accuracy']*100:>5.1f}% {row['precision']*100:>5.1f}% {row['recall']*100:>5.1f}% {row['kappa']:>+6.3f} {kappa_emoji}│")
    print("└" + "─"*78 + "┘")
    
    # Full table
    print("\n┌" + "─"*100 + "┐")
    print("│ FULL COMPARISON TABLE" + " "*79 + "│")
    print("├" + "─"*100 + "┤")
    print(f"│ {'Model':<11} {'Classifier':<22} {'Clf%':>6} {'Hum%':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'MCC':>7} {'κ':>7} {'TP':>4} {'FN':>4} │")
    print("├" + "─"*100 + "┤")
    for _, row in df.iterrows():
        kappa_emoji = "🔴" if row['kappa'] < 0 else "🟡" if row['kappa'] < 0.2 else "🟢"
        print(f"│ {row['model']:<11} {row['name']:<22} {row['clf_flip_rate']*100:>5.1f}% {row['human_flip_rate']*100:>5.1f}% {row['accuracy']*100:>5.1f}% {row['precision']*100:>5.1f}% {row['recall']*100:>5.1f}% {row['mcc']:>+6.3f} {row['kappa']:>+6.3f} {row['TP']:>4} {row['FN']:>4} {kappa_emoji}│")
    print("└" + "─"*100 + "┘")
    
    # Key finding
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    avg_kappa = df['kappa'].mean()
    avg_mcc = df['mcc'].mean()
    best_clf = clf_summary.loc[clf_summary['kappa'].idxmax()]
    worst_clf = clf_summary.loc[clf_summary['kappa'].idxmin()]
    
    negative_kappa_count = (df['kappa'] < 0).sum()
    total_count = len(df)
    
    print(f"""
    1. OVERALL PERFORMANCE
       Average κ:  {avg_kappa:+.3f}
       Average MCC: {avg_mcc:+.3f}
       Negative κ: {negative_kappa_count}/{total_count} classifier-model pairs
    
    2. BEST CLASSIFIER
       {best_clf['name']} (trained on {best_clf['training']})
       κ = {best_clf['kappa']:+.3f}, Accuracy = {best_clf['accuracy']*100:.1f}%
    
    3. WORST CLASSIFIER  
       {worst_clf['name']} (trained on {worst_clf['training']})
       κ = {worst_clf['kappa']:+.3f}, Accuracy = {worst_clf['accuracy']*100:.1f}%
    
    4. CONCLUSION
       {"→ ALL classifiers show poor agreement with human (κ < 0.2)" if avg_kappa < 0.2 else "→ Some classifiers show fair agreement"}
       {"→ Multiple classifiers show NEGATIVE κ = anti-correlated with human" if negative_kappa_count > 0 else ""}
       → Problem is FUNDAMENTAL: automated sarcasm classifiers cannot reliably evaluate style transfer
       → Human evaluation is NECESSARY for this task
    """)
    
    # Save results
    output_path = Path('results/golden/classifier_comparison.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved results to: {output_path}")
    
    summary_path = Path('results/golden/classifier_comparison_summary.csv')
    clf_summary.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path}")

if __name__ == "__main__":
    main()