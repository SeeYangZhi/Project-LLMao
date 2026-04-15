"""
Sarcasm Rewriting Evaluation Pipeline (CS4248 NLP Project)
Usage:
    python scripts/eval_pipeline.py --input data.csv --output results.csv [--gemini_key KEY]
    
Multi-classifier mode:
    python scripts/eval_pipeline.py --input data.csv --output results.csv --multi_classifier

Input CSV columns: [id, input, output, subtype]
"""
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import json
import os

# ============================================================
# CLASSIFIERS CONFIG
# ============================================================
CLASSIFIERS = [
    {
        'name': 'RoBERTa-Twitter',
        'model': 'cardiffnlp/twitter-roberta-base-irony',
        'type': 'RoBERTa',
        'training': 'Twitter',
        'sarcastic_labels': ['irony'],
        'non_sarcastic_labels': ['non_irony'],
    },
    {
        'name': 'Bert-Kaggle',
        'model': 'helinivan/english-sarcasm-detector',
        'type': 'Bert',
        'training': 'Kaggle',
        'sarcastic_labels': ['LABEL_1', 'sarcasm', 'sarcastic'],
        'non_sarcastic_labels': ['LABEL_0', 'not_sarcasm', 'normal'],
    },
    {
        'name': 'RoBERTa-News',
        'model': 'jkhan447/sarcasm-detection-RoBerta-base-POS',
        'type': 'RoBERTa',
        'training': 'News',
        'sarcastic_labels': ['LABEL_1', 'sarcastic', 'sarcasm'],
        'non_sarcastic_labels': ['LABEL_0', 'not_sarcastic'],
    },
]

def load_classifier(model_name, device):
    """Load a classifier pipeline."""
    from transformers import pipeline as hf_pipeline
    try:
        clf = hf_pipeline(
            "text-classification",
            model=model_name,
            device=0 if device == "cuda" else -1,
            truncation=True,
            max_length=512
        )
        return clf
    except Exception as e:
        print(f"  ERROR loading {model_name}: {e}")
        return None

def predict_sarcasm_label(clf, text, sarcastic_labels, non_sarcastic_labels):
    """
    Predict if text is sarcastic.
    Returns (is_sarcastic, confidence, irony_score)
    """
    try:
        result = clf(text[:512], truncation=True, max_length=128)[0]
        label = result['label']
        score = result['score']
        
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
        
        # Fallback inference
        if is_sarcastic is None:
            if 'non' in label_lower or 'not' in label_lower or 'normal' in label_lower:
                is_sarcastic = False
            elif 'irony' in label_lower or 'sarcas' in label_lower:
                is_sarcastic = True
            else:
                is_sarcastic = '1' in label
        
        # Compute irony score (probability of being sarcastic)
        irony_score = score if is_sarcastic else (1 - score)
        
        return is_sarcastic, score, irony_score
            
    except Exception as e:
        return None, None, None

def compute_flip_rate(inputs, outputs, classifier, clf_info=None):
    """
    METRIC 1: Sarcasm Flip Rate.
    
    If clf_info is provided, uses smart label detection.
    Otherwise, uses legacy behavior for backward compatibility.
    """
    clf_name = clf_info['name'] if clf_info else "classifier"
    print(f"\n[1/7] Computing Sarcasm Flip Rate ({clf_name})...")
    
    results = []
    
    if clf_info:
        # New multi-classifier mode
        sarcastic_labels = clf_info['sarcastic_labels']
        non_sarcastic_labels = clf_info['non_sarcastic_labels']
        
        for inp, out in tqdm(zip(inputs, outputs), total=len(inputs)):
            inp_is_sarc, inp_conf, inp_irony = predict_sarcasm_label(
                classifier, inp, sarcastic_labels, non_sarcastic_labels
            )
            out_is_sarc, out_conf, out_irony = predict_sarcasm_label(
                classifier, out, sarcastic_labels, non_sarcastic_labels
            )
            
            if inp_is_sarc is None or out_is_sarc is None:
                hard_flipped = 0
                flip_delta = 0
                inp_irony = 0.5
                out_irony = 0.5
            else:
                # Flipped = input was sarcastic AND output is NOT sarcastic
                hard_flipped = 1 if (inp_is_sarc and not out_is_sarc) else 0
                flip_delta = inp_irony - out_irony
            
            results.append({
                "hard_flipped": hard_flipped,
                "flip_delta": flip_delta,
                "inp_irony_score": inp_irony,
                "out_irony_score": out_irony,
            })
    else:
        # Legacy mode (original behavior)
        for inp, out in tqdm(zip(inputs, outputs), total=len(inputs)):
            inp_result = classifier(inp, truncation=True, max_length=128)[0]
            inp_label = inp_result["label"].lower()
            inp_score = inp_result["score"]
            inp_irony_score = inp_score if "irony" in inp_label or inp_label == "sarcastic" else 1 - inp_score
            
            out_result = classifier(out, truncation=True, max_length=128)[0]
            out_label = out_result["label"].lower()
            out_score = out_result["score"]
            out_irony_score = out_score if "irony" in out_label or out_label == "sarcastic" else 1 - out_score
            
            hard_flipped = 1 if (
                ("irony" in inp_label or inp_label == "sarcastic") and
                ("non" in out_label or out_label == "non-sarcastic")
            ) else 0
            flip_delta = inp_irony_score - out_irony_score
            
            results.append({
                "hard_flipped": hard_flipped,
                "flip_delta": flip_delta,
                "inp_irony_score": inp_irony_score,
                "out_irony_score": out_irony_score,
            })
    
    flip_rate = sum(r["hard_flipped"] for r in results) / len(results)
    mean_flip_delta = np.mean([r["flip_delta"] for r in results])
    
    print(f"    Hard Flip Rate:   {flip_rate:.2%}")
    print(f"    Mean Flip Delta:  {mean_flip_delta:+.4f}")
    
    return {
        "flip_rate": flip_rate,
        "mean_flip_delta": mean_flip_delta,
        "per_sample": results,
        "classifier_name": clf_name,
    }

def compute_semantic_similarity(inputs, outputs, sim_model):
    """
    METRIC 2: Semantic Similarity using sentence-transformers and cosine similarity.
    """
    print("[2/7] Computing semantic similarity...")
    from sklearn.metrics.pairwise import cosine_similarity
    inp_embeddings = sim_model.encode(inputs, show_progress_bar=True, batch_size=32)
    out_embeddings = sim_model.encode(outputs, show_progress_bar=True, batch_size=32)
    per_sample = []
    for i in range(len(inputs)):
        sim = cosine_similarity(
            inp_embeddings[i].reshape(1, -1),
            out_embeddings[i].reshape(1, -1),
        )[0][0]
        per_sample.append(float(sim))
    mean_similarity = float(np.mean(per_sample))
    return {
        "mean_similarity": mean_similarity,
        "per_sample": per_sample,
    }

def compute_perplexity(outputs, lm_model, lm_tokenizer, device):
    """
    METRIC 3: Perplexity using GPT-2.
    """
    print("[3/7] Computing perplexity...")
    per_sample = []
    lm_model.eval()
    for text in tqdm(outputs, desc="Perplexity"):
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
    filtered = [p for p in per_sample if p < 10000 and not np.isnan(p)]
    mean_perplexity = float(np.mean(filtered)) if filtered else float("nan")
    return {
        "mean_perplexity": mean_perplexity,
        "per_sample": per_sample,
    }

def compute_bleu(inputs, outputs, references=None):
    """
    METRIC 4: BLEU score using nltk.
    """
    print("[4/7] Computing BLEU score...")
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoothie = SmoothingFunction().method1
    per_sample = []
    if references is not None:
        mode = "vs_reference"
        for out, ref in tqdm(zip(outputs, references), total=len(outputs), desc="BLEU (ref)"):
            ref_tokens = ref.split()
            out_tokens = out.split()
            score = sentence_bleu([ref_tokens], out_tokens, smoothing_function=smoothie)
            per_sample.append(float(score))
    else:
        mode = "vs_input"
        for inp, out in tqdm(zip(inputs, outputs), total=len(inputs), desc="BLEU (input)"):
            inp_tokens = inp.split()
            out_tokens = out.split()
            score = sentence_bleu([inp_tokens], out_tokens, smoothing_function=smoothie)
            per_sample.append(float(score))
    mean_bleu = float(np.mean(per_sample))
    return {
        "mean_bleu": mean_bleu,
        "mode": mode,
        "per_sample": per_sample,
    }

def compute_llm_judge(inputs, outputs, gemini_key, sample_size=50, flip_per_sample=None):
    """
    METRIC 5: LLM-as-Judge using Gemini 2.5 Flash.
    """
    print(f"\n[5/7] Running LLM-as-Judge (Gemini, batch of {sample_size} samples)...")
    import random
    from google import genai
    client = genai.Client(api_key=gemini_key)
    indices = list(range(len(inputs)))
    if len(inputs) > sample_size:
        indices = random.sample(indices, sample_size)
    indices = sorted(indices)
    pairs_text = ""
    for rank, i in enumerate(indices):
        inp = inputs[i].replace('"', "'")
        out = outputs[i].replace('"', "'")
        pairs_text += f'{rank}. Input: "{inp}" | Output: "{out}"\n'
    prompt = f"""You are evaluating a sarcasm-removal rewriting system.
Each pair shows: (sarcastic input headline, non-sarcastic output rewrite).
Rate each pair on 3 dimensions (1-5):
- sarcasm_removed: Did sarcasm get successfully removed? (1=still very sarcastic, 5=fully non-sarcastic)
- meaning_preserved: Does the output keep the same topic/meaning? (1=completely different, 5=same meaning)
- fluency: Is the output natural grammatical English? (1=broken/unnatural, 5=perfectly fluent)
Pairs to evaluate:
{pairs_text}
Respond ONLY with a valid JSON array, no explanation, no markdown:
[
  {{"id": 0, "sarcasm_removed": X, "meaning_preserved": X, "fluency": X}},
  {{"id": 1, "sarcasm_removed": X, "meaning_preserved": X, "fluency": X}},
  ...
]"""
    print(f"    Sending {len(indices)} pairs in one API call...")
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        scored = json.loads(raw)
    except Exception as e:
        print(f"    ERROR: Could not get/parse Gemini response: {e}")
        return {}
    results = []
    for entry in scored:
        rank = entry["id"]
        if rank >= len(indices):
            continue
        original_idx = indices[rank]
        results.append({
            "dataset_id": original_idx,
            "sarcasm_removed": entry.get("sarcasm_removed", 3),
            "meaning_preserved": entry.get("meaning_preserved", 3),
            "fluency": entry.get("fluency", 3),
        })
    if not results:
        print("    ERROR: No results parsed.")
        return {}
    mean_sarcasm = np.mean([r["sarcasm_removed"] for r in results])
    mean_meaning = np.mean([r["meaning_preserved"] for r in results])
    mean_fluency = np.mean([r["fluency"] for r in results])
    
    from sklearn.metrics import cohen_kappa_score
    judge_binary = [1 if r["sarcasm_removed"] >= 4 else 0 for r in results]
    kappa = float("nan")
    if flip_per_sample is not None:
        classifier_binary = []
        for r in results:
            idx = r["dataset_id"]
            if idx < len(flip_per_sample):
                classifier_binary.append(flip_per_sample[idx]["hard_flipped"])
            else:
                classifier_binary.append(0)
        if len(set(judge_binary)) > 1 or len(set(classifier_binary)) > 1:
            try:
                kappa = cohen_kappa_score(classifier_binary, judge_binary)
            except Exception:
                kappa = float("nan")
        else:
            kappa = 1.0
    
    print(f"    Samples judged: {len(results)}")
    print(f"    LLM Judge - Sarcasm Removed: {mean_sarcasm:.2f}/5")
    print(f"    LLM Judge - Meaning Preserved: {mean_meaning:.2f}/5")
    print(f"    LLM Judge - Fluency: {mean_fluency:.2f}/5")
    
    return {
        "mean_sarcasm_removed": mean_sarcasm,
        "mean_meaning_preserved": mean_meaning,
        "mean_fluency": mean_fluency,
        "cohens_kappa": kappa,
        "per_sample": results,
        "judged_indices": indices,
    }

def word_levenshtein(s1, s2):
    """Word-level Levenshtein distance."""
    w1 = s1.split()
    w2 = s2.split()
    n, m = len(w1), len(w2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if w1[i - 1] == w2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]

def compute_edit_distance(inputs, outputs):
    """
    METRIC 6: Word-level edit distance.
    """
    print("[6/7] Computing word-level edit distance...")
    per_sample_raw = []
    per_sample_norm = []
    for inp, out in tqdm(zip(inputs, outputs), total=len(inputs), desc="Edit distance"):
        raw = word_levenshtein(inp, out)
        max_len = max(len(inp.split()), len(out.split()), 1)
        norm = raw / max_len
        per_sample_raw.append(raw)
        per_sample_norm.append(norm)
    return {
        "mean_raw": float(np.mean(per_sample_raw)),
        "mean_normalized": float(np.mean(per_sample_norm)),
        "per_sample_raw": per_sample_raw,
        "per_sample_norm": per_sample_norm,
    }

def compute_paraphrase_score(sim_per_sample, bleu_per_sample):
    """
    METRIC 7: Paraphrase score.
    """
    print("[7/7] Computing paraphrase score...")
    per_sample = []
    for sim, bleu in zip(sim_per_sample, bleu_per_sample):
        clamped_sim = max(0.0, min(1.0, sim))
        score = clamped_sim * (1 - bleu)
        per_sample.append(float(score))
    return {
        "mean_paraphrase_score": float(np.mean(per_sample)),
        "per_sample": per_sample,
    }

def run_multi_classifier_evaluation(inputs, outputs, device):
    """
    Run all classifiers and compare results.
    """
    print("\n" + "=" * 70)
    print("MULTI-CLASSIFIER EVALUATION")
    print("=" * 70)
    
    all_results = []
    
    for clf_info in CLASSIFIERS:
        print(f"\n--- Loading: {clf_info['name']} ({clf_info['type']}, {clf_info['training']}) ---")
        clf = load_classifier(clf_info['model'], device)
        
        if clf is None:
            print(f"  Skipping {clf_info['name']} (failed to load)")
            continue
        
        flip_results = compute_flip_rate(inputs, outputs, clf, clf_info=clf_info)
        
        all_results.append({
            'name': clf_info['name'],
            'type': clf_info['type'],
            'training': clf_info['training'],
            'flip_rate': flip_results['flip_rate'],
            'mean_flip_delta': flip_results['mean_flip_delta'],
            'per_sample': flip_results['per_sample'],
        })
    
    # Print comparison
    print("\n" + "=" * 70)
    print("CLASSIFIER COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Classifier':<25} {'Type':<12} {'Training':<10} {'Flip Rate':>10} {'Flip Delta':>12}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['name']:<25} {r['type']:<12} {r['training']:<10} {r['flip_rate']*100:>9.1f}% {r['mean_flip_delta']:>+11.4f}")
    
    return all_results

def export_human_eval_samples(df, output_path, top_n=30):
    """Export top-N most suspicious samples for human evaluation."""
    print(f"\nExporting top {top_n} samples for human evaluation...")
    df = df.copy()
    df["suspicion_score"] = df["paraphrase_score"] + (1 - df["flip_delta"].abs())
    df_top = df.nlargest(top_n, "suspicion_score").copy()
    
    flag_reasons = []
    for _, row in df_top.iterrows():
        reasons = []
        if row.get("paraphrase_score", 0) > 0.05:
            reasons.append("high_paraphrase")
        if abs(row.get("flip_delta", 1)) < 0.1:
            reasons.append("low_flip_delta")
        if row.get("similarity", 0) > 0.8 and row.get("bleu", 0) > 0.3:
            reasons.append("very_similar_wording")
        flag_reasons.append("; ".join(reasons) if reasons else "none")
    df_top["flag_reason"] = flag_reasons
    df_top["annotator_1_sarcasm_removed"] = ""
    df_top["annotator_1_meaning_preserved"] = ""
    df_top["annotator_2_sarcasm_removed"] = ""
    df_top["annotator_2_meaning_preserved"] = ""
    
    base, ext = os.path.splitext(output_path)
    human_eval_path = f"{base}_human_eval{ext}"
    df_top.to_csv(human_eval_path, index=False)
    print(f"Human eval samples saved to: {human_eval_path}")
    return human_eval_path

def compute_subtype_breakdown(df, flip_per_sample, sim_per_sample, bleu_per_sample,
                               edit_per_sample=None, paraphrase_per_sample=None):
    """Compute per-subtype breakdown of metrics."""
    df = df.copy()
    df["flip_rate"] = [s["hard_flipped"] for s in flip_per_sample]
    df["similarity"] = sim_per_sample
    df["bleu"] = bleu_per_sample
    if edit_per_sample is not None:
        df["edit_dist"] = edit_per_sample
    if paraphrase_per_sample is not None:
        df["para_score"] = paraphrase_per_sample
    
    grouped = df.groupby("subtype")
    rows = []
    for name, group in grouped:
        row = {
            "subtype": name,
            "count": len(group),
            "flip_rate": f"{group['flip_rate'].mean() * 100:.1f}%",
            "mean_sim": f"{group['similarity'].mean():.3f}",
            "mean_bleu": f"{group['bleu'].mean():.3f}",
        }
        if edit_per_sample is not None:
            row["mean_edit_dist"] = f"{group['edit_dist'].mean():.2f}"
        if paraphrase_per_sample is not None:
            row["mean_para_score"] = f"{group['para_score'].mean():.4f}"
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Sarcasm Rewriting Evaluation Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output", type=str, default="results.csv", help="Path to output CSV")
    parser.add_argument("--skip_judge", action="store_true", help="Skip LLM judge evaluation")
    parser.add_argument("--judge_sample", type=int, default=50, help="Number of samples for LLM judge")
    parser.add_argument("--gemini_key", type=str, default=os.environ.get("GEMINI_API_KEY", None))
    parser.add_argument("--references", type=str, default=None, help="Path to references CSV for BLEU")
    parser.add_argument("--human_eval_n", type=int, default=30, help="Samples for human eval export")
    parser.add_argument("--multi_classifier", action="store_true", 
                        help="Run all 3 classifiers and compare")
    parser.add_argument("--classifier", type=str, default="twitter",
                        choices=["twitter", "reddit", "news", "all"],
                        help="Which classifier to use (default: twitter)")
    args = parser.parse_args()
    
    # Load data
    print(f"\nLoading data from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    if "subtype" in df.columns:
        print(f"  Subtypes: {df['subtype'].value_counts().to_dict()}")
    
    df["output"] = df["output"].fillna(df["input"])
    inputs = df["input"].tolist()
    outputs = df["output"].tolist()
    
    # Load references if provided
    references = None
    if args.references:
        ref_df = pd.read_csv(args.references)
        references = ref_df["reference"].tolist()
        print(f"  Loaded {len(references)} references")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Multi-classifier mode
    if args.multi_classifier or args.classifier == "all":
        multi_results = run_multi_classifier_evaluation(inputs, outputs, device)
        
        # Save multi-classifier results
        multi_df = pd.DataFrame([{
            'classifier': r['name'],
            'type': r['type'],
            'training': r['training'],
            'flip_rate': r['flip_rate'],
            'mean_flip_delta': r['mean_flip_delta'],
        } for r in multi_results])
        
        base, ext = os.path.splitext(args.output)
        multi_path = f"{base}_multi_classifier{ext}"
        multi_df.to_csv(multi_path, index=False)
        print(f"\nMulti-classifier results saved to: {multi_path}")
        
        # Use first (Twitter) classifier for main results
        flip_results = {
            'flip_rate': multi_results[0]['flip_rate'],
            'mean_flip_delta': multi_results[0]['mean_flip_delta'],
            'per_sample': multi_results[0]['per_sample'],
        }
        
        # Save per-sample results for each classifier
        for r in multi_results:
            clf_name = r['name'].lower().replace('-', '_').replace(' ', '_')
            df[f"hard_flipped_{clf_name}"] = [s["hard_flipped"] for s in r['per_sample']]
            df[f"flip_delta_{clf_name}"] = [s["flip_delta"] for s in r['per_sample']]
    else:
        # Single classifier mode
        print("\nLoading models...")
        from transformers import pipeline as hf_pipeline, AutoModelForCausalLM, AutoTokenizer
        
        # Select classifier
        clf_map = {
            'twitter': CLASSIFIERS[0],
            'reddit': CLASSIFIERS[1],
            'news': CLASSIFIERS[2],
        }
        clf_info = clf_map.get(args.classifier, CLASSIFIERS[0])
        
        print(f"  Loading classifier: {clf_info['name']}...")
        classifier = load_classifier(clf_info['model'], device)
        
        if classifier is None:
            print("ERROR: Could not load classifier")
            return
        
        # METRIC 1: Flip Rate
        flip_results = compute_flip_rate(inputs, outputs, classifier, clf_info=clf_info)
    
    # Load other models
    print("\nLoading other models...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    
    print("  Loading sentence-transformer...")
    sim_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    print("  Loading GPT-2...")
    lm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    lm_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    print("Models loaded.\n")
    
    # METRIC 2: Semantic Similarity
    sim_results = compute_semantic_similarity(inputs, outputs, sim_model)
    print(f"  Mean similarity: {sim_results['mean_similarity']:.4f}\n")
    
    # METRIC 3: Perplexity
    ppl_results = compute_perplexity(outputs, lm_model, lm_tokenizer, device)
    print(f"  Mean perplexity: {ppl_results['mean_perplexity']:.2f}\n")
    
    # METRIC 4: BLEU
    bleu_results = compute_bleu(inputs, outputs, references=references)
    print(f"  Mean BLEU ({bleu_results['mode']}): {bleu_results['mean_bleu']:.4f}\n")
    
    # METRIC 5: LLM Judge
    judge_results = None
    if not args.skip_judge and args.gemini_key:
        judge_results = compute_llm_judge(
            inputs, outputs, args.gemini_key,
            sample_size=args.judge_sample,
            flip_per_sample=flip_results["per_sample"],
        )
    elif args.skip_judge:
        print("[5/7] Skipping LLM judge (--skip_judge flag set)\n")
    else:
        print("[5/7] Skipping LLM judge (no Gemini API key provided)\n")
    
    # METRIC 6: Edit Distance
    edit_results = compute_edit_distance(inputs, outputs)
    print(f"  Mean edit distance (norm): {edit_results['mean_normalized']:.4f}\n")
    
    # METRIC 7: Paraphrase Score
    para_results = compute_paraphrase_score(sim_results["per_sample"], bleu_results["per_sample"])
    print(f"  Mean paraphrase score: {para_results['mean_paraphrase_score']:.4f}\n")
    
    # Subtype breakdown
    if "subtype" in df.columns:
        print("=" * 60)
        print("SUBTYPE BREAKDOWN")
        print("=" * 60)
        breakdown = compute_subtype_breakdown(
            df,
            flip_results["per_sample"],
            sim_results["per_sample"],
            bleu_results["per_sample"],
            edit_per_sample=edit_results["per_sample_norm"],
            paraphrase_per_sample=para_results["per_sample"],
        )
        print(breakdown.to_string(index=False))
        print()
    
    # Save per-sample results
    results_df = df.copy()
    results_df["hard_flipped"] = [s["hard_flipped"] for s in flip_results["per_sample"]]
    results_df["inp_irony_score"] = [s["inp_irony_score"] for s in flip_results["per_sample"]]
    results_df["out_irony_score"] = [s["out_irony_score"] for s in flip_results["per_sample"]]
    results_df["flip_delta"] = [s["flip_delta"] for s in flip_results["per_sample"]]
    results_df["similarity"] = sim_results["per_sample"]
    results_df["bleu"] = bleu_results["per_sample"]
    results_df["perplexity"] = ppl_results["per_sample"]
    results_df["edit_dist_raw"] = edit_results["per_sample_raw"]
    results_df["edit_dist_norm"] = edit_results["per_sample_norm"]
    results_df["paraphrase_score"] = para_results["per_sample"]
    
    results_df.to_csv(args.output, index=False)
    print(f"Per-sample results saved to: {args.output}")
    
    # Export human eval samples
    export_human_eval_samples(results_df, args.output, top_n=args.human_eval_n)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Flip rate:             {flip_results['flip_rate'] * 100:.1f}%")
    print(f"  Mean flip delta:       {flip_results['mean_flip_delta']:.4f}")
    print(f"  Semantic similarity:   {sim_results['mean_similarity']:.4f}")
    print(f"  Perplexity (GPT-2):    {ppl_results['mean_perplexity']:.2f}")
    print(f"  BLEU ({bleu_results['mode']:>12s}):  {bleu_results['mean_bleu']:.4f}")
    print(f"  Edit dist (norm):      {edit_results['mean_normalized']:.4f}")
    print(f"  Paraphrase score:      {para_results['mean_paraphrase_score']:.4f}")
    if judge_results:
        print(f"  LLM sarcasm_removed:   {judge_results['mean_sarcasm_removed']:.2f}")
        print(f"  LLM meaning_preserved: {judge_results['mean_meaning_preserved']:.2f}")
        print(f"  LLM fluency:           {judge_results['mean_fluency']:.2f}")
    print("=" * 60)
    print("Done.")

if __name__ == "__main__":
    main()