from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_OUTPUTS_DIR = PROJECT_ROOT / "model_outputs_clean"
DATA_DIR = PROJECT_ROOT / "data"
HELDOUT_FILE = DATA_DIR / "sarcasm_heldout_manual_annotated.jsonl"
HUMAN_EVAL_CSV = DATA_DIR / "human_eval.csv"
CROSS_VAL_FILE = DATA_DIR / "processed" / "intermediate" / "cross_validation_secondary.jsonl"

# Multi-classifier eval (commit 29dd7a7) — 3 sarcasm classifiers per model
MULTI_CLF_FILE = RESULTS_DIR / "all_models_multi_classifier.csv"
GOLDEN_DIR = RESULTS_DIR / "golden"
GOLDEN_SUMMARY_FILE = GOLDEN_DIR / "summary.csv"
GOLDEN_CLF_SUMMARY_FILE = GOLDEN_DIR / "summary_all_classifiers.csv"

CLASSIFIERS = ["RoBERTa-Twitter", "DistilBERT-Reddit", "RoBERTa-News"]
# Golden eval covers exactly these 3 models (140 samples × 2 annotators)
GOLDEN_MODELS = {
    "t5_base_joint": "T5 Base Joint",
    "t5_base_control": "T5 Base Control",
    "bart_base_rl": "BART RL",
}

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"

BART_CE_RL_PATH = PROJECT_ROOT / "outputs" / "bart-base-ce-rl" / "sar-to-non" / "best"

LLAMA_SYSTEM_PROMPT = (
    "You are a writing assistant. Rewrite sarcastic news headlines as neutral, "
    "factual equivalents that preserve the core meaning without irony or mockery. "
    "Respond with only the rewritten headline, no explanation."
)

BART_PREFIX = "rewrite to non-sarcastic: "

# Model display metadata
MODEL_REGISTRY = {
    "bart_base": {"display": "BART Base", "type": "main"},
    "bart_base_ce": {"display": "BART CE", "type": "main"},
    "bart_base_rl": {"display": "BART RL", "type": "main"},
    "bart_base_ce_rl": {"display": "BART CE+RL", "type": "main"},
    "llama_3_2_1b": {"display": "LLaMA 3.2 1B", "type": "main"},
    "t5_control": {"display": "T5 Control", "type": "baseline"},
    "t5_base_joint": {"display": "T5 Base Joint", "type": "baseline"},
    "joint": {"display": "Joint", "type": "baseline"},
    "ablation_without_sarcasm": {"display": "w/o Sarcasm", "type": "ablation"},
    "ablation_without_irony": {"display": "w/o Irony", "type": "ablation"},
    "ablation_without_satire": {"display": "w/o Satire", "type": "ablation"},
    "ablation_without_overstatement": {"display": "w/o Overstatement", "type": "ablation"},
    "ablation_without_understatement": {"display": "w/o Understatement", "type": "ablation"},
    "ablation_without_rhetorical_question": {"display": "w/o Rhetorical Q", "type": "ablation"},
}

METRIC_COLUMNS = [
    "hard_flipped", "flip_delta", "similarity", "bleu",
    "perplexity", "edit_dist_norm", "paraphrase_score",
]

STRATEGIES = [
    "sarcasm", "irony", "satire",
    "overstatement", "understatement", "rhetorical_question",
]

# Baselines from gold human rewrites (iSarcasmEval)
BASELINES = {
    "hard_flipped": 0.4614,
    "flip_delta": 0.0089,
    "similarity": 0.5625,
    "bleu": 0.1018,
    "perplexity": 230.85,
    "edit_dist_norm": 0.8187,
    "paraphrase_score": 0.0757,
}
