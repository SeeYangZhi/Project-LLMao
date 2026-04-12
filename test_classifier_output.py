"""
Test what labels each classifier actually returns.
Run this FIRST before writing any evaluation logic.
"""
from transformers import pipeline

# Test sentences
SARCASTIC = "Oh great, another meeting that could have been an email"
NON_SARCASTIC = "The company announced quarterly earnings today"

CLASSIFIERS = [
    ("RoBERTa-Twitter", "cardiffnlp/twitter-roberta-base-irony"),
    ("DistilBERT-Reddit", "helinivan/english-sarcasm-detector"),
    ("RoBERTa-News", "jkhan447/sarcasm-detection-RoBerta-base-POS"),
]

print("=" * 70)
print("CLASSIFIER OUTPUT LABEL TEST")
print("=" * 70)

for name, model in CLASSIFIERS:
    print(f"\n--- {name} ---")
    print(f"Model: {model}")
    
    try:
        clf = pipeline("text-classification", model=model)
        
        # Test sarcastic input
        result_sarc = clf(SARCASTIC)[0]
        print(f"\nSarcastic input: '{SARCASTIC[:50]}...'")
        print(f"  Label: '{result_sarc['label']}'")
        print(f"  Score: {result_sarc['score']:.4f}")
        
        # Test non-sarcastic input
        result_non = clf(NON_SARCASTIC)[0]
        print(f"\nNon-sarcastic input: '{NON_SARCASTIC[:50]}...'")
        print(f"  Label: '{result_non['label']}'")
        print(f"  Score: {result_non['score']:.4f}")
        
        # Get all possible labels
        print(f"\nModel config labels: {clf.model.config.id2label}")
        
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "=" * 70)
print("NOW YOU KNOW THE EXACT LABELS!")
print("=" * 70)

# Add this to test actual headlines
headlines = [
    "Tiger Woods to make comeback by competing at 2020 Tour de France",
    "Study: Most Americans Get News From Facebook While Scrolling Past It",
]
for h in headlines:
    print(f"{h[:50]}... -> {clf(h)[0]}")