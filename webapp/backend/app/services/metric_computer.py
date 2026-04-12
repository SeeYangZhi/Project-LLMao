"""Lightweight real-time metric computation for the playground."""

from __future__ import annotations

import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def edit_distance_normalized(text_a: str, text_b: str) -> float:
    words_a = text_a.lower().split()
    words_b = text_b.lower().split()
    m, n = len(words_a), len(words_b)
    if m == 0 and n == 0:
        return 0.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if words_a[i - 1] == words_b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] / max(m, n)


def bleu_score(reference: str, hypothesis: str) -> float:
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    if len(hyp_tokens) == 0:
        return 0.0
    smoothie = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)


def compute_metrics(input_text: str, output_text: str) -> dict:
    edit_dist = edit_distance_normalized(input_text, output_text)
    bleu = bleu_score(input_text, output_text)
    return {
        "edit_dist_norm": round(edit_dist, 4),
        "bleu_vs_input": round(bleu, 4),
        "paraphrase_score": round(bleu * min(1.0, max(0.0, 1.0 - edit_dist)), 4),
    }
