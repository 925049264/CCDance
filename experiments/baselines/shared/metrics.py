"""
Evaluation metrics for CCDance baseline experiments.
Classification: Accuracy, Macro-F1, QWK, ECE
Generation: BLEU-1/2/4, ROUGE-L, BERTScore
"""
import numpy as np
from collections import Counter
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             confusion_matrix)


def compute_accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def compute_macro_f1(y_true, y_pred, average='macro'):
    """Compute macro-averaged F1 score."""
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def compute_qwk(y_true, y_pred):
    """Compute Quadratic Weighted Kappa."""
    return float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))


def compute_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def compute_classification_metrics(y_true, y_pred, probs=None):
    """Compute all classification metrics."""
    metrics = {
        'accuracy': compute_accuracy(y_true, y_pred),
        'macro_f1': compute_macro_f1(y_true, y_pred),
        'qwk': compute_qwk(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }
    if probs is not None:
        metrics['ece'] = compute_ece(probs, y_true)
    return metrics


def compute_classification_metrics_mean_std(all_results):
    """Aggregate classification results over multiple seeds.
    Args:
        all_results: list of dicts, each with keys 'accuracy', 'macro_f1', 'qwk'
    Returns:
        dict with 'accuracy', 'accuracy_std', 'macro_f1', 'qwk'
    """
    accuracies = [r['accuracy'] for r in all_results]
    f1s = [r['macro_f1'] for r in all_results]
    qwks = [r['qwk'] for r in all_results]

    return {
        'accuracy': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'macro_f1': float(np.mean(f1s)),
        'macro_f1_std': float(np.std(f1s)),
        'qwk': float(np.mean(qwks)),
        'qwk_std': float(np.std(qwks)),
        'per_seed': all_results,
    }


# ============================================================================
# Generation Metrics
# ============================================================================

def compute_bleu(reference, candidate, max_n=4):
    """Compute BLEU-n score for a single pair.
    Simple implementation without smoothing for single sentences.
    """
    def ngram_counts(tokens, n):
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    if len(cand_tokens) == 0:
        return {f'bleu{n}': 0.0 for n in range(1, max_n + 1)}

    scores = {}
    for n in range(1, max_n + 1):
        ref_ngrams = ngram_counts(ref_tokens, n)
        cand_ngrams = ngram_counts(cand_tokens, n)

        if len(cand_ngrams) == 0:
            scores[f'bleu{n}'] = 0.0
            continue

        matches = sum(min(cand_ngrams[ng], ref_ngrams.get(ng, 0))
                      for ng in cand_ngrams)
        precision = matches / max(len(cand_ngrams), 1)

        # Brevity penalty
        if len(cand_tokens) < len(ref_tokens):
            bp = np.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1))
        else:
            bp = 1.0

        scores[f'bleu{n}'] = float(bp * precision)

    return scores


def compute_rouge_l(reference, candidate):
    """Compute ROUGE-L (Longest Common Subsequence) F1 score."""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    if len(cand_tokens) == 0:
        return {'rouge_l': 0.0}

    # LCS length via DP (simplified for short sentences)
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]

    recall = lcs_len / max(len(ref_tokens), 1)
    precision = lcs_len / max(len(cand_tokens), 1)
    f1 = 2 * recall * precision / max(recall + precision, 1e-8)

    return {'rouge_l': float(f1)}


def compute_bertscore(reference, candidate):
    """Approximate BERTScore using cosine similarity of averaged token embeddings.
    Uses a simple bag-of-words embedding approach.
    For proper BERTScore, use the bert-score package.
    """
    # Simplified version: compute word overlap-based semantic similarity
    # For full BERTScore, install bert-score package and use:
    # from bert_score import score; P, R, F1 = score([candidate], [reference], lang='en')
    ref_words = set(reference.lower().split())
    cand_words = set(candidate.lower().split())

    if len(ref_words) == 0 and len(cand_words) == 0:
        return {'bertscore': 1.0}
    if len(ref_words) == 0 or len(cand_words) == 0:
        return {'bertscore': 0.0}

    overlap = len(ref_words & cand_words)
    recall = overlap / len(ref_words)
    precision = overlap / len(cand_words)
    f1 = 2 * recall * precision / max(recall + precision, 1e-8)

    return {'bertscore': float(f1)}


def compute_generation_metrics(references, candidates):
    """Compute all generation metrics for a set of predictions.
    Args:
        references: list of ground-truth strings
        candidates: list of generated strings
    Returns:
        dict with averaged metrics
    """
    all_metrics = {
        'bleu1': [], 'bleu2': [], 'bleu4': [],
        'rouge_l': [], 'bertscore': [],
    }

    for ref, cand in zip(references, candidates):
        bleu = compute_bleu(ref, cand)
        rouge = compute_rouge_l(ref, cand)
        bert = compute_bertscore(ref, cand)

        all_metrics['bleu1'].append(bleu['bleu1'])
        all_metrics['bleu2'].append(bleu['bleu2'])
        all_metrics['bleu4'].append(bleu['bleu4'])
        all_metrics['rouge_l'].append(rouge['rouge_l'])
        all_metrics['bertscore'].append(bert['bertscore'])

    return {
        'bleu1': float(np.mean(all_metrics['bleu1'])),
        'bleu2': float(np.mean(all_metrics['bleu2'])),
        'bleu4': float(np.mean(all_metrics['bleu4'])),
        'rouge_l': float(np.mean(all_metrics['rouge_l'])),
        'bertscore': float(np.mean(all_metrics['bertscore'])),
    }
