"""Evaluation and sentence-level scoring helpers."""

from __future__ import annotations

import re

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn_prg.metrics import precision_recall_gain_curve


def gaussian_smooth(scores, sigma: float = 3.0) -> np.ndarray:
    array = np.asarray(scores)
    if sigma == 0:
        return array
    return gaussian_filter1d(array, sigma=sigma)


def span_auroc(scores, mask) -> float:
    return float(roc_auc_score(np.asarray(mask).astype(int), np.asarray(scores)))


def span_auprc(scores, mask) -> float:
    return float(average_precision_score(np.asarray(mask).reshape(-1), np.asarray(scores).reshape(-1)))


def normalized_auprc(scores, mask) -> float:
    mask_array = np.asarray(mask).reshape(-1)
    base_rate = float(mask_array.mean())
    if base_rate >= 1.0:
        return 1.0
    return float((span_auprc(scores, mask_array) - base_rate) / (1 - base_rate))


def span_auprg(scores, mask) -> float:
    precision_gain, recall_gain = precision_recall_gain_curve(
        np.asarray(mask).astype(int),
        np.asarray(scores),
    )
    return float(np.trapz(precision_gain, recall_gain))


def sentence_spans(prompt: str) -> list[tuple[str, int, int]]:
    pieces = re.split(r"(?<=[.!?])\s+", prompt.strip())
    spans = []
    cursor = 0
    for piece in pieces:
        if not piece:
            continue
        start = prompt.find(piece, cursor)
        if start < 0:
            start = cursor
        end = start + len(piece)
        spans.append((piece, start, end))
        cursor = end
    return spans


def mean_sentence_scores(
    scores,
    token_offsets: list[tuple[int, int]],
    prompt: str,
) -> list[tuple[str, float]]:
    score_array = np.asarray(scores)
    sentence_scores = []
    for sentence, sentence_start, sentence_end in sentence_spans(prompt):
        indices = [
            index
            for index, (token_start, token_end) in enumerate(token_offsets)
            if not (token_end <= sentence_start or token_start >= sentence_end)
        ]
        if indices:
            sentence_score = float(score_array[indices].mean())
        else:
            sentence_score = float("-inf")
        sentence_scores.append((sentence, sentence_score))
    return sentence_scores


def predict_sentence(
    scores,
    token_offsets: list[tuple[int, int]],
    prompt: str,
    threshold: float | None = None,
) -> tuple[str | None, float]:
    sentence_scores = mean_sentence_scores(scores, token_offsets, prompt)
    prediction, score = max(sentence_scores, key=lambda item: item[1])
    if threshold is not None and score < threshold:
        return None, score
    return prediction, score


def classification_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    denom = 2 * tp + fp + fn
    f1 = (2 * tp) / denom if denom else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
