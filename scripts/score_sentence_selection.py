from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ambiguity_attribution.metrics import classification_metrics, gaussian_smooth, predict_sentence


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def max_sentence_score(item: dict, sigma: float) -> float:
    scores = gaussian_smooth(np.asarray(item["attributions"]), sigma=sigma)
    _, score = predict_sentence(scores, item["token_offsets"], item["prompt"])
    return float(score)


def evaluate_fold(
    ambiguous_items: list[dict],
    clear_items: list[dict],
    threshold: float,
    sigma: float,
) -> dict[str, float]:
    tp = 0
    fp = 0
    fn = 0

    for item in clear_items:
        scores = gaussian_smooth(np.asarray(item["attributions"]), sigma=sigma)
        prediction, _ = predict_sentence(scores, item["token_offsets"], item["prompt"], threshold=threshold)
        if prediction is not None:
            fp += 1

    for item in ambiguous_items:
        scores = gaussian_smooth(np.asarray(item["attributions"]), sigma=sigma)
        prediction, _ = predict_sentence(scores, item["token_offsets"], item["prompt"], threshold=threshold)
        if prediction is None:
            fn += 1
        elif prediction == item["target_sentence"]:
            tp += 1
        else:
            fn += 1

    return classification_metrics(tp, fp, fn)


def choose_threshold(
    ambiguous_train: list[dict],
    clear_train: list[dict],
    sigma: float,
) -> float:
    train_scores = [
        max_sentence_score(item, sigma)
        for item in ambiguous_train + clear_train
    ]
    if not train_scores:
        return 0.0
    min_score = min(train_scores)
    max_score = max(train_scores)
    thresholds = np.linspace(min_score, max_score, 200)

    best_threshold = thresholds[0]
    best_f1 = -1.0
    for threshold in thresholds:
        metrics = evaluate_fold(ambiguous_train, clear_train, float(threshold), sigma)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
    return best_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sentence-level selection from token attributions.")
    parser.add_argument("--ambiguous-results", type=Path, required=True)
    parser.add_argument("--clear-results", type=Path, required=True)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-path", type=Path, default=Path("sentence_selection_metrics.json"))
    args = parser.parse_args()

    ambiguous_results = load_results(args.ambiguous_results)
    clear_results = load_results(args.clear_results)
    shared_ids = sorted(set(ambiguous_results).intersection(clear_results))

    fold_metrics = []
    chosen_thresholds = []
    for fold_index in range(args.folds):
        eval_ids = {prompt_id for i, prompt_id in enumerate(shared_ids) if i % args.folds == fold_index}
        train_ids = [prompt_id for prompt_id in shared_ids if prompt_id not in eval_ids]

        ambiguous_train = [ambiguous_results[prompt_id] for prompt_id in train_ids]
        clear_train = [clear_results[prompt_id] for prompt_id in train_ids]
        ambiguous_eval = [ambiguous_results[prompt_id] for prompt_id in shared_ids if prompt_id in eval_ids]
        clear_eval = [clear_results[prompt_id] for prompt_id in shared_ids if prompt_id in eval_ids]

        threshold = choose_threshold(ambiguous_train, clear_train, sigma=args.sigma)
        chosen_thresholds.append(threshold)
        fold_metrics.append(evaluate_fold(ambiguous_eval, clear_eval, threshold, sigma=args.sigma))

    summary = {
        "thresholds": chosen_thresholds,
        "fold_metrics": fold_metrics,
        "mean": {
            metric: float(np.mean([fold[metric] for fold in fold_metrics]))
            for metric in ("precision", "recall", "f1")
        },
        "std": {
            metric: float(np.std([fold[metric] for fold in fold_metrics]))
            for metric in ("precision", "recall", "f1")
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
