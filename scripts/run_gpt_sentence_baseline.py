from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from ambiguity_attribution.constants import DEFAULT_BASELINE_MODEL, PROBE_SPLIT_RANDOM_STATES
from ambiguity_attribution.data import load_combined_records, load_gold_records, make_probe_split
from ambiguity_attribution.metrics import classification_metrics
from ambiguity_attribution.prompts import (
    BASELINE_SYSTEM_PROMPT,
    normalize_baseline_response,
    render_gpt_sentence_baseline_prompt,
)


def load_evaluation_records(dataset: str, data_root: Path):
    if dataset == "gold":
        return [record for record in load_gold_records() if record.has_edit]
    if dataset != "combined":
        raise ValueError("The GPT sentence baseline is defined for the combined synthetic set or the gold set.")
    records = load_combined_records(data_root=data_root)
    split = make_probe_split(records, random_state=PROBE_SPLIT_RANDOM_STATES["combined"])
    held_out_ids = set(split.test_prompt_ids)
    return [record for record in records if record.prompt_id in held_out_ids and record.has_edit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GPT-5.4 sentence-selection baseline.")
    parser.add_argument("--dataset", choices=["combined", "gold"], default="combined")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-path", type=Path, default=Path("gpt_attributions/predictions.json"))
    parser.add_argument("--model", default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-completion-tokens", type=int, default=500)
    args = parser.parse_args()

    records = load_evaluation_records(args.dataset, args.data_root)
    client = OpenAI()
    predictions = {"clear": {}, "ambiguous": {}}

    for record in tqdm(records, desc="clear"):
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": render_gpt_sentence_baseline_prompt(record.clear_prompt)},
            ],
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
        )
        predictions["clear"][record.prompt_id] = normalize_baseline_response(response.choices[0].message.content)

    for record in tqdm(records, desc="ambiguous"):
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": render_gpt_sentence_baseline_prompt(record.ambiguous_prompt)},
            ],
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
        )
        predictions["ambiguous"][record.prompt_id] = normalize_baseline_response(response.choices[0].message.content)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")

    clear_correct = 0
    false_positives = 0
    true_positives = 0
    false_negatives = 0
    for record in records:
        clear_prediction = predictions["clear"][record.prompt_id]
        if clear_prediction == "NONE":
            clear_correct += 1
        else:
            false_positives += 1

        ambiguous_prediction = predictions["ambiguous"][record.prompt_id]
        if ambiguous_prediction == "NONE":
            false_negatives += 1
        elif (
            record.ambiguous_sentence.strip() in ambiguous_prediction.strip()
            or ambiguous_prediction.strip() in record.ambiguous_sentence.strip()
        ):
            true_positives += 1
        else:
            false_negatives += 1

    metrics = classification_metrics(true_positives, false_positives, false_negatives)
    metrics["clear_accuracy"] = clear_correct / len(records) if records else 0.0
    metrics["tp"] = true_positives
    metrics["fp"] = false_positives
    metrics["fn"] = false_negatives
    (args.output_path.parent / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
