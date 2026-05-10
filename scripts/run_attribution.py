from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ambiguity_attribution.attribution import (
    average_embedding_baseline,
    gradient_x_input_attribution,
    integrated_gradients_attribution,
    residual_integrated_gradients,
)
from ambiguity_attribution.constants import MID_LAYER_OPTIONS, PROBE_LAYERS, PROBE_SPLIT_RANDOM_STATES
from ambiguity_attribution.data import (
    load_combined_records,
    load_domain_records,
    load_gold_records,
    make_probe_split,
)
from ambiguity_attribution.modeling import load_model, mask_for_target_text, slice_prompt_tokens
from ambiguity_attribution.probe import load_probe_weights


def load_records_for_dataset(dataset: str, data_root: Path):
    if dataset == "combined":
        return load_combined_records(data_root=data_root)
    if dataset == "gold":
        return load_gold_records()
    return load_domain_records(dataset, data_root=data_root)


def held_out_records(dataset: str, data_root: Path):
    records = load_records_for_dataset(dataset, data_root)
    if dataset == "gold":
        return records
    split = make_probe_split(records, random_state=PROBE_SPLIT_RANDOM_STATES[dataset])
    record_by_id = {record.prompt_id: record for record in records}
    return [record_by_id[prompt_id] for prompt_id in split.test_prompt_ids]


def default_output_dir(dataset: str, method: str, probe_dataset: str, target: str) -> Path:
    base = Path(f"{dataset}_attributions")
    if target == "clear":
        base = Path(f"{dataset}_attributions_clear")
    return base / probe_dataset / method


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def serialize_example(record, prompt_tokens, token_offsets, mask, attributions, prompt_text, target_sentence):
    return {
        "prompt_id": record.prompt_id,
        "source_id": record.source_id,
        "domain": record.domain,
        "prompt": prompt_text,
        "target_sentence": target_sentence,
        "tokens": prompt_tokens,
        "token_offsets": [list(offset) for offset in token_offsets],
        "mask": mask.tolist(),
        "attributions": attributions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run attribution over a held-out evaluation set.")
    parser.add_argument("--evaluation-dataset", choices=["code", "math", "writing", "combined", "gold"], required=True)
    parser.add_argument("--probe-dataset", choices=["code", "math", "writing", "combined"], required=True)
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--method", choices=["prig", "gradient", "ig"], required=True)
    parser.add_argument("--target", choices=["ambiguous", "clear"], default="ambiguous")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    probe_dir = args.probe_dir or Path("probes") / args.probe_dataset
    output_dir = args.output_dir or default_output_dir(
        args.evaluation_dataset,
        args.method,
        args.probe_dataset,
        args.target,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [record for record in held_out_records(args.evaluation_dataset, args.data_root) if record.has_edit]
    probe_weights = load_probe_weights(probe_dir)

    model = load_model(args.model_name, device=args.device, torch_dtype=args.torch_dtype)
    ig_baseline = None
    if args.method == "ig":
        evaluation_records = load_records_for_dataset(args.evaluation_dataset, args.data_root)
        split = make_probe_split(
            evaluation_records,
            random_state=PROBE_SPLIT_RANDOM_STATES.get(args.evaluation_dataset, 1),
        ) if args.evaluation_dataset != "gold" else None
        baseline_prompts = [sample.prompt_text for sample in split.samples] if split is not None else [r.clear_prompt for r in records]
        ig_baseline = average_embedding_baseline(model, baseline_prompts)

    if args.method == "prig":
        for start_layer, end_layer in MID_LAYER_OPTIONS:
            results = {}
            for record in tqdm(records, desc=f"{start_layer}-{end_layer}"):
                prompt_text = record.ambiguous_prompt if args.target == "ambiguous" else record.clear_prompt
                target_sentence = record.ambiguous_sentence if args.target == "ambiguous" else record.original_sentence
                prompt_view, mask = mask_for_target_text(model, prompt_text, target_sentence)
                tokens, scores = residual_integrated_gradients(
                    model,
                    prompt_view.formatted_prompt,
                    probe_layer=end_layer,
                    attribution_layer=start_layer,
                    probe_weights=probe_weights[end_layer],
                    steps=args.steps,
                )
                prompt_tokens, prompt_scores = slice_prompt_tokens(tokens, np.asarray(scores), prompt_view)
                results[record.prompt_id] = serialize_example(
                    record,
                    prompt_tokens,
                    prompt_view.token_offsets,
                    mask,
                    prompt_scores.tolist(),
                    prompt_text,
                    target_sentence,
                )
            save_json(output_dir / f"prig_{start_layer}_{end_layer}_attribution_results.json", results)
        return

    results = {}
    for record in tqdm(records):
        prompt_text = record.ambiguous_prompt if args.target == "ambiguous" else record.clear_prompt
        target_sentence = record.ambiguous_sentence if args.target == "ambiguous" else record.original_sentence
        prompt_view, mask = mask_for_target_text(model, prompt_text, target_sentence)
        attributions = {}
        prompt_tokens = None
        for layer in PROBE_LAYERS:
            if args.method == "gradient":
                tokens, scores = gradient_x_input_attribution(
                    model,
                    prompt_view.formatted_prompt,
                    probe_layer=layer,
                    probe_weights=probe_weights[layer],
                )
            else:
                tokens, scores = integrated_gradients_attribution(
                    model,
                    prompt_view.formatted_prompt,
                    probe_layer=layer,
                    probe_weights=probe_weights[layer],
                    steps=args.steps,
                    baseline_vector=ig_baseline,
                )
            prompt_tokens, prompt_scores = slice_prompt_tokens(tokens, np.asarray(scores), prompt_view)
            attributions[str(layer)] = prompt_scores.tolist()
        results[record.prompt_id] = serialize_example(
            record,
            prompt_tokens,
            prompt_view.token_offsets,
            mask,
            attributions,
            prompt_text,
            target_sentence,
        )
    save_json(output_dir / "attribution_results.json", results)


if __name__ == "__main__":
    main()
