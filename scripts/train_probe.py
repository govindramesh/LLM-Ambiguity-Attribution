from __future__ import annotations

import argparse
from pathlib import Path

from ambiguity_attribution.constants import PROBE_SPLIT_RANDOM_STATES
from ambiguity_attribution.data import (
    load_combined_records,
    load_domain_records,
    make_probe_split,
    save_probe_split,
)
from ambiguity_attribution.modeling import format_prompt, load_model
from ambiguity_attribution.probe import (
    extract_mean_pooled_residual_activations,
    save_probe_artifacts,
    train_layerwise_probes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the layerwise ambiguity probes.")
    parser.add_argument("--dataset", choices=["code", "math", "writing", "combined"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--random-state", type=int)
    args = parser.parse_args()

    output_dir = args.output_dir or Path("probes") / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "combined":
        records = load_combined_records(data_root=args.data_root)
    else:
        records = load_domain_records(args.dataset, data_root=args.data_root)

    random_state = args.random_state
    if random_state is None:
        random_state = PROBE_SPLIT_RANDOM_STATES[args.dataset]

    split = make_probe_split(records, random_state=random_state)
    save_probe_split(split, output_dir / "split.json")

    model = load_model(args.model_name, device=args.device, torch_dtype=args.torch_dtype)
    formatted_prompts = [format_prompt(model, sample.prompt_text) for sample in split.samples]
    activations = extract_mean_pooled_residual_activations(model, formatted_prompts)
    labels = [sample.label for sample in split.samples]
    accuracies, weights = train_layerwise_probes(
        activations,
        labels,
        split.train_indices,
        split.test_indices,
    )

    metadata = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "random_state": random_state,
        "num_records": len(records),
        "num_samples": len(split.samples),
    }
    save_probe_artifacts(output_dir, accuracies, weights, metadata)


if __name__ == "__main__":
    main()
