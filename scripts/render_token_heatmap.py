from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ambiguity_attribution.metrics import gaussian_smooth
from ambiguity_attribution.visualization import (
    render_full_html_document,
    visualize_token_attributions,
)


def load_example(results_path: Path, key: str | None, index: int | None) -> tuple[str, dict]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    items = list(payload.items())
    if key is not None:
        return key, payload[key]
    if index is None:
        index = 0
    return items[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a token-attribution heatmap as HTML.")
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--key")
    parser.add_argument("--index", type=int)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--max-alpha", type=float, default=1.0)
    parser.add_argument("--heatmap-color", default="255, 0, 0")
    parser.add_argument("--title")
    args = parser.parse_args()

    example_key, example = load_example(args.results_path, args.key, args.index)
    if isinstance(example["attributions"], dict):
        if args.layer is None:
            raise ValueError("Please provide --layer when the results file contains per-layer attributions.")
        attributions = np.asarray(example["attributions"][str(args.layer)])
        title = args.title or f"{example_key} layer {args.layer}"
    else:
        attributions = np.asarray(example["attributions"])
        title = args.title or example_key

    if args.sigma:
        attributions = gaussian_smooth(attributions, sigma=args.sigma)

    snippet = visualize_token_attributions(
        example["tokens"],
        attributions,
        example["mask"],
        max_alpha=args.max_alpha,
        heatmap_color=args.heatmap_color,
    )
    html_document = render_full_html_document(snippet, title=title)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(html_document, encoding="utf-8")


if __name__ == "__main__":
    main()
