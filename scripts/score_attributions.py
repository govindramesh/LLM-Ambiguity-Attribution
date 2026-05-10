from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ambiguity_attribution.constants import MID_LAYER_OPTIONS, PROBE_LAYERS
from ambiguity_attribution.metrics import gaussian_smooth, span_auprg, span_auroc


def score_layer_results(results: dict, output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "metric", "sigma", "mean", "std"])
        for sigma in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
            for layer in PROBE_LAYERS:
                aurocs = []
                auprgs = []
                for item in results.values():
                    mask = np.asarray(item["mask"])
                    attribution = gaussian_smooth(np.asarray(item["attributions"][str(layer)]), sigma=sigma)
                    aurocs.append(span_auroc(attribution, mask))
                    auprgs.append(span_auprg(attribution, mask))
                writer.writerow([layer, "auroc", sigma, float(np.mean(aurocs)), float(np.std(aurocs))])
                writer.writerow([layer, "auprg", sigma, float(np.mean(auprgs)), float(np.std(auprgs))])


def score_prig_results(results_dir: Path, output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_layer", "end_layer", "metric", "sigma", "mean", "std"])
        for start_layer, end_layer in MID_LAYER_OPTIONS:
            results_path = results_dir / f"prig_{start_layer}_{end_layer}_attribution_results.json"
            if not results_path.exists():
                continue
            results = json.loads(results_path.read_text(encoding="utf-8"))
            for sigma in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
                aurocs = []
                auprgs = []
                for item in results.values():
                    mask = np.asarray(item["mask"])
                    attribution = gaussian_smooth(np.asarray(item["attributions"]), sigma=sigma)
                    aurocs.append(span_auroc(attribution, mask))
                    auprgs.append(span_auprg(attribution, mask))
                writer.writerow([start_layer, end_layer, "auroc", sigma, float(np.mean(aurocs)), float(np.std(aurocs))])
                writer.writerow([start_layer, end_layer, "auprg", sigma, float(np.mean(auprgs)), float(np.std(auprgs))])


def main() -> None:
    parser = argparse.ArgumentParser(description="Score attribution outputs and write CSV summaries.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["layers", "prig"], required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("results.csv"))
    args = parser.parse_args()

    output_csv = args.output_csv
    if not output_csv.is_absolute():
        output_csv = args.results_dir / output_csv

    if args.mode == "layers":
        results = json.loads((args.results_dir / "attribution_results.json").read_text(encoding="utf-8"))
        score_layer_results(results, output_csv)
    else:
        score_prig_results(args.results_dir, output_csv)


if __name__ == "__main__":
    main()
