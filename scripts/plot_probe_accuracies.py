from __future__ import annotations

import argparse
from pathlib import Path

from ambiguity_attribution.probe import load_probe_accuracies, plot_probe_accuracies


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot probe accuracies across datasets.")
    parser.add_argument(
        "--probe-dirs",
        nargs="+",
        type=Path,
        default=[Path("probes/code"), Path("probes/math"), Path("probes/writing"), Path("probes/combined")],
    )
    parser.add_argument("--output-path", type=Path, default=Path("figures/probe_plot.png"))
    args = parser.parse_args()

    accuracy_by_dataset = {probe_dir.name: load_probe_accuracies(probe_dir) for probe_dir in args.probe_dirs}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_probe_accuracies(accuracy_by_dataset, args.output_path)


if __name__ == "__main__":
    main()
