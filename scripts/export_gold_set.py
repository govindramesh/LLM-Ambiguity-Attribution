from __future__ import annotations

import argparse
import json
from pathlib import Path

from ambiguity_attribution.data import export_gold_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the gold set files used in the paper.")
    parser.add_argument("--output-root", type=Path, default=Path("."))
    args = parser.parse_args()

    clear_prompts, rewrites = export_gold_payload()
    gold_dir = args.output_root / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    (args.output_root / "gold_problems.json").write_text(
        json.dumps(clear_prompts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for example_id, payload in rewrites.items():
        (gold_dir / f"{example_id}.txt").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
