from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from ambiguity_attribution.constants import DEFAULT_GENERATION_MODEL
from ambiguity_attribution.data import load_domain_records
from ambiguity_attribution.prompts import (
    GENERATION_SYSTEM_PROMPT,
    parse_rewrite_response,
    render_generation_prompt,
)


def default_output_dir(domain: str, output_root: Path) -> Path:
    if domain == "code":
        return output_root / "leetcode" / "unclear_portion"
    return output_root / domain


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ambiguity rewrites.")
    parser.add_argument("--domain", choices=["code", "math", "writing"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-completion-tokens", type=int, default=500)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = default_output_dir(args.domain, args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_domain_records(args.domain, data_root=args.data_root)
    if args.limit is not None:
        records = records[: args.limit]

    client = OpenAI()
    failures = []

    for record in tqdm(records):
        output_path = output_dir / f"{record.source_id}.txt"
        if output_path.exists() and not args.overwrite:
            continue

        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": render_generation_prompt(record.domain, record.clear_prompt)},
            ],
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
        )
        raw_text = response.choices[0].message.content.strip()
        try:
            original, rewritten = parse_rewrite_response(raw_text)
        except Exception:
            failures.append(record.prompt_id)
            continue

        output_path.write_text(
            json.dumps({original: rewritten}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if failures:
        print("Failed to parse responses for:")
        for prompt_id in failures:
            print(prompt_id)


if __name__ == "__main__":
    main()
