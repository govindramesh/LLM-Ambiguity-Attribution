"""Dataset loading and split helpers."""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from .constants import (
    DEFAULT_TEST_SIZE,
    DOMAINS,
    MATH_CATEGORIES,
    PROMPT_SAMPLING_SEED,
)
from .gold import GOLD_EXAMPLES
from .prompts import parse_rewrite_response


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    source_id: int
    domain: str
    clear_prompt: str
    original_sentence: str | None = None
    ambiguous_sentence: str | None = None

    @property
    def has_edit(self) -> bool:
        return self.original_sentence is not None and self.ambiguous_sentence is not None

    @property
    def ambiguous_prompt(self) -> str | None:
        if not self.has_edit:
            return None
        return self.clear_prompt.replace(self.original_sentence, self.ambiguous_sentence, 1)


@dataclass(frozen=True)
class ProbeSample:
    prompt_id: str
    domain: str
    prompt_text: str
    label: int


@dataclass(frozen=True)
class ProbeSplit:
    samples: list[ProbeSample]
    train_indices: list[int]
    test_indices: list[int]
    train_prompt_ids: list[str]
    test_prompt_ids: list[str]


def _count_sentences_with_period_split(prompt: str) -> int:
    return len(prompt.split("."))


def _load_rewrite_map(directory: Path) -> dict[int, tuple[str, str]]:
    rewrites: dict[int, tuple[str, str]] = {}
    if not directory.exists():
        return rewrites

    for path in sorted(directory.glob("*.txt")):
        try:
            source_id = int(path.stem)
        except ValueError:
            continue
        raw_text = path.read_text(encoding="utf-8")
        try:
            rewrites[source_id] = parse_rewrite_response(raw_text)
        except Exception:
            continue
    return rewrites


def _load_code_records(data_root: Path) -> list[PromptRecord]:
    dataset = load_dataset("newfacade/LeetCodeDataset", trust_remote_code=True)
    hard_problems = [example for example in dataset["train"] if example["difficulty"] == "Hard"]
    rewrites = _load_rewrite_map(data_root / "leetcode" / "unclear_portion")

    records = []
    for example in hard_problems:
        prompt = example["problem_description"].split("\n\xa0")[0].replace("\xa0", " ")
        if _count_sentences_with_period_split(prompt) < 5:
            continue
        source_id = int(example["question_id"])
        original_sentence, ambiguous_sentence = rewrites.get(source_id, (None, None))
        if original_sentence is not None and original_sentence not in prompt:
            original_sentence = None
            ambiguous_sentence = None
        records.append(
            PromptRecord(
                prompt_id=f"code:{source_id}",
                source_id=source_id,
                domain="code",
                clear_prompt=prompt,
                original_sentence=original_sentence,
                ambiguous_sentence=ambiguous_sentence,
            )
        )
    return records


def _load_math_records(data_root: Path) -> list[PromptRecord]:
    rewrites = _load_rewrite_map(data_root / "math")
    prompts = []
    for category in MATH_CATEGORIES:
        dataset = load_dataset("EleutherAI/hendrycks_math", category)
        prompts.extend(item["problem"] for item in dataset["train"])
        prompts.extend(item["problem"] for item in dataset["test"])

    records = []
    for source_id, prompt in enumerate(prompts):
        if len(prompt.split(". ")) < 4 or "\n" in prompt:
            continue
        original_sentence, ambiguous_sentence = rewrites.get(source_id, (None, None))
        if original_sentence is not None and original_sentence not in prompt:
            original_sentence = None
            ambiguous_sentence = None
        records.append(
            PromptRecord(
                prompt_id=f"math:{source_id}",
                source_id=source_id,
                domain="math",
                clear_prompt=prompt,
                original_sentence=original_sentence,
                ambiguous_sentence=ambiguous_sentence,
            )
        )
    return records


def _load_writing_records(data_root: Path) -> list[PromptRecord]:
    dataset = load_dataset("PromptTensor/prompttensor-promptbank")
    rewrites = _load_rewrite_map(data_root / "writing")

    prompts = []
    recent_stems: deque[str] = deque([""] * 5)
    for item in dataset["train"]:
        if item["intent"] not in {"planning", "generation"}:
            continue
        stem = item["prompt_text"][:15]
        if stem in recent_stems:
            continue
        prompts.append(item["prompt_text"])
        recent_stems.append(stem)
        recent_stems.popleft()

    records = []
    for source_id, prompt in enumerate(prompts):
        if len(prompt.split(". ")) < 4:
            continue
        original_sentence, ambiguous_sentence = rewrites.get(source_id, (None, None))
        if original_sentence is not None and original_sentence not in prompt:
            original_sentence = None
            ambiguous_sentence = None
        records.append(
            PromptRecord(
                prompt_id=f"writing:{source_id}",
                source_id=source_id,
                domain="writing",
                clear_prompt=prompt,
                original_sentence=original_sentence,
                ambiguous_sentence=ambiguous_sentence,
            )
        )
    return records


def load_domain_records(domain: str, data_root: str | Path = ".") -> list[PromptRecord]:
    root = Path(data_root)
    if domain == "code":
        return _load_code_records(root)
    if domain == "math":
        return _load_math_records(root)
    if domain == "writing":
        return _load_writing_records(root)
    raise ValueError(f"Unsupported domain: {domain}")


def load_combined_records(domains: tuple[str, ...] = DOMAINS, data_root: str | Path = ".") -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for domain in domains:
        records.extend(load_domain_records(domain, data_root=data_root))
    return records


def load_gold_records() -> list[PromptRecord]:
    return [
        PromptRecord(
            prompt_id=f"gold:{example.example_id}",
            source_id=example.example_id,
            domain=example.domain,
            clear_prompt=example.clear_prompt,
            original_sentence=example.original_sentence,
            ambiguous_sentence=example.ambiguous_sentence,
        )
        for example in GOLD_EXAMPLES
    ]


def sample_probe_dataset(
    records: list[PromptRecord],
    sampling_seed: int = PROMPT_SAMPLING_SEED,
) -> list[ProbeSample]:
    rng = random.Random(sampling_seed)
    samples: list[ProbeSample] = []
    for record in records:
        use_ambiguous = record.has_edit and rng.choice([False, True])
        prompt_text = record.ambiguous_prompt if use_ambiguous else record.clear_prompt
        samples.append(
            ProbeSample(
                prompt_id=record.prompt_id,
                domain=record.domain,
                prompt_text=prompt_text or record.clear_prompt,
                label=int(use_ambiguous),
            )
        )
    return samples


def make_probe_split(
    records: list[PromptRecord],
    random_state: int,
    test_size: float = DEFAULT_TEST_SIZE,
    sampling_seed: int = PROMPT_SAMPLING_SEED,
) -> ProbeSplit:
    samples = sample_probe_dataset(records, sampling_seed=sampling_seed)
    labels = [sample.label for sample in samples]
    indices = list(range(len(samples)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    return ProbeSplit(
        samples=samples,
        train_indices=list(train_indices),
        test_indices=list(test_indices),
        train_prompt_ids=[samples[i].prompt_id for i in train_indices],
        test_prompt_ids=[samples[i].prompt_id for i in test_indices],
    )


def export_gold_payload() -> tuple[list[str], dict[int, dict[str, str]]]:
    clear_prompts = [example.clear_prompt for example in GOLD_EXAMPLES]
    rewrites = {
        example.example_id: {example.original_sentence: example.ambiguous_sentence}
        for example in GOLD_EXAMPLES
    }
    return clear_prompts, rewrites


def save_probe_split(split: ProbeSplit, output_path: str | Path) -> None:
    payload = {
        "samples": [
            {
                "prompt_id": sample.prompt_id,
                "domain": sample.domain,
                "label": sample.label,
                "prompt_text": sample.prompt_text,
            }
            for sample in split.samples
        ],
        "train_indices": split.train_indices,
        "test_indices": split.test_indices,
        "train_prompt_ids": split.train_prompt_ids,
        "test_prompt_ids": split.test_prompt_ids,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
