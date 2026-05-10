"""OpenAI prompt templates and parsing helpers."""

from __future__ import annotations

import ast
import json

from .constants import DOMAIN_PROMPT_NAMES
from .gold import gold_examples_for_domain

GENERATION_SYSTEM_PROMPT = (
    "You are a research assistant working on a project investigating how large language models (LLMs) "
    "process ambiguity in prompts they are given. The prompts being evaluated are complex, therefore you "
    "are trying to see what area of the prompt causes the underlying task asked in the prompt to be unclear "
    "to the LLM. You are currently working on producing a dataset for the project by generating ambiguous "
    "versions of given prompts."
)

BASELINE_SYSTEM_PROMPT = ""

GENERATION_USER_TEMPLATE = """You are given a {domain} prompt consisting of multiple sentences.

Your task has two steps:

1. Identify the sentence in the prompt that is most critical to understanding what the {domain} question is asking. This should be the sentence whose removal or corruption would most severely impair comprehension of the task.

2. For this sentence, produce an **obscured version** that obfuscates the underlying meaning such that, if substituted back into the prompt, the overall meaning of the problem would become unclear. You must obscure the sentence enough that the {domain} problem is no longer understandable logically if substituted in. An LLM should not respond to the modified prompt because of how unclear it is.

Note: **DO NOT** only create ambiguity through scope ambiguous vocabulary/quantifiers (words like "maybe", "sometimes", "vaguely" or "random") or words synonymous with 'vague' (words like "mystery", "random", "something", "however", "any", "may", "specified") for the obfuscation. Instead, consider using techniques like adding random information, deleting key information, and replacing words. Just make sure the meaning is obfuscated enough that the whole prompt becomes unclear and unanswerable.

#####

Here are examples of good obfuscations:

{examples}


NOTES:
- The changed sentence still has a similar meaning as the original, but is more vague.
- The changed sentence would make the overall prompt unclear if substituted in.
- Every part of the changed sentence is more vague than the original, so the whole new sentence is ambiguous.
- The changed sentence should be unclear on its own.

#####
Output Instructions:

You must output **only** the following JSON-like object, with no extra text before or after:

{{
   "**original sentence**": "**obscured version**"
}}

Important constraints:
- Choose only 1 sentence to obfuscate that would most impair comprehension of the task if changed.
- The obscured version should not be longer than the original sentence.
- The key must be an exact sentence copied verbatim from the prompt.
- The value must be an changed version of only that sentence.
- Do not paraphrase the original text in the keys.
- Do not include explanations, bullet points, or formatting outside the schema.

Here is the {domain} prompt:
"{prompt}"
"""

GPT_SENTENCE_BASELINE_TEMPLATE = """You are given a prompt below. Your task is to identify which part of the prompt, if any, is ambiguous.
#####

Output Instructions:
- You must output **only** the sentence that you have identified as ambiguous, or the word "NONE" if there is no ambiguous sentence.
- If a sentence is chosen, it must be copied verbatim from the prompt.
- Do not include explanations, bullet points, or other formatting.

Here is the prompt to analyze:
"{prompt}"
"""


def render_generation_examples(domain: str) -> str:
    examples = []
    for i, example in enumerate(gold_examples_for_domain(domain)):
        examples.append(
            f"**Example {i}:**\n"
            f"Original Prompt: {example.clear_prompt}\n"
            f'Output: {json.dumps({example.original_sentence: example.ambiguous_sentence}, ensure_ascii=False)}'
        )
    return "\n\n".join(examples)


def render_generation_prompt(domain: str, prompt: str) -> str:
    return GENERATION_USER_TEMPLATE.format(
        domain=DOMAIN_PROMPT_NAMES[domain],
        examples=render_generation_examples(domain),
        prompt=prompt,
    )


def render_gpt_sentence_baseline_prompt(prompt: str) -> str:
    return GPT_SENTENCE_BASELINE_TEMPLATE.format(prompt=prompt)


def parse_rewrite_response(raw_text: str) -> tuple[str, str]:
    text = raw_text.strip()
    if "```json" in text:
        text = text.split("```json", maxsplit=1)[1].split("```", maxsplit=1)[0].strip()
    elif "```" in text:
        text = text.split("```", maxsplit=1)[1].split("```", maxsplit=1)[0].strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = ast.literal_eval(text)

    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError("Expected a single key/value rewrite mapping.")

    original, rewritten = next(iter(payload.items()))
    if not isinstance(original, str) or not isinstance(rewritten, str):
        raise ValueError("Rewrite mapping must contain string values.")
    return original, rewritten


def normalize_baseline_response(raw_text: str) -> str:
    text = raw_text.strip().strip("`").strip().strip('"').strip("'")
    if text.upper() == "NONE":
        return "NONE"
    return text


__all__ = [
    "BASELINE_SYSTEM_PROMPT",
    "GENERATION_SYSTEM_PROMPT",
    "GPT_SENTENCE_BASELINE_TEMPLATE",
    "normalize_baseline_response",
    "parse_rewrite_response",
    "render_generation_examples",
    "render_generation_prompt",
    "render_gpt_sentence_baseline_prompt",
]
