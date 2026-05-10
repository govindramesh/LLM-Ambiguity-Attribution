"""Model loading and prompt-formatting helpers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import torch
from huggingface_hub import login
from transformer_lens import HookedTransformer

from .constants import DEFAULT_MODEL_NAME

_DATE_BLOCK_RE = re.compile(
    r"Cutting Knowledge Date:.*?\nToday Date:.*?\n\n",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class PromptView:
    formatted_prompt: str
    prompt_text: str
    prompt_start: int
    prompt_end: int
    token_indices: list[int]
    token_offsets: list[tuple[int, int]]


def login_to_hugging_face_from_env() -> None:
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)


def _parse_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return mapping[dtype]


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cuda",
    torch_dtype: str | torch.dtype = "float16",
) -> HookedTransformer:
    login_to_hugging_face_from_env()
    return HookedTransformer.from_pretrained(
        model_name,
        device=device,
        torch_dtype=_parse_torch_dtype(torch_dtype),
    )


def format_prompt(model: HookedTransformer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    formatted = model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return _DATE_BLOCK_RE.sub("", formatted, count=1)


def _prompt_span(formatted_prompt: str, prompt: str) -> tuple[int, int]:
    prompt_start = formatted_prompt.find(prompt)
    if prompt_start < 0:
        raise ValueError("Prompt text was not found in the formatted chat prompt.")
    return prompt_start, prompt_start + len(prompt)


def build_prompt_view(model: HookedTransformer, prompt: str) -> PromptView:
    formatted_prompt = format_prompt(model, prompt)
    prompt_start, prompt_end = _prompt_span(formatted_prompt, prompt)
    tokenized = model.tokenizer(
        formatted_prompt,
        return_offsets_mapping=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    offsets = tokenized["offset_mapping"][0].tolist()
    token_indices = []
    relative_offsets = []
    for index, (start, end) in enumerate(offsets):
        if end <= prompt_start or start >= prompt_end:
            continue
        token_indices.append(index)
        relative_offsets.append((start - prompt_start, end - prompt_start))
    return PromptView(
        formatted_prompt=formatted_prompt,
        prompt_text=prompt,
        prompt_start=prompt_start,
        prompt_end=prompt_end,
        token_indices=token_indices,
        token_offsets=relative_offsets,
    )


def mask_for_target_text(
    model: HookedTransformer,
    prompt: str,
    target_text: str,
) -> tuple[PromptView, torch.Tensor]:
    view = build_prompt_view(model, prompt)
    target_start = prompt.find(target_text)
    if target_start < 0:
        raise ValueError("Target text was not found in the prompt.")
    target_end = target_start + len(target_text)
    mask = torch.zeros(len(view.token_indices), dtype=torch.long)
    for i, (start, end) in enumerate(view.token_offsets):
        if end <= target_start or start >= target_end:
            continue
        mask[i] = 1
    return view, mask


def slice_prompt_tokens(
    tokens: list[str],
    scores,
    prompt_view: PromptView,
):
    prompt_tokens = [tokens[index] for index in prompt_view.token_indices]
    if scores is None:
        return prompt_tokens, None
    return prompt_tokens, scores[prompt_view.token_indices]
