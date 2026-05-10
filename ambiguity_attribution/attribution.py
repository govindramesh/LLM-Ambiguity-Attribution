"""Attribution methods used in the paper."""

from __future__ import annotations

import numpy as np
import torch
from transformer_lens import HookedTransformer


def _probe_weight_tensor(probe_weights, device: str) -> torch.Tensor:
    if isinstance(probe_weights, torch.Tensor):
        return probe_weights.to(device=device, dtype=torch.float16)
    return torch.tensor(probe_weights, device=device, dtype=torch.float16)


def _probe_logit(residual: torch.Tensor, probe_weights: torch.Tensor) -> torch.Tensor:
    return torch.dot(residual.mean(dim=0), probe_weights)


def gradient_x_input_attribution(
    model: HookedTransformer,
    formatted_prompt: str,
    probe_layer: int,
    probe_weights,
) -> tuple[list[str], np.ndarray]:
    model.zero_grad()
    device = model.cfg.device
    tokens = model.to_tokens(formatted_prompt, prepend_bos=False).to(device)
    token_embeds = model.embed(tokens).to(torch.float16)
    token_embeds.requires_grad_(True)
    token_embeds.retain_grad()

    hook_name = f"blocks.{probe_layer}.hook_resid_post"
    cache: dict[str, torch.Tensor] = {}

    def cache_hook(activation, hook):
        cache[hook.name] = activation
        return activation

    with model.hooks(fwd_hooks=[(hook_name, cache_hook)]):
        _ = model(token_embeds, start_at_layer=0)

    probe_weight_tensor = _probe_weight_tensor(probe_weights, device)
    score = _probe_logit(cache[hook_name][0, :, :], probe_weight_tensor)
    score.backward()

    attribution = torch.sum(token_embeds.grad[0] * token_embeds[0], dim=1).detach().cpu().numpy()
    return model.to_str_tokens(tokens), attribution


def integrated_gradients_attribution(
    model: HookedTransformer,
    formatted_prompt: str,
    probe_layer: int,
    probe_weights,
    steps: int = 50,
    baseline_vector: torch.Tensor | None = None,
) -> tuple[list[str], np.ndarray]:
    model.zero_grad()
    device = model.cfg.device
    tokens = model.to_tokens(formatted_prompt, prepend_bos=False).to(device)
    token_embeds = model.embed(tokens).to(torch.float16).detach()

    if baseline_vector is None:
        pad_token_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
        baseline = model.embed(pad_token_id * torch.ones_like(tokens)).to(torch.float16).detach()
    else:
        baseline = baseline_vector.to(device=device, dtype=torch.float16).unsqueeze(0).expand_as(token_embeds)

    total_grads = torch.zeros_like(token_embeds)
    alphas = torch.linspace(1 / steps, 1, steps, device=device)
    hook_name = f"blocks.{probe_layer}.hook_resid_post"
    probe_weight_tensor = _probe_weight_tensor(probe_weights, device)

    for alpha in alphas:
        model.zero_grad()
        scaled = baseline + alpha * (token_embeds - baseline)
        scaled.requires_grad_(True)
        scaled.retain_grad()

        cache: dict[str, torch.Tensor] = {}

        def cache_hook(activation, hook):
            cache[hook.name] = activation
            return activation

        with model.hooks(fwd_hooks=[(hook_name, cache_hook)]):
            _ = model(scaled, start_at_layer=0)

        score = _probe_logit(cache[hook_name][0, :, :], probe_weight_tensor)
        score.backward()
        total_grads += scaled.grad

    ig = (token_embeds - baseline) * (total_grads / steps)
    attribution = ig[0].sum(dim=1).detach().cpu().numpy()
    return model.to_str_tokens(tokens), attribution


def residual_integrated_gradients(
    model: HookedTransformer,
    formatted_prompt: str,
    probe_layer: int,
    attribution_layer: int,
    probe_weights,
    steps: int = 50,
) -> tuple[list[str], np.ndarray]:
    model.zero_grad()
    device = model.cfg.device
    tokens = model.to_tokens(formatted_prompt, prepend_bos=False).to(device)

    base_hook = f"blocks.{attribution_layer}.hook_resid_post"
    probe_hook = f"blocks.{probe_layer}.hook_resid_post"
    initial_cache: dict[str, torch.Tensor] = {}

    def cache_base(activation, hook):
        initial_cache[hook.name] = activation.detach()
        return activation

    with model.hooks(fwd_hooks=[(base_hook, cache_base)]):
        _ = model(tokens)

    residual_start = initial_cache[base_hook].detach()
    baseline = torch.zeros_like(residual_start)
    total_grads = torch.zeros_like(residual_start)
    alphas = torch.linspace(1 / steps, 1, steps, device=device)
    probe_weight_tensor = _probe_weight_tensor(probe_weights, device)

    for alpha in alphas:
        model.zero_grad()
        scaled = baseline + alpha * (residual_start - baseline)
        scaled = scaled.detach().requires_grad_(True)
        scaled.retain_grad()

        probe_cache: dict[str, torch.Tensor] = {}

        def cache_probe(activation, hook):
            probe_cache[hook.name] = activation
            return activation

        with model.hooks(fwd_hooks=[(probe_hook, cache_probe)]):
            _ = model(scaled, start_at_layer=attribution_layer)

        score = _probe_logit(probe_cache[probe_hook][0], probe_weight_tensor)
        score.backward()
        total_grads += scaled.grad.detach()

    ig = (residual_start - baseline) * (total_grads / steps)
    attribution = ig[0].sum(dim=1).detach().cpu().numpy()
    return model.to_str_tokens(tokens), attribution


def average_embedding_baseline(
    model: HookedTransformer,
    prompts: list[str],
) -> torch.Tensor:
    baseline = torch.zeros(model.cfg.d_model, device=model.cfg.device, dtype=torch.float16)
    total_tokens = 0
    for prompt in prompts:
        tokens = model.to_tokens(prompt, prepend_bos=False).to(model.cfg.device)
        embeds = model.embed(tokens)[0]
        baseline += embeds.sum(dim=0).detach()
        total_tokens += embeds.shape[0]
    if total_tokens == 0:
        raise ValueError("Cannot compute an average embedding baseline from an empty prompt list.")
    return baseline / total_tokens
