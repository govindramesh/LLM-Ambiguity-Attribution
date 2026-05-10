"""Probe extraction and training utilities."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from transformer_lens import HookedTransformer


def extract_mean_pooled_residual_activations(
    model: HookedTransformer,
    formatted_prompts: list[str],
) -> dict[int, np.ndarray]:
    activations = {layer: [] for layer in range(model.cfg.n_layers)}
    hook_points = [f"blocks.{layer}.hook_resid_post" for layer in range(model.cfg.n_layers)]

    for prompt in formatted_prompts:
        tokens = model.to_tokens(prompt, prepend_bos=False).to(model.cfg.device)
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in hook_points)
        for layer in range(model.cfg.n_layers):
            pooled = cache[hook_points[layer]][0, :, :].mean(dim=0).detach().cpu().numpy()
            activations[layer].append(pooled)

    return {layer: np.asarray(values) for layer, values in activations.items()}


def train_layerwise_probes(
    activations: dict[int, np.ndarray],
    labels: list[int] | np.ndarray,
    train_indices: list[int],
    test_indices: list[int],
) -> tuple[dict[int, float], np.ndarray]:
    y = np.asarray(labels)
    accuracies: dict[int, float] = {}
    weights = np.zeros((len(activations), activations[0].shape[1]), dtype=np.float32)

    for layer in range(len(activations)):
        X = activations[layer]
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        probe = LogisticRegression(max_iter=5000, fit_intercept=False)
        probe.fit(X_train, y_train)

        accuracies[layer] = float(probe.score(X_test, y_test))
        weights[layer] = probe.coef_[0].astype(np.float32)

    return accuracies, weights


def save_probe_artifacts(
    output_dir: str | Path,
    accuracies: dict[int, float],
    weights: np.ndarray,
    metadata: dict,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "probe_weights.npy", weights)
    (output_path / "probe_accuracies.json").write_text(
        json.dumps({str(layer): value for layer, value in accuracies.items()}, indent=2),
        encoding="utf-8",
    )
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_probe_weights(probe_dir: str | Path) -> np.ndarray:
    return np.load(Path(probe_dir) / "probe_weights.npy")


def load_probe_accuracies(probe_dir: str | Path) -> dict[int, float]:
    payload = json.loads((Path(probe_dir) / "probe_accuracies.json").read_text(encoding="utf-8"))
    return {int(layer): float(value) for layer, value in payload.items()}


def plot_probe_accuracies(
    accuracy_by_dataset: dict[str, dict[int, float]],
    output_path: str | Path,
) -> None:
    plt.figure(figsize=(6, 4))
    for label, values in accuracy_by_dataset.items():
        layers = sorted(values)
        plt.plot(layers, [values[layer] for layer in layers], marker="o", label=label)
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
