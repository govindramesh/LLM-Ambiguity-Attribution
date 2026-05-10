"""HTML visualization helpers for token-level attributions."""

from __future__ import annotations

import html

import numpy as np


def visualize_token_attributions_blue(
    tokens,
    attributions,
    mask,
    max_alpha: float = 1.0,
    heatmap_color: str = "255, 0, 0",
) -> str:
    """
    Render the notebook-style token heatmap used for qualitative examples.

    This follows the `visualize_token_attributions_blue` notebook function that
    generated the paper examples: token background opacity is based on the
    absolute attribution magnitude, and ground-truth mask tokens are
    underlined.

    The original notebook function name is preserved for traceability, even
    though the default heatmap color is not blue.
    """

    attribution_array = np.asarray(attributions, dtype=float)
    mask_array = np.asarray(mask)

    max_val = np.max(np.abs(attribution_array)) + 1e-8
    normalized = np.abs(attribution_array) / max_val

    html_tokens = []
    for token, attr, in_mask in zip(tokens, normalized, mask_array):
        token_escaped = html.escape(token)
        alpha = float(attr) * max_alpha
        background = f"rgba({heatmap_color}, {alpha})"
        style = f"background-color: {background};"
        if in_mask:
            style += " text-decoration: underline; text-decoration-thickness: 2px; font-weight:bold;"
        html_tokens.append(f'<span style="{style}">{token_escaped}</span>')

    html_str = "".join(html_tokens)
    return f"""
<div style="
    font-family: monospace;
    white-space: pre-wrap;
    line-height: 1.5;
    max-width: 80ch;
">
{html_str}
</div>
"""


def render_full_html_document(
    body_html: str,
    title: str = "Token Attribution Heatmap",
) -> str:
    """Wrap a visualization snippet in a simple standalone HTML document."""

    title_escaped = html.escape(title)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title_escaped}</title>
</head>
<body style="margin: 24px; font-family: sans-serif;">
  <h1 style="font-size: 18px; margin-bottom: 16px;">{title_escaped}</h1>
  {body_html}
</body>
</html>
"""
