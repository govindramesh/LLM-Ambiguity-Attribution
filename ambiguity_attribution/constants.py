"""Project-wide constants."""

from __future__ import annotations

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_GENERATION_MODEL = "gpt-5.4"
DEFAULT_BASELINE_MODEL = "gpt-5.4"

DOMAINS = ("code", "math", "writing")
DOMAIN_PROMPT_NAMES = {
    "code": "coding",
    "math": "math",
    "writing": "writing",
}

MATH_CATEGORIES = (
    "algebra",
    "counting_and_probability",
    "number_theory",
    "geometry",
    "intermediate_algebra",
    "precalculus",
)

PROMPT_SAMPLING_SEED = 42
PROBE_SPLIT_RANDOM_STATES = {
    "code": 0,
    "math": 42,
    "writing": 10,
    "combined": 1,
}

PROBE_LAYERS = (12, 14, 16)
MID_LAYER_OPTIONS = (
    (8, 10),
    (8, 12),
    (10, 12),
    (12, 14),
    (13, 15),
    (12, 16),
    (16, 20),
    (18, 20),
)

DEFAULT_TEST_SIZE = 0.25
DEFAULT_SIGMA = 3.0
DEFAULT_IG_STEPS = 50
