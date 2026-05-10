# LLM Ambiguity Attribution

This repository contains the cleaned research code for the ambiguity-attribution experiments in the paper. The original implementation lived in notebooks; the runnable code now lives in small Python modules and scripts. The notebooks are still present as reference artifacts for now.

## Layout

- `ambiguity_attribution/`: shared code for data loading, prompts, probe training, attribution, evaluation, and visualization.
- `scripts/`: command-line entrypoints for the paper workflows.
- `*.ipynb`: original reference notebooks.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Set credentials through environment variables instead of editing source files:

```bash
export OPENAI_API_KEY=...
export HF_TOKEN=...
```

`HF_TOKEN` is needed for loading `meta-llama/Llama-3.1-8B-Instruct`.

## Expected Data Layout

Synthetic rewrites are expected in the same per-domain layout used by the notebooks:

- `leetcode/unclear_portion/<question_id>.txt`
- `math/<id>.txt`
- `writing/<id>.txt`

Each file should contain a single JSON mapping from the original sentence to its ambiguous rewrite.

The gold set can be exported from the embedded paper examples with:

```bash
python scripts/export_gold_set.py
```

This writes:

- `gold_problems.json`
- `gold/<id>.txt`

## Main Scripts

Generate synthetic rewrites for one domain:

```bash
python scripts/generate_synthetic_rewrites.py --domain code
python scripts/generate_synthetic_rewrites.py --domain math
python scripts/generate_synthetic_rewrites.py --domain writing
```

Train probes:

```bash
python scripts/train_probe.py --dataset code
python scripts/train_probe.py --dataset math
python scripts/train_probe.py --dataset writing
python scripts/train_probe.py --dataset combined
```

Plot the probe-accuracy figure:

```bash
python scripts/plot_probe_accuracies.py
```

Run attributions on a held-out split:

```bash
python scripts/run_attribution.py --evaluation-dataset code --probe-dataset code --method mid_layer
python scripts/run_attribution.py --evaluation-dataset code --probe-dataset code --method gradient
python scripts/run_attribution.py --evaluation-dataset code --probe-dataset code --method ig
```

Score attribution outputs:

```bash
python scripts/score_attributions.py --results-dir code_attributions/code/mid_layer --mode mid-layer
python scripts/score_attributions.py --results-dir code_attributions/code/gradient --mode layers
python scripts/score_attributions.py --results-dir code_attributions/code/ig --mode layers
```

Run the GPT-5.4 sentence-selection baseline on the held-out combined set:

```bash
python scripts/run_gpt_sentence_baseline.py --dataset combined
```

Score sentence-level PRIG selection from held-out combined attributions:

```bash
python scripts/score_sentence_selection.py \
  --ambiguous-results combined_attributions/combined/mid_layer/mid_layer_12_14_attribution_results.json \
  --clear-results combined_attributions_clear/combined/mid_layer/mid_layer_12_14_attribution_results.json
```

Render a paper-style token heatmap as standalone HTML:

```bash
python scripts/render_token_heatmap.py \
  --results-path gold_attributions/combined/mid_layer/mid_layer_12_14_attribution_results.json \
  --index 2 \
  --output-path export/example_heatmap.html \
  --sigma 1
```

## Reproducing the Paper Workflows

1. Data construction
   - Add or generate the synthetic rewrite files for `code`, `math`, and `writing`.
   - Export the embedded gold examples if you want the local `gold/` files.

2. Probe training
   - Run `scripts/train_probe.py` for `code`, `math`, `writing`, and `combined`.
   - Run `scripts/plot_probe_accuracies.py` to recreate the probe-accuracy figure.

3. Attribution runs
   - For each evaluation dataset, run `scripts/run_attribution.py` with:
     - `--method mid_layer` for the paper’s main PRIG method.
     - `--method gradient` for Grad×Input.
     - `--method ig` for embedding-space IG.
   - Use `--probe-dataset` to swap between in-domain, combined, and cross-domain probes.
   - Use `--target clear` to run the clear-prompt control evaluation.

4. Attribution scoring
   - Run `scripts/score_attributions.py` on each output directory to write the CSV summaries used for layer and interval selection.

5. GPT baseline
   - Run `scripts/run_gpt_sentence_baseline.py --dataset combined` for the sentence-selection baseline reported in the paper.
   - Run `scripts/score_sentence_selection.py` on the combined clear/ambiguous PRIG outputs to reproduce the thresholded sentence-level evaluation for the attribution method.

## Notes

- The cleaned scripts keep the notebooks’ dataset filters and directory conventions unless the paper specifies otherwise.
- The HTML heatmap helper used for qualitative examples is available as `ambiguity_attribution.visualize_token_attributions_blue`, matching the notebook function name for traceability.
