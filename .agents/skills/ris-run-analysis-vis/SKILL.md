---
name: ris-run-analysis-vis
description: Analyze completed training runs and generate evaluation figures for the RIS phase-shift optimization repository. Use when Codex needs to work with utils_vis.py or run artifacts such as metrics.jsonl, episode_metrics.csv, checkpoints, learned phase profiles, coverage maps, learning curves, or experiment summaries. Use it for plotting no-RIS versus optimized RIS coverage, learning curves, phase heatmaps, and run-level comparisons. Do not use for changing runtime setup, environment geometry, or SAC training logic.
---

# RIS Run Analysis and Visualization

Work from artifacts outward. Read completed runs before proposing conclusions.

## Primary files and directories

- `utils_vis.py`
- `runs/`
- `metrics.jsonl`
- `episode_metrics.csv`
- saved checkpoints and final agent files

## Analysis principles

1. Prefer reproducible summaries over ad hoc screenshots.
2. Read both baseline and episode-level metrics when available.
3. Keep figure titles and labels publication-friendly.
4. Preserve comparability across runs.
5. If a metric is missing, say so directly instead of inferring it.

## Standard workflow

### 1. Resolve the artifact source

Accept any of these inputs:

- a run directory
- `metrics.jsonl`
- `episode_metrics.csv`
- a saved phase profile or checkpoint-derived phase matrix

Normalize to the repository's expected artifact structure before plotting.

### 2. Determine the output type

Choose one primary goal:

- learning curve
- baseline comparison
- coverage comparison
- final phase heatmap
- multi-run comparison summary

Do not combine everything into one overloaded figure unless explicitly requested.

### 3. Keep baseline context visible

Whenever possible, include the baseline reference used during training or evaluation. If the baseline is unavailable, label the figure clearly rather than pretending the reference exists.

### 4. Save clean outputs

Prefer explicit output paths and stable filenames. Keep generated figures suitable for README files, reports, or papers.

## Good tasks for this skill

- plot training reward vs reflector baseline
- compare no-RIS and DRL coverage maps
- visualize final 100x100 phase profile
- summarize the best run from several run folders
- sanity-check whether logged metrics support the claimed improvement

## Anti-patterns

Do not:

- alter training logic here
- silently smooth away problematic behavior
- mix incompatible runs without labeling their config differences
- infer unavailable baselines from unrelated runs

## Validation checklist

After analysis work, verify:

- the figure source files are identified
- axis labels match the metric units
- smoothing settings are disclosed where relevant
- baseline source is stated
- saved figure paths are explicit

## Output expectations

When you use this skill, provide:

1. which artifact(s) were analyzed
2. which figure(s) were created
3. which baseline or reference was used
4. any caveat about missing or incomplete metrics
