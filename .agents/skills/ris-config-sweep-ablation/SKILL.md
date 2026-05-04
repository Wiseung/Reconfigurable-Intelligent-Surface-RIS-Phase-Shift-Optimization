---
name: ris-config-sweep-ablation
description: Create and manage controlled experiment configs and ablation plans for the RIS phase-shift optimization repository. Use when Codex needs to add or modify YAML experiment configs, compare hyperparameter settings, build small ablation studies, generate sweep-ready config variants, or organize experiments around grouped action size, replay behavior, reward shaping, sampling budgets, evaluation cadence, and runtime settings. Do not use for deep code refactors when only one experiment should be changed, or for visualization-only tasks.
---

# RIS Config Sweep and Ablation

Treat experiment configs as the primary interface for controlled comparisons.

## Primary files

- `config.yaml`
- `configs/*.yaml`
- `train_loop.py`
- run-directory outputs used for comparison

## Principles

1. Change one main variable per ablation unless the user explicitly wants a compound experiment.
2. Keep defaults conservative and close to a known-good starting point.
3. Name experiments so the run directory is self-explanatory.
4. Keep smoke, pilot, and longer runs clearly separated.
5. Record the intent of each config in comments when helpful.

## Standard workflow

### 1. Start from the nearest stable config

Prefer copying the closest existing config instead of starting from scratch.

Typical categories:

- smoke validation
- pilot training
- ablation run
- heavier comparison run

### 2. Define the ablation axis explicitly

Choose one primary axis such as:

- grouped action granularity
- replay strategy
- reward shaping strength
- path or coverage sample count
- evaluation frequency
- deterministic collection probability
- checkpoint cadence
- TensorFlow device mode

Avoid changing several unrelated axes in one new config unless the user asked for a compound experiment.

### 3. Keep naming systematic

Use experiment names that expose the comparison axis, for example:

- `sac_ris_smoke`
- `sac_ris_pilot`
- `sac_ris_ablation_reward_margin_02`
- `sac_ris_ablation_grouped_8x8`

### 4. Preserve interpretability

When generating multiple configs, include a short comparison note listing:

- what changed
- what stayed fixed
- what outcome the run is intended to test

## Good tasks for this skill

- create three reward-margin ablation configs
- compare prioritized replay on vs off
- compare grouped action 8x8 vs 10x10 vs 20x20
- create a cheaper smoke config that validates imports and one short episode
- add a pilot config with slightly longer evaluation but unchanged model structure

## Anti-patterns

Do not:

- bury major experiment changes directly in `config.yaml` without creating a named variant
- change code when a config-only change is sufficient
- create ablations that are impossible to compare because too many axes changed at once
- reuse the same `experiment_name` for meaningfully different runs

## Output expectations

When you use this skill, provide:

1. the base config used
2. the ablation axis
3. the list of generated config files
4. the expected interpretation of each run
