---
name: ris-train-loop-reward-shaping
description: Edit the training orchestration, reward shaping, baseline reference logic, evaluation cadence, logging outputs, and experiment scheduling for the RIS phase-shift optimization repository. Use when Codex needs to modify train_loop.py, including warmup behavior, policy delay, updates-per-step, deterministic evaluation, reward margin logic, replay priority shaping, checkpoint cadence, config merging, or metrics writing. Do not use for low-level SAC architecture work, TensorFlow runtime setup, or visualization-only tasks.
---

# RIS Train Loop and Reward Shaping

Treat `train_loop.py` as experiment control, not as a dumping ground for unrelated logic.

## Primary file

- `train_loop.py`

## Core contract to preserve

- config is loaded and merged predictably
- environment is created before training starts
- agent construction depends on resolved `state_dim`
- training produces run artifacts in a dedicated run directory
- per-episode metrics remain machine-readable
- short runs stay easy to execute

## Training-loop principles

1. Keep orchestration readable from top to bottom.
2. Prefer explicit schedules over implicit side effects.
3. Preserve the meaning of existing metric names unless there is a strong reason to rename them.
4. If adding a new metric, also wire it through JSONL, CSV, and optional TensorBoard logging where appropriate.
5. Keep reward shaping separable from raw environment reward.

## Standard workflow

### 1. Find the real control point

Choose the narrowest correct section:

- config defaults
- action collection schedule
- update cadence
- reward shaping
- replay priority policy
- deterministic evaluation
- checkpointing
- run artifact logging

Edit that section instead of scattering logic across the file.

### 2. Preserve raw vs shaped reward distinction

If the task changes reward behavior:

- keep raw environment reward observable
- keep shaped reward explicit
- keep the reference baseline identifiable
- avoid burying shaping inside unrelated code paths

### 3. Keep configs first-class

If a new behavior needs tuning, expose it through config with a conservative default. Avoid hidden constants in the body of the loop.

### 4. Preserve experiment traceability

If you add or change metrics, make sure a future run can still answer:

- what baseline was used
- what schedule was used
- what exploration setting was used
- what checkpoint corresponds to the logged metrics

## Good tasks for this skill

- add a new reward-baseline option
- change warmup or exploration schedules
- add a new evaluation cadence
- improve checkpoint naming
- add a summary metric for ablations
- refine replay priority weighting for hard examples

## Anti-patterns

Do not:

- move core SAC math into this file
- change environment public methods here unless absolutely necessary
- add visualization code into the training loop
- hide key hyperparameters in local variables only
- let config keys drift from what is actually logged

## Validation checklist

After edits, verify:

- the config still resolves correctly
- a short run still writes `resolved_config.yaml`
- per-episode rows still land in JSONL and CSV
- reward and baseline metrics are not silently dropped
- checkpoint cadence still behaves as expected

## Output expectations

When you use this skill, provide:

1. the specific training control being changed
2. any new config keys and defaults
3. any new metrics emitted
4. the exact short-run command to validate the change
