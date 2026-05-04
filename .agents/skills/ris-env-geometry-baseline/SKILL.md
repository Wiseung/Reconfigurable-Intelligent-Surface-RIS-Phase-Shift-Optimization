---
name: ris-env-geometry-baseline
description: Modify or extend the Sionna RIS environment and classical baselines for the RIS phase-shift optimization repository. Use when Codex needs to edit TX, RX, or RIS geometry, blind-spot search, coverage-map settings, path computation settings, RIS placement heuristics, baseline evaluation such as no-RIS and phase-gradient-reflector rates, or environment-level state and reward calculations in env_sionna.py. Do not use for SAC network changes, replay buffer logic, training loop scheduling, or visualization-only tasks.
---

# RIS Environment Geometry and Baseline

Work inside the existing environment design. Extend it without breaking the current training contract.

## Primary file

- `env_sionna.py`

## Core contract to preserve

- `reset()` returns a NumPy state vector
- `step(action)` accepts a flat physical RIS action and returns `(next_state, reward)`
- `evaluate_no_ris_rate()` remains available
- `evaluate_phase_gradient_reflector_rate()` remains available when native RIS support exists
- coverage-map helpers remain callable from visualization and evaluation code

## Geometry editing rules

1. Keep coordinates, search areas, and dimensions explicit and readable.
2. Prefer adding configuration hooks over hard-coding one-off numbers.
3. Preserve the blind-spot discovery logic unless the task explicitly asks to replace it.
4. Keep receiver height handling consistent.
5. If you change physical geometry assumptions, make sure the baseline methods still work.

## Baseline rules

Always preserve these baselines unless the user explicitly asks to remove one:

- no RIS
- phase-gradient reflector

If you add a new baseline, make it comparable to existing outputs and avoid changing the meaning of current metric names.

## Standard workflow

### 1. Read the environment contract

Inspect:

- constants near the top of `env_sionna.py`
- `SionnaRISEnv.__init__`
- `reset()` and `step()`
- baseline evaluators
- coverage-map functions

Write down which public methods other files depend on before editing.

### 2. Scope the change

Choose the smallest category that fits:

- geometry placement
- search-space refinement
- baseline extension
- state construction
- reward calculation
- coverage-map computation

Do not mix categories unless the task requires it.

### 3. Implement without breaking call sites

Prefer additive changes such as:

- new config parameter with sensible default
- extra helper method
- new baseline method
- optional argument on an existing method

Avoid renaming public methods already used by `train_loop.py` or `utils_vis.py`.

### 4. Re-check numerical safety

After edits, verify these conditions conceptually or by test:

- no invalid RIS action dimension mismatch
- state vector length is stable
- rates stay finite for typical cases
- no-RIS path computation still works
- phase-gradient baseline still restores RIS state correctly

## Good tasks for this skill

- change RIS search candidates
- expose carrier frequency or bandwidth more cleanly
- add a quantized phase baseline at the environment layer
- refine blind-spot candidate filtering
- change how the state selects strongest paths
- add an optional SINR-style reward function while keeping the current default

## Anti-patterns

Do not:

- change grouped action dimensions here
- move SAC logic into `env_sionna.py`
- rewrite training metrics from inside the environment
- overload one method with unrelated responsibilities

## Output expectations

When you use this skill, provide:

1. the environment-level assumption being changed
2. the exact public methods kept stable
3. the baseline implications
4. the quickest way to validate the edit
