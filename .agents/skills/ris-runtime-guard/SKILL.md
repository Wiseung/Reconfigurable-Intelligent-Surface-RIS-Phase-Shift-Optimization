---
name: ris-runtime-guard
description: Stabilize runtime setup for the RIS phase-shift optimization repository. Use when Codex needs to run, debug, or modify mixed TensorFlow/Sionna and PyTorch execution, especially around SIONNA_TF_DEVICE, CUDA visibility, import order, smoke tests, environment setup, dependency mismatches, TensorFlow-on-CPU and PyTorch-on-GPU coexistence, or runtime failures before or during training. Do not use for SAC algorithm design, reward shaping, scene geometry tuning, or experiment plotting.
---

# RIS Runtime Guard

Treat runtime safety as the first constraint. Preserve the repository's verified operating model unless the user explicitly asks to change it.

## Repository assumptions

- Keep Sionna and TensorFlow on CPU by default.
- Keep PyTorch on GPU by default when CUDA is available.
- Set `SIONNA_TF_DEVICE` before importing `env_sionna.py`.
- Avoid edits that silently move TensorFlow initialization earlier in the process.
- Prefer minimal, reversible changes.

## Primary files

- `env_sionna.py`
- `train_loop.py`
- `requirements.txt`
- `README.md`
- `docs/runtime_and_experiment_guide.md`

## Working rules

1. Check the current import order before changing runtime code.
2. Preserve `_configure_tensorflow_runtime()` semantics unless the task explicitly requires a redesign.
3. Keep CPU/GPU ownership explicit. Do not let TensorFlow opportunistically claim GPU memory unless the user requested GPU-backed TensorFlow.
4. If you change startup behavior, also update the run instructions.
5. Prefer adding a smoke test over making undocumented assumptions.

## Standard workflow

### 1. Inspect runtime entry points

Read these first:

- `train_loop.py` for environment setup and `SIONNA_TF_DEVICE`
- `env_sionna.py` for module-level TensorFlow behavior
- `README.md` and runtime docs for the intended launch flow

Summarize the actual startup order before editing.

### 2. Classify the failure mode

Put the issue in one of these buckets:

- import-order bug
- TensorFlow device visibility bug
- CUDA memory contention
- missing package or wrong package version
- Sionna RT API mismatch
- smoke-run failure
- logging or artifact path issue

Fix the narrowest layer possible.

### 3. Apply guarded edits

Typical safe edits include:

- moving environment-variable setup earlier
- restoring the previous `CUDA_VISIBLE_DEVICES` behavior after importing `env_sionna.py`
- adding explicit error messages
- adding a small validation script or command
- tightening configuration documentation

Avoid broad refactors unless required.

### 4. Validate with the cheapest useful run

Prefer this order:

1. import smoke test
2. environment construction test
3. single-episode or smoke config run
4. pilot run

Do not jump straight to long training unless asked.

### 5. Document the fix

Whenever you change runtime behavior, update:

- command to reproduce
- expected environment variables
- what must happen before `env_sionna.py` import
- whether TensorFlow is expected to use CPU or GPU

## Safe edit patterns

### Import smoke test

Create or update a tiny check that verifies:

- `SIONNA_TF_DEVICE` is set before environment import
- `SionnaRISEnv` constructs successfully
- `env.reset()` returns a finite state vector

### Minimal launch verification

Prefer a short command such as:

```bash
python train_loop.py --config configs/smoke_cpu.yaml
```

If that config does not exist, use the closest smoke or pilot config already present in the repo and keep the run short.

## Anti-patterns

Do not:

- enable GPU TensorFlow implicitly
- delete runtime guards to "simplify" the code
- move TensorFlow import logic into unrelated files without a strong reason
- mix dependency upgrades with algorithm changes in one patch
- treat a long training run as the first validation step

## Output expectations

When you use this skill, produce:

1. a short diagnosis of the runtime issue
2. the minimal code or config change
3. the exact validation command
4. any required README or doc update

If the failure is caused by the local machine rather than repository code, say so clearly and keep the code untouched unless a repository-side mitigation is justified.
