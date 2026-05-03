# Runtime and Experiment Guide

## Why The Runtime Strategy Changed

This repository originally targeted a mixed `TensorFlow + PyTorch` GPU runtime
with a hard TensorFlow memory cap. On this machine, that approach is not the
most stable option because:

- the GPU is an `RTX 5090`, i.e. compute capability `12.0`
- the TensorFlow side is pinned to `2.15.x`
- Sionna RT uses Mitsuba/Dr.Jit under the hood and selects its backend at import time

In practice, allowing TensorFlow/Sionna RT to use CUDA caused PTX JIT and
cross-runtime instability. The validated strategy is therefore:

- `TensorFlow + Sionna RT` on CPU
- `PyTorch` on GPU

This avoids the PTX JIT path while still letting the DRL agent use the RTX 5090.

## Import-Order Rule

`Sionna RT` decides between `cuda_ad_rgb` and `llvm_ad_rgb` when `sionna.rt` is
imported. Because of that, the runtime mode must be fixed before importing
`env_sionna.py`.

The repo now does this in `train_loop.py`:

1. Read `config.yaml`
2. Set `SIONNA_TF_DEVICE`
3. Import `env_sionna.py`
4. Let `env_sionna.py` hide CUDA from TensorFlow when `tf_device: cpu`
5. Let Sionna RT initialize with `llvm_ad_rgb`

Do not move the `from env_sionna import SionnaRISEnv` import back to the
module top level unless you also preserve this ordering another way.

## Default Runtime Parameters

The default `config.yaml` now uses:

- `tf_device: cpu`
- `tf_memory_limit_mb: 14336`
- `max_depth: 2`
- `path_num_samples: 6000`
- `coverage_num_samples: 30000`
- `probe_num_samples: 64`
- `state_num_paths: 64`
- fixed `tx_boresight_target: [0.0, 90.0, 1.5]`
- pilot-region `blind_spot_search_center: [-10.0, 90.0, 1.5]`
- static pilot RIS location `[-180.0, 100.0, 20.0]`

Rationale:

- `max_depth=3` and `num_samples=1e6` were scientifically attractive but too
  expensive for stable day-to-day iteration on CPU Sionna RT in this mixed stack.
- path solves and coverage solves have different cost profiles, so they are now
  budgeted separately instead of sharing one oversized global sample count.
- the visibility probe only needs a tiny budget and should not inherit the main
  path-tracing cost.
- the reward/state path extractor now uses full, non-clustered CIR paths because
  the local Sionna build can emit `NaN` for clustered RIS paths.
- the RIS is oriented using the Sionna tutorial convention
  `look_at((TX + RX) / 2)`, which was necessary to obtain action-sensitive rewards
  in the `etoile` scene.

## Recommended Experiment Ladder

### 1. End-to-end smoke run

Use [configs/smoke_cpu.yaml](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/configs/smoke_cpu.yaml).

Goal:

- verify `env -> agent -> replay -> update -> logging` works end to end
- confirm artifacts are produced
- avoid spending long CPU time before the pipeline is validated

### 2. Short CPU-Sionna pilot

Use [configs/pilot_cpu.yaml](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/configs/pilot_cpu.yaml).

Goal:

- inspect early reward trends
- verify that the chosen `etoile` blind-spot region gives nonzero baselines
- confirm that random RIS actions perturb the reward before longer training

### 3. Longer training

Once the pilot is stable:

- increase `num_episodes`
- optionally increase `path_num_samples`
- keep `tf_device: cpu` unless TensorFlow/Sionna versions are upgraded to a stack that natively supports the GPU without PTX JIT

## Known Runtime Signals

Expected:

- TensorFlow may still print generic CUDA discovery messages because it was
  built with CUDA support
- `tf.config.list_physical_devices("GPU")` should end up empty in the CPU runtime
- Mitsuba variant should be `llvm_ad_rgb`

Not expected:

- `TensorFlow was not built with CUDA kernel binaries compatible with compute capability 12.0`
- `GPU:0 unknown device`
- Sionna RT selecting `cuda_ad_rgb`

If any of those appear, check whether:

- `SIONNA_TF_DEVICE` was set before importing `env_sionna.py`
- another module imported `sionna.rt` too early
- a custom script bypassed `train_loop.py`

## Training Commands

Smoke run:

```bash
/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/.conda-py310/bin/python train_loop.py --config configs/smoke_cpu.yaml
```

Default run:

```bash
/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/.conda-py310/bin/python train_loop.py --config config.yaml
```

Validated pilot run:

```bash
/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/.conda-py310/bin/python train_loop.py --config configs/pilot_cpu.yaml
```

## Output Artifacts

Each run directory under `runs/` contains:

- `resolved_config.yaml`
- `training.log`
- `metrics.jsonl`
- `episode_metrics.csv`
- `tensorboard/`
- checkpoints
- `final_agent.pt`

These outputs are the intended inputs for:

- `plot_learning_curve()`
- `plot_phase_profile()`
- `plot_coverage_comparison()`

## Experimental Caveat

The current CPU-Sionna default is the most stable engineering choice for this
machine and software stack, but it is not the final word on scientific fidelity.
For publication-grade results, revisit:

- path depth
- coverage-map sample density
- state representation
- baseline geometry
- reward design

after the end-to-end pipeline has been validated.
