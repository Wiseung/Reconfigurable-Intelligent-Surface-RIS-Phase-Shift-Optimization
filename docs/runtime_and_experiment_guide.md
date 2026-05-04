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

### 4. Current formal pilot line

Use [configs/pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval.yaml](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/configs/pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval.yaml).

This is the current recommended mainline because it balances three things
better than the other branches we tested:

- deterministic final evaluation quality
- stability on the more reliable `rx_block_episodes=2` schedule
- explicit checkpoint retention based on `eval_avg_reward`

Key mechanisms enabled in this config:

- `hard_replay_route_mode: exclusive`
- `hard_actor_update_gap_threshold: 4.0`
- `hard_actor_policy_delay: 6`
- `eval_every_episodes: 2`
- `save_best_eval_checkpoint: true`
- `best_eval_metric: eval_avg_reward`

Recommended command:

```bash
/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/.conda-py310/bin/python train_loop.py --config configs/pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval.yaml
```

Current best validated run:

- run dir:
  [runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046)
- best checkpoint:
  [best_eval_agent.pt](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046/best_eval_agent.pt)

Measured summary:

- `avg_all = 9.285641`
- `avg_tail10 = 8.577790`
- `avg_tail5 = 8.686258`
- `best_eval = 11.301114`
- `last_eval = 11.301114`

Interpretation:

- This line is slightly stronger than the older `rx2` baseline on best
  deterministic eval (`11.301114` vs `11.243512`).
- The `rx3` line still produces higher training averages and tail training
  rewards, but it remains weaker on final deterministic generalization.
- Best-checkpoint capture should now be treated as part of the standard run
  protocol, not an optional add-on.

### 5. Recommended deployment-selection protocol

Use [configs/pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval.yaml](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/configs/pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval.yaml).

This protocol keeps the same training dynamics as the current mainline but adds
one important post-training step:

- deterministic re-evaluation of multiple candidate checkpoints

The intended behavior is:

- keep `best_eval_agent.pt` as the best in-training checkpoint
- keep periodic checkpoints for candidate comparison
- keep `final_agent.pt` as the literal training endpoint
- run a final deterministic re-ranking pass and save the winner as
  `best_final_reeval_agent.pt`
- always export `recommended_agent.pt` as the single downstream handoff file

Validated run:

- run dir:
  [runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530)
- selected checkpoint:
  [best_final_reeval_agent.pt](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/best_final_reeval_agent.pt)

What happened in this run:

- training metrics matched the current best mainline trajectory
- final 4-episode deterministic re-evaluation preferred
  `checkpoint_episode_0036.pt`
- `checkpoint_episode_0036.pt` beat both `best_eval_agent.pt` and `final_agent.pt`
  on the stricter post-training evaluation protocol

That means the recommended reporting flow is now:

1. train with dense deterministic eval and periodic checkpoints
2. store `best_eval_agent.pt` during training
3. run final multi-checkpoint deterministic re-evaluation
4. report and deploy `recommended_agent.pt`

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

The same import-order rule also applies to post-training analysis scripts.
`evaluate_fixed_blind_spots.py` deliberately constructs `SionnaRISEnv` before
importing the PyTorch SAC module so that Sionna RT stays on `llvm_ad_rgb`.

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

Formal fixed blind-spot-set evaluation for the current recommended deployment
checkpoint:

```bash
/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/.conda-py310/bin/python evaluate_fixed_blind_spots.py --run-dir runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530 --num-rollouts-per-candidate 4
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

When final checkpoint re-evaluation is enabled, the run can also contain:

- `best_final_reeval_agent.pt`
- `recommended_agent.pt`
- `final_checkpoint_reeval_candidate` events in `metrics.jsonl`
- `final_checkpoint_reeval_best` event in `metrics.jsonl`
- `recommended_checkpoint` event in `metrics.jsonl`

When the formal fixed-set evaluation is run on a deployment checkpoint, the
selected run directory can also contain:

- `formal_fixed_blind_spot_eval/formal_fixed_blind_spot_rollout_metrics.csv`
- `formal_fixed_blind_spot_eval/formal_fixed_blind_spot_candidate_metrics.csv`
- `formal_fixed_blind_spot_eval/formal_fixed_blind_spot_aggregate_summary.csv`
- `formal_fixed_blind_spot_eval/formal_fixed_blind_spot_summary.yaml`
- `formal_fixed_blind_spot_eval/formal_fixed_blind_spot_report.md`
- `formal_fixed_blind_spot_eval/formal_learning_curve.png`
- `formal_fixed_blind_spot_eval/formal_fixed_blind_spot_rate_by_candidate.png`
- `formal_fixed_blind_spot_eval/formal_fixed_blind_spot_margin_vs_reflector.png`
- `formal_fixed_blind_spot_eval/formal_fixed_blind_spot_aggregate_bar.png`
- `formal_fixed_blind_spot_eval/formal_fixed_blind_spot_phase_profile_representative.png`
- `formal_fixed_blind_spot_eval/formal_fixed_blind_spot_coverage_representative.png`

These outputs are the intended inputs for:

- `plot_learning_curve()`
- `plot_phase_profile()`
- `plot_coverage_comparison()`

For the current formal mainline run, the generated report-ready figures are:

- [learning_curve_best_eval_checkpoint.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046/learning_curve_best_eval_checkpoint.png)
- [phase_profile_best_eval_checkpoint.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046/phase_profile_best_eval_checkpoint.png)
- [coverage_comparison_best_eval_checkpoint.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046/coverage_comparison_best_eval_checkpoint.png)
- [best_eval_visualization_summary.yaml](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046/best_eval_visualization_summary.yaml)

The saved visualization summary also records one representative sampled blind
spot RX where:

- `no_ris_rate = 5.635887`
- `phase_gradient_reflector_rate = 11.535450`
- `best_eval_policy_rate_at_sampled_rx = 5.263661`

This is a useful reminder that run-level average improvements and local hard-RX
performance are not yet the same thing in the current environment.

For the current re-eval-selected checkpoint, the generated report-ready figures
are:

- [learning_curve_best_final_reeval.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/learning_curve_best_final_reeval.png)
- [phase_profile_best_final_reeval.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/phase_profile_best_final_reeval.png)
- [coverage_comparison_best_final_reeval.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/coverage_comparison_best_final_reeval.png)
- [best_final_reeval_visualization_summary.yaml](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/best_final_reeval_visualization_summary.yaml)

For the stricter fixed blind-spot set evaluation of the exported
`recommended_agent.pt`, the generated report-ready artifacts are:

- [formal_fixed_blind_spot_report.md](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/formal_fixed_blind_spot_eval/formal_fixed_blind_spot_report.md)
- [formal_fixed_blind_spot_summary.yaml](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/formal_fixed_blind_spot_eval/formal_fixed_blind_spot_summary.yaml)
- [formal_learning_curve.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/formal_fixed_blind_spot_eval/formal_learning_curve.png)
- [formal_fixed_blind_spot_rate_by_candidate.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/formal_fixed_blind_spot_eval/formal_fixed_blind_spot_rate_by_candidate.png)
- [formal_fixed_blind_spot_margin_vs_reflector.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/formal_fixed_blind_spot_eval/formal_fixed_blind_spot_margin_vs_reflector.png)
- [formal_fixed_blind_spot_aggregate_bar.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/formal_fixed_blind_spot_eval/formal_fixed_blind_spot_aggregate_bar.png)
- [formal_fixed_blind_spot_phase_profile_representative.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/formal_fixed_blind_spot_eval/formal_fixed_blind_spot_phase_profile_representative.png)
- [formal_fixed_blind_spot_coverage_representative.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/formal_fixed_blind_spot_eval/formal_fixed_blind_spot_coverage_representative.png)

This stricter set-level evaluation currently shows that the exported DRL policy
is still below the classical reflector on average over the 5 cached blind-spot
candidates, even though the post-training re-eval protocol remains useful for
choosing the most reliable DRL checkpoint.

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
