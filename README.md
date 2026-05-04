# Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization

This project implements a mixed-framework RIS phase-shift optimization stack:

- `Sionna RT + TensorFlow` for physically grounded channel and coverage synthesis
- `PyTorch` for the SAC-based DRL agent
- `Matplotlib + Seaborn` for evaluation plots

The current validated pilot geometry uses:

- a fixed `TX` boresight instead of steering toward every sampled user
- full-path CIR extraction for reward/state instead of `cluster_ris_paths=True`
- an RIS orientation that follows the Sionna RIS tutorial convention
  `look_at((TX + RX) / 2)`

## Current Runtime Strategy

The validated runtime strategy on this machine is:

- `Sionna/TF on CPU`
- `PyTorch on GPU`

This is intentional. With `TensorFlow 2.15.x` on an `RTX 5090 (compute capability 12.0)`,
letting TensorFlow/Sionna RT use CUDA triggers PTX JIT and can break mixed-runtime
stability. The repo therefore defaults to:

- `tf_device: cpu`
- Mitsuba/Sionna RT variant: `llvm_ad_rgb`
- PyTorch device: `cuda` when available

If you change this, do it carefully: `Sionna RT` chooses its backend at import
time, so the environment must be configured before `env_sionna.py` is imported.

## Repository Layout

- [env_sionna.py](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/env_sionna.py)
  Sionna RT environment, RIS control, coverage maps, CIR-based state/reward
- [agent_drl.py](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/agent_drl.py)
  Grouped-action SAC agent and CPU-only replay buffer
- [train_loop.py](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/train_loop.py)
  Main RL training loop and TF/NumPy/Torch bridge
- [utils_vis.py](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/utils_vis.py)
  Coverage, learning-curve, and phase-profile plotting
- [config.yaml](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/config.yaml)
  Default CPU-Sionna training configuration
- [configs/pilot_cpu.yaml](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/configs/pilot_cpu.yaml)
  Validated pilot configuration with a nonzero baseline and action-sensitive reward
- [configs/smoke_cpu.yaml](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/configs/smoke_cpu.yaml)
  Small end-to-end smoke configuration for quick validation

## Quick Start

Use the project environment:

```bash
/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/.conda-py310/bin/python train_loop.py --config configs/smoke_cpu.yaml
```

Use the validated pilot configuration:

```bash
/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/.conda-py310/bin/python train_loop.py --config configs/pilot_cpu.yaml
```

Use the current formal 48-episode mainline configuration:

```bash
/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/.conda-py310/bin/python train_loop.py --config configs/pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval.yaml
```

Use the default longer CPU-Sionna configuration:

```bash
/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/.conda-py310/bin/python train_loop.py --config config.yaml
```

Artifacts are written under `runs/` and include:

- `metrics.jsonl`
- `episode_metrics.csv`
- `training.log`
- `tensorboard/` if TensorBoard is available
- checkpoints and `final_agent.pt`

## Current Mainline Result

The current recommended formal pilot configuration is:

- [configs/pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval.yaml](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/configs/pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval.yaml)

This run keeps the more reliable `rx_block_episodes=2` schedule and adds:

- exclusive hard-replay routing
- actor-update gating on hard baseline gaps
- deterministic evaluation every 2 episodes
- automatic `best_eval_agent.pt` checkpoint capture

Current best validated run:

- run dir:
  [runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046)
- best checkpoint:
  [best_eval_agent.pt](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046/best_eval_agent.pt)

Headline metrics from `episode_metrics.csv` and `metrics.jsonl`:

- `avg_all = 9.285641`
- `avg_tail10 = 8.577790`
- `avg_tail5 = 8.686258`
- `best_eval = 11.301114`
- `last_eval = 11.301114`

This slightly improves the previous `rx2` mainline best deterministic eval
(`11.243512`) while preserving much better final reliability than the `rx3`
branch, which reached higher training returns but collapsed on final
deterministic evaluation.

Generated evaluation artifacts for the current mainline run:

- learning curve:
  [learning_curve_best_eval_checkpoint.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046/learning_curve_best_eval_checkpoint.png)
- phase profile:
  [phase_profile_best_eval_checkpoint.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046/phase_profile_best_eval_checkpoint.png)
- coverage comparison:
  [coverage_comparison_best_eval_checkpoint.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046/coverage_comparison_best_eval_checkpoint.png)
- visualization summary:
  [best_eval_visualization_summary.yaml](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046/best_eval_visualization_summary.yaml)

One important caveat from the sampled visualization RX: the best checkpoint
still underperforms the phase-gradient reflector on some local hard-user
positions. That matches the run-level observation that average deterministic
performance improved, but the hardest blind-spot realizations are not yet fully
solved.

## Documentation

- Literature and design blueprint:
  [docs/literature_to_implementation.md](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/docs/literature_to_implementation.md)
- Runtime and experiment guide:
  [docs/runtime_and_experiment_guide.md](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/docs/runtime_and_experiment_guide.md)
