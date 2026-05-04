# Experiment Summary 2026-05-04

This note consolidates the main pilot ablations completed on 2026-05-04 and
records the currently recommended formal baseline for the repository.

## Recommended Mainline

- config:
  [configs/pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval.yaml](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/configs/pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval.yaml)
- run:
  [runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046)
- best checkpoint:
  [best_eval_agent.pt](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046/best_eval_agent.pt)

Why this line is preferred:

- it preserves the more reliable `rx_block_episodes=2` schedule
- it beats the earlier `rx2` line on best deterministic evaluation
- it avoids the final-eval collapse observed on the `rx3` family
- it now stores the best deterministic checkpoint explicitly

## Result Table

| Branch | Run | avg_all | avg_tail10 | avg_tail5 | best_eval | last_eval |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `rx2` actor-gate baseline | `...132700` | 9.313652 | 8.634944 | 8.695944 | 11.243512 | 11.243512 |
| `rx2` best-eval mainline | `...145046` | 9.285641 | 8.577790 | 8.686258 | 11.301114 | 11.301114 |
| `rx3` high-return branch | `...141408` | 9.933273 | 11.008738 | 10.950314 | 11.201273 | 8.371767 |
| `rx3` best-checkpoint branch | `...144004` | 9.904858 | 11.012259 | 10.961141 | 11.228499 | 8.314148 |

## Interpretation

- `rx3` is still the strongest branch if only training averages are considered.
- `rx3` is not the recommended formal line because deterministic final
  generalization remains weak.
- `rx2 + exclusive hard replay + actor gate + dense deterministic eval + best
  checkpoint capture` is the best current balance between training stability
  and final deterministic evaluation quality.

## Report-Ready Figures

The current mainline run already has generated figures:

- [learning_curve_best_eval_checkpoint.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046/learning_curve_best_eval_checkpoint.png)
- [phase_profile_best_eval_checkpoint.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046/phase_profile_best_eval_checkpoint.png)
- [coverage_comparison_best_eval_checkpoint.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046/coverage_comparison_best_eval_checkpoint.png)

## Caveat

The run-level best deterministic policy is still not uniformly stronger than
the phase-gradient reflector on every sampled hard RX point. A representative
visualization case recorded in
[best_eval_visualization_summary.yaml](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046/best_eval_visualization_summary.yaml)
shows a local hard point where:

- `no_ris_rate = 5.635887`
- `phase_gradient_reflector_rate = 11.535450`
- `best_eval_policy_rate_at_sampled_rx = 5.263661`

This should guide the next round of work toward stronger local hard-RX
robustness rather than only maximizing average training reward.
