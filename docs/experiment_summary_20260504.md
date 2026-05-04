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

## Recommended Deployment Protocol

- config:
  [configs/pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval.yaml](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/configs/pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval.yaml)
- run:
  [runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530)
- deployment checkpoint:
  [recommended_agent.pt](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/recommended_agent.pt)

Why this protocol is now preferred for reporting or deployment:

- it preserves the strongest current training trajectory
- it compares multiple candidate checkpoints under one stricter deterministic
  evaluation protocol
- it decouples in-training `best_eval` tracking from final model selection
- it selected `checkpoint_episode_0036.pt` over both `best_eval_agent.pt` and
  `final_agent.pt` in the validated run
- it exports a single stable artifact name for downstream scripts and reports

## Formal Fixed Blind-Spot Evaluation

- output dir:
  [formal_fixed_blind_spot_eval](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/formal_fixed_blind_spot_eval)
- report:
  [formal_fixed_blind_spot_report.md](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/formal_fixed_blind_spot_eval/formal_fixed_blind_spot_report.md)
- summary:
  [formal_fixed_blind_spot_summary.yaml](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/formal_fixed_blind_spot_eval/formal_fixed_blind_spot_summary.yaml)

This stricter evaluation uses the cached blind-spot candidate set directly,
forces `rx_jitter_xy_m = 0.0`, and runs 4 deterministic rollouts per candidate.

Key aggregate results for the exported `recommended_agent.pt`:

- `no_ris_rate_mean_over_candidates = 9.766755`
- `phase_gradient_reflector_rate_mean_over_candidates = 11.855664`
- `drl_avg_rate_mean_over_candidates = 9.742294`
- `drl_final_rate_mean_over_candidates = 9.742294`
- `candidate_mean_avg_beats_reflector_rate = 0.2`
- `candidate_mean_avg_beats_no_ris_rate = 0.6`

Interpretation:

- the re-eval protocol is still the right deployment-selection mechanism
- but the currently recommended checkpoint is not yet strong enough on the full
  fixed hard-RX set
- on this stricter set-level metric, the classical phase-gradient reflector is
  still clearly stronger than the DRL policy

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
- the newest improvement is procedural rather than architectural:
  final deployment should use post-training deterministic checkpoint re-ranking
  instead of blindly trusting the last `best_eval_agent.pt`

## Report-Ready Figures

The current mainline run already has generated figures:

- [learning_curve_best_eval_checkpoint.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046/learning_curve_best_eval_checkpoint.png)
- [phase_profile_best_eval_checkpoint.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046/phase_profile_best_eval_checkpoint.png)
- [coverage_comparison_best_eval_checkpoint.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_20260504_145046/coverage_comparison_best_eval_checkpoint.png)

The current re-eval-selected run also has generated figures:

- [learning_curve_best_final_reeval.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/learning_curve_best_final_reeval.png)
- [phase_profile_best_final_reeval.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/phase_profile_best_final_reeval.png)
- [coverage_comparison_best_final_reeval.png](/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/coverage_comparison_best_final_reeval.png)

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
