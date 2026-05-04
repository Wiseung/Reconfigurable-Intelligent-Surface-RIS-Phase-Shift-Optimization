# Formal Fixed Blind-Spot Evaluation

## Protocol

- Run directory: `/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530`
- Config snapshot: `/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/resolved_config.yaml`
- Checkpoint: `/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_cpu_48ep_hard_replay_exclusive_actor_gate_best_eval_reeval_20260504_185530/recommended_agent.pt`
- TensorFlow runs on CPU and PyTorch runs on GPU when available.
- RX jitter is forced to `0.0 m` so every blind-spot candidate is evaluated at a fixed position.
- Each candidate is re-evaluated with `4` deterministic rollouts.

## Aggregate Results

| metric | mean_over_candidates | std_over_candidates |
| --- | --- | --- |
| No RIS rate | 9.766755 | 2.521621 |
| Phase-gradient reflector rate | 11.855664 | 1.146259 |
| DRL average episode rate | 9.742294 | 2.572220 |
| DRL final-step rate | 9.742294 | 2.572220 |
| DRL avg minus reflector | -2.113370 | 2.644522 |
| DRL final minus reflector | -2.113370 | 2.644522 |

## Win Rates

| comparison | rate |
| --- | --- |
| Candidate-mean DRL avg > reflector | 0.200000 |
| Candidate-mean DRL final > reflector | 0.200000 |
| Candidate-mean DRL avg > no RIS | 0.600000 |
| Candidate-mean DRL final > no RIS | 0.600000 |
| Rollout DRL avg > reflector | 0.200000 |
| Rollout DRL final > reflector | 0.200000 |

## Candidate-Level Results

| candidate_index | rx_x_m | rx_y_m | no_ris_rate_mean | phase_gradient_reflector_rate_mean | drl_avg_rate_mean | drl_final_rate_mean | avg_minus_reflector_mean | final_minus_reflector_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | -6.000000 | 86.000000 | 4.734995 | 11.362795 | 4.609275 | 4.609275 | -6.753519 | -6.753519 |
| 1 | -14.000000 | 62.000000 | 10.778571 | 12.502798 | 10.797489 | 10.797489 | -1.705309 | -1.705309 |
| 2 | -22.000000 | 102.000000 | 10.948647 | 13.219586 | 10.910268 | 10.910268 | -2.309317 | -2.309317 |
| 3 | -30.000000 | 102.000000 | 11.070988 | 12.299717 | 11.091199 | 11.091199 | -1.208519 | -1.208519 |
| 4 | -14.000000 | 54.000000 | 11.300574 | 9.893426 | 11.303240 | 11.303240 | 1.409813 | 1.409813 |

## Representative Candidate

- Representative candidate index: `1` (chosen as the DRL-average-rate candidate closest to the set mean unless explicitly overridden).
- Representative RX position: `[-14.0, 62.0, 1.5]`

## Exported Figures

- Learning curve: `formal_learning_curve.png`
- Candidate rate comparison: `formal_fixed_blind_spot_rate_by_candidate.png`
- Aggregate bar chart: `formal_fixed_blind_spot_aggregate_bar.png`
- Reflector margin chart: `formal_fixed_blind_spot_margin_vs_reflector.png`
- Representative phase profile: `formal_fixed_blind_spot_phase_profile_representative.png`
- Representative coverage comparison: `formal_fixed_blind_spot_coverage_representative.png`
