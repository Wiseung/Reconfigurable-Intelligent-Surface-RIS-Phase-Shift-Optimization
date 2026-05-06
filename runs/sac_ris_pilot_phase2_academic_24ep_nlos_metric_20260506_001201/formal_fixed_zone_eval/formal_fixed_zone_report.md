# Formal Fixed-Zone Evaluation

## Protocol

- Run directory: `/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_phase2_academic_24ep_nlos_metric_20260506_001201`
- Config snapshot: `/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_phase2_academic_24ep_nlos_metric_20260506_001201/resolved_config.yaml`
- Checkpoint: `/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_phase2_academic_24ep_nlos_metric_20260506_001201/recommended_agent.pt`
- TensorFlow runs on CPU and PyTorch runs on GPU when available.
- RX jitter is forced to `0.0 m` so every LOS/NLOS candidate is evaluated at a fixed position.
- Each candidate is re-evaluated with `4` deterministic rollouts.

## Zone-Level Results

| zone_name | num_candidates | no_ris_rate_mean | phase_gradient_reflector_rate_mean | drl_avg_rate_mean | drl_final_rate_mean | avg_minus_reflector_mean | final_minus_reflector_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| los | 4 | 17.556660 | 17.501489 | 17.556398 | 17.556398 | 0.054908 | 0.054908 |
| nlos | 4 | 10.922554 | 12.388848 | 10.935661 | 10.935661 | -1.453187 | -1.453187 |

## Candidate-Level Results

| zone_name | candidate_index | rx_x_m | rx_y_m | no_ris_rate_mean | phase_gradient_reflector_rate_mean | drl_avg_rate_mean | drl_final_rate_mean | avg_minus_reflector_mean | final_minus_reflector_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| los | 0 | -118.000000 | 6.000000 | 18.553603 | 18.553603 | 18.553603 | 18.553603 | 0.000000 | 0.000000 |
| los | 1 | -86.000000 | -10.000000 | 17.737473 | 17.516792 | 17.736426 | 17.736426 | 0.219633 | 0.219633 |
| los | 2 | -102.000000 | -10.000000 | 16.685164 | 16.685164 | 16.685164 | 16.685164 | 0.000000 | 0.000000 |
| los | 3 | -134.000000 | 6.000000 | 17.250399 | 17.250399 | 17.250399 | 17.250399 | 0.000000 | 0.000000 |
| nlos | 0 | -46.000000 | 126.000000 | 10.721343 | 12.177482 | 10.737136 | 10.737136 | -1.440346 | -1.440346 |
| nlos | 1 | -6.000000 | 118.000000 | 10.816306 | 13.127376 | 10.840921 | 10.840921 | -2.286455 | -2.286455 |
| nlos | 2 | -46.000000 | 86.000000 | 10.878767 | 13.336188 | 10.889252 | 10.889252 | -2.446936 | -2.446936 |
| nlos | 3 | -14.000000 | 86.000000 | 11.273800 | 10.914345 | 11.275335 | 11.275335 | 0.360990 | 0.360990 |

## Representative Candidate

- Zone: `nlos`
- Candidate index: `2`
- RX position: `[-46.0, 86.0, 1.5]`

## Exported Figures

- `formal_zone_learning_curve.png`
- `formal_fixed_zone_rate_by_candidate.png`
- `formal_fixed_zone_aggregate_bar.png`
- `formal_fixed_zone_phase_profile_representative.png`
- `formal_fixed_zone_coverage_representative.png`
