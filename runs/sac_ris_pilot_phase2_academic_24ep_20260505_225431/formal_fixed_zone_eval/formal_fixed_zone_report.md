# Formal Fixed-Zone Evaluation

## Protocol

- Run directory: `/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_phase2_academic_24ep_20260505_225431`
- Config snapshot: `/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_phase2_academic_24ep_20260505_225431/resolved_config.yaml`
- Checkpoint: `/home/developer716/workspace/Reconfigurable-Intelligent-Surface-RIS-Phase-Shift-Optimization/runs/sac_ris_pilot_phase2_academic_24ep_20260505_225431/recommended_agent.pt`
- TensorFlow runs on CPU and PyTorch runs on GPU when available.
- RX jitter is forced to `0.0 m` so every LOS/NLOS candidate is evaluated at a fixed position.
- Each candidate is re-evaluated with `4` deterministic rollouts.

## Zone-Level Results

| zone_name | num_candidates | no_ris_rate_mean | phase_gradient_reflector_rate_mean | drl_avg_rate_mean | drl_final_rate_mean | avg_minus_reflector_mean | final_minus_reflector_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| los | 4 | 17.556660 | 17.501489 | 17.557117 | 17.557117 | 0.055627 | 0.055627 |
| nlos | 4 | 10.922554 | 12.388848 | 10.923044 | 10.923044 | -1.465804 | -1.465804 |

## Candidate-Level Results

| zone_name | candidate_index | rx_x_m | rx_y_m | no_ris_rate_mean | phase_gradient_reflector_rate_mean | drl_avg_rate_mean | drl_final_rate_mean | avg_minus_reflector_mean | final_minus_reflector_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| los | 0 | -118.000000 | 6.000000 | 18.553603 | 18.553603 | 18.553603 | 18.553603 | 0.000000 | 0.000000 |
| los | 1 | -86.000000 | -10.000000 | 17.737473 | 17.516792 | 17.739302 | 17.739302 | 0.222509 | 0.222509 |
| los | 2 | -102.000000 | -10.000000 | 16.685164 | 16.685164 | 16.685164 | 16.685164 | 0.000000 | 0.000000 |
| los | 3 | -134.000000 | 6.000000 | 17.250399 | 17.250399 | 17.250399 | 17.250399 | 0.000000 | 0.000000 |
| nlos | 0 | -46.000000 | 126.000000 | 10.721343 | 12.177482 | 10.753830 | 10.753830 | -1.423651 | -1.423651 |
| nlos | 1 | -6.000000 | 118.000000 | 10.816306 | 13.127376 | 10.777140 | 10.777140 | -2.350236 | -2.350236 |
| nlos | 2 | -46.000000 | 86.000000 | 10.878767 | 13.336188 | 10.878500 | 10.878500 | -2.457688 | -2.457688 |
| nlos | 3 | -14.000000 | 86.000000 | 11.273800 | 10.914345 | 11.282705 | 11.282705 | 0.368359 | 0.368359 |

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
