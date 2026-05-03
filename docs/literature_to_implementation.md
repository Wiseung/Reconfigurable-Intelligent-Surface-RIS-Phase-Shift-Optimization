# Literature to Implementation Blueprint

Access date: 2026-05-03

## Source Scope

This note maps verified literature into concrete engineering decisions for the
RIS phase-shift optimization project. It focuses on three threads:

1. Sionna and differentiable ray tracing as the physical-layer simulator.
2. RIS optimization with ray-tracing-based coverage analysis.
3. DRL-based phase-shift optimization for continuous and large action spaces.

## Verified Literature Matrix

| Citation key | Year / venue | Problem | Scenario | Method | Metrics / outputs | Limits | Why it matters here |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hoydis2025-SionnaRT | 2025 / arXiv technical report | Explain Sionna RT algorithms and differentiability | General radio propagation | Differentiable RT for CIR and radio maps | CIR, radio maps, gradients, runtime tradeoffs | Technical report, not the same as a controlled DRL paper | Justifies using Sionna RT as the physical environment core and explains why coverage maps and CIR generation have different algorithmic cost profiles. |
| Hoydis2024-LearningRadioEnv | 2024 / IEEE TMLCN | Learn environment parameters by differentiable RT | Indoor calibrated scene | Gradient-based calibration of material, scattering, antenna patterns | Parameter recovery, CIR fidelity | Calibration-focused rather than RIS control | Strong peer-reviewed evidence that differentiable RT is not just for forward simulation but for trainable radio-environment modeling. |
| Hoydis2022-Sionna | 2022 / arXiv white paper | GPU-accelerated link-level research platform | General PHY simulation | TensorFlow-based end-to-end communication simulation | Reproducibility, modularity, NN integration | I did not verify an IEEE TCCN journal version for this exact title | Supports the Tensor-based system decomposition and our TensorFlow side of the project. |
| Guneser2025-UrbanRIS | 2025 / IEEE VTC-Spring | RIS optimization in urban digital twins with Sionna RT | Urban Munich-like RT scenario | Additional RIS algorithms on top of native Sionna capabilities | Coverage maps, received power, deployment comparison | Conference paper; algorithm family is not DRL-first | Direct evidence that Sionna RT plus RIS coverage maps is a meaningful benchmark setting for our environment. |
| Kilcioglu2025-IndoorRIS | 2025 / EuCAP | Indoor RIS optimization and coverage enhancement | Indoor office-like RT scene | Ray-tracing-based algorithmic optimization for blind spots | Coverage ratio, path gain, blind-zone reduction | Focuses on deployment/coverage rather than actor-critic learning | Very useful for defining non-DRL baselines and blind-spot-oriented rewards. |
| Huang2020-DDPGRIS | 2020 / IEEE JSAC | Continuous RIS phase control for multiuser MISO | Multiuser MISO | DRL with actor-critic for joint optimization | Sum rate and control performance | Channel model is more abstract than Sionna RT | Best baseline paper for our first PyTorch DRL agent, especially if we start from DDPG or SAC-style continuous actions. |
| Puspitasari2023-RLSurvey | 2023 / Sensors | Survey RL algorithms for RIS | Multiple RIS-assisted scenarios | DQN, DDPG, TD3, PPO taxonomy | Comparative discussion, design caveats | Survey, not a new algorithm | Good reference for why we should keep both continuous-action and discrete-action experiment paths. |
| Bao2025-HDRL | 2025 / arXiv preprint, later conference metadata visible online | Large-action-space RIS phase optimization | Secure satellite RSMA scenario | Heuristic DRL with reduced action space | Secure sum rate, convergence efficiency | Different domain than terrestrial RT | Useful when scaling to large RIS grids where naive DRL action spaces become too hard to explore. |

## Method Clusters

### 1. Simulator and differentiable-RT foundation

- `Hoydis2025-SionnaRT`
- `Hoydis2024-LearningRadioEnv`
- `Hoydis2022-Sionna`

Takeaway:
Use Sionna RT as the environment engine, but do not assume every output should
remain in the differentiable graph. For DRL, most training will be black-box
interaction, so the practical win is physically grounded coverage/CIR synthesis,
not end-to-end backpropagation across TensorFlow and PyTorch.

### 2. Ray-tracing RIS coverage optimization baselines

- `Guneser2025-UrbanRIS`
- `Kilcioglu2025-IndoorRIS`

Takeaway:
Coverage maps and blind-spot metrics are natural environment outputs and can be
used both as evaluation figures and as components of the RL reward. These
papers also justify including a non-learning baseline that explicitly targets
low-power cells or coverage holes.

### 3. DRL policy design for RIS phase control

- `Huang2020-DDPGRIS`
- `Puspitasari2023-RLSurvey`
- `Bao2025-HDRL`

Takeaway:
Start from continuous control with a DDPG/SAC family agent. Keep a discrete or
factorized fallback path for large RIS arrays, because full joint phase control
scales poorly with the number of reflecting elements.

## Project Decisions Derived From The Papers

### `env_sionna.py`

- Build a TensorFlow/Sionna environment that returns physically meaningful
  observations such as coverage-map summaries, sampled receiver power, and
  optionally compressed CIR descriptors.
- Separate heavy scene generation from per-step control where possible.
- Support both `coverage_map`-style outputs and pointwise channel queries.
- Treat RIS placement and scene geometry as environment configuration, while RIS
  phase values are the control variables exposed to the agent.

### `agent_drl.py`

- Use actor-critic as the first family.
- Keep `DDPG` as the canonical literature baseline because Huang 2020 is the
  classic continuous-phase RIS DRL reference.
- Keep `SAC` as the likely mainline algorithm for our project because it is
  usually more stable than vanilla DDPG under noisy rewards.
- Reserve a future `HDRL` or action-pruning path for larger RIS arrays.

### `train_loop.py`

- Do not build a cross-framework autograd graph.
- Use TensorFlow only for environment-side simulation and export detached
  observations into NumPy, then into PyTorch tensors for the agent.
- Replay buffer entries should store compact state features, actions, rewards,
  next states, and done flags only.
- Avoid storing raw per-path tensors or full-resolution RT intermediates in the
  replay buffer.

### `utils_vis.py`

- Plot coverage maps before and after RIS optimization.
- Plot training curves: episode return, mean received power, blind-spot ratio,
  outage-like counts, and if applicable sum-rate proxies.
- Keep comparison figures that show random baseline, heuristic baseline, and
  DRL policy on the same scenario.

### `config.yaml`

- Split config into:
  - scene and carrier parameters
  - RIS geometry and phase resolution
  - observation design
  - reward weights
  - DRL hyperparameters
  - training schedule
  - memory and batching safeguards

## Recommended First Experiment Ladder

1. `Random phase` baseline.
2. `Heuristic coverage baseline` targeting low-power cells or target points.
3. `DDPG` baseline matched to Huang 2020 style continuous control.
4. `SAC` main experiment.
5. `HDRL` or action-space reduction only if RIS size makes 3 and 4 unstable.

## State, Action, Reward Guidance

### State

- Preferred first version:
  - Tx/Rx/RIS geometry summary
  - current RIS phase summary
  - sampled receiver powers
  - blind-spot ratio from a coarse coverage map
  - optional compressed CIR features
- Avoid full dense coverage tensors as the first RL state because they will
  inflate memory and slow replay.

### Action

- First version: continuous phase vector in `[-pi, pi]` or normalized `[-1, 1]`
  mapped to phase.
- If array size becomes large, switch to:
  - grouped phase control
  - codebook-based actions
  - heuristic candidate pruning

### Reward

- Main term: average received power or sum-rate proxy improvement.
- Coverage term: reduce the number of low-power cells or increase coverage
  ratio above a threshold.
- Regularization term: penalize abrupt phase changes if training oscillates.
- Optional cost term: penalize overly large active control dimensionality in
  grouped-control experiments.

## Engineering Guardrails For This Repo

- On this machine, the validated stable runtime is `Sionna/TF on CPU` and
  `PyTorch on GPU`. This avoids PTX JIT with `TensorFlow 2.15.x` on an
  `RTX 5090`.
- If TensorFlow GPU execution is re-enabled in a future software stack, every
  executable entry must apply hard TensorFlow GPU memory isolation first:
  `memory_limit = 1024 * 14`.
- PyTorch should own the remaining VRAM whenever mixed GPU execution is used.
- Start with batch size `1` on the Sionna side and increase only after memory
  profiling.
- Split sample budgets by task:
  - path solves
  - coverage-map solves
  - visibility probes
- Prefer coarse grids or sampled user points before full dense maps.
- Precompute static scene assets once; do not rebuild the RT scene every RL step.
- Keep TensorFlow outputs in `float32` and transfer only compact arrays to
  PyTorch.

## Missing-Source Caveats

- The exact title `Sionna: An Open-Source Library for Next-Generation Physical
  Layer Research` was verified as a 2022 arXiv white paper on NVIDIA Research.
  I did not verify a journal publication with that exact title.
- `Hoydis2025-SionnaRT` is a technical report and should be cited as such.
- `Bao2025-HDRL` is useful for large-action-space ideas, but its secure
  satellite RSMA scenario is not a direct match to our terrestrial RIS setting.

## Next Action

Use this blueprint to implement:

1. TensorFlow memory isolation and environment bootstrap in `env_sionna.py`.
2. A minimal `DDPG` plus optional `SAC` scaffold in `agent_drl.py`.
3. A detached TensorFlow-to-NumPy-to-PyTorch bridge in `train_loop.py`.

## Verified Source Links

- Sionna RT technical report:
  https://research.nvidia.com/publication/2025-04_sionna-rt-technical-report
- Learning Radio Environments by Differentiable Ray Tracing:
  https://research.nvidia.com/publication/2024-10_learning-radio-environments-differentiable-ray-tracing
- Sionna white paper:
  https://research.nvidia.com/publication/2022-03_sionna-open-source-library-next-generation-physical-layer-research
- Urban Sionna RT RIS paper:
  https://research.itu.edu.tr/en/publications/ris-optimization-algorithms-for-urban-wireless-scenarios-in-sionn/
- Indoor RIS optimization paper:
  https://ieeexplore.ieee.org/document/10999325/
- Huang 2020 RIS DRL paper:
  https://ieeexplore.ieee.org/document/9110869/
- RL for RIS survey:
  https://www.mdpi.com/1424-8220/23/5/2554
- HDRL large-action-space paper:
  https://www.researchgate.net/publication/388317034_Heuristic_Deep_Reinforcement_Learning_for_Phase_Shift_Optimization_in_RIS-assisted_Secure_Satellite_Communication_Systems_with_RSMA
