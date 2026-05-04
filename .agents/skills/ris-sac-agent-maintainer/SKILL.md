---
name: ris-sac-agent-maintainer
description: Maintain and extend the grouped-action Soft Actor-Critic implementation for the RIS phase-shift optimization repository. Use when Codex needs to modify SACConfig, ReplayBuffer, prioritized replay, actor or critic architecture, grouped-to-physical RIS mapping, entropy tuning, action sampling, checkpoint loading or saving, or other agent-specific logic in agent_drl.py. Do not use for TensorFlow or Sionna runtime issues, environment geometry, coverage-map logic, experiment plotting, or run-directory analysis.
---

# RIS SAC Agent Maintainer

Modify the agent as an agent, not as a general repository refactor.

## Primary file

- `agent_drl.py`

## Core contract to preserve

- grouped policy action space remains separate from the physical RIS action space
- `select_action()` can still return environment-ready action output
- replay storage keeps grouped actions, not expanded 10000-D actions
- `update()` performs one SAC update and returns scalar diagnostics
- `save()` and `load()` preserve trainable state needed to resume training

## Design intent to respect

- grouped control keeps the optimization tractable
- replay lives on CPU, and only sampled mini-batches move to device
- state inputs may be tiny, so normalization at the model input matters

Do not erase these design choices casually.

## Standard workflow

### 1. Identify the layer being changed

Classify the task as one of:

- config surface
- replay behavior
- action mapping
- actor architecture
- critic architecture
- entropy and alpha behavior
- checkpoint compatibility

Keep the patch focused on that layer.

### 2. Preserve the grouped-action contract

Before editing, verify these dimensions stay coherent:

- grouped action dim = `grouped_rows * grouped_cols`
- physical action dim = `physical_rows * physical_cols`
- mapper expansion stays consistent with environment expectations

If one dimension changes, update all dependent logic explicitly.

### 3. Keep diagnostics usable

If you change `update()`, continue returning metrics that training code can log. Prefer adding keys over removing existing ones.

### 4. Protect checkpoint usability

If architecture changes are incompatible with old checkpoints, state that clearly in code comments or release notes. Avoid silent breakage.

## Safe edits

Good examples:

- changing hidden width or depth
- adding dropout only if justified and documented
- refining target entropy logic
- improving prioritized replay sampling or priority updates
- adding an alternate grouped mapper that still expands into the same physical shape
- exposing more config knobs through `SACConfig`

## Anti-patterns

Do not:

- store full expanded physical actions in replay unless the task explicitly requires it
- move environment shaping logic into the agent
- hard-code experiment-specific constants outside `SACConfig`
- remove input normalization without a strong reason
- change action range semantics without updating mapper and training loop together

## Validation checklist

After edits, verify at least conceptually or by test:

- actor output shape matches grouped action dim
- mapper output shape matches environment action dim
- replay can add and sample without shape errors
- `update()` runs one step without device mismatch
- checkpoint save and load still function for the edited model

## Output expectations

When you use this skill, produce:

1. a statement of which agent layer changed
2. the shape contract before and after
3. any checkpoint compatibility note
4. the smallest useful validation command or test
