"""Main training loop for the RIS SAC project."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from agent_drl import SACAgent, SACConfig

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 7,
    "env": {
        "carrier_frequency_hz": 3.5e9,
        "bandwidth_hz": 1.0e7,
        "tx_power_dbm": 30.0,
        "noise_temperature_k": 290.0,
        "tf_device": "cpu",
        "tf_memory_limit_mb": 1024 * 14,
        "max_depth": 2,
        "path_num_samples": 20000,
        "coverage_num_samples": 100000,
        "probe_num_samples": 64,
        "state_num_paths": 64,
        "rx_height_m": 1.5,
        "rx_jitter_xy_m": 2.0,
        "blind_spot_search_center": [0.0, 0.0, 1.5],
        "blind_spot_search_size": [400.0, 400.0],
        "blind_spot_cell_size": [8.0, 8.0],
        "num_blind_spot_candidates": 16,
        "require_native_ris": True,
        "discrete_phase": False,
        "num_bits": 2,
        "mobility_enabled": False,
        "zone_spawn_mode": "all",
    },
    "agent": {
        "grouped_rows": 10,
        "grouped_cols": 10,
        "physical_rows": 100,
        "physical_cols": 100,
        "hidden_dim": 512,
        "gamma": 0.99,
        "tau": 0.005,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "alpha_lr": 3e-4,
        "init_alpha": 0.2,
        "batch_size": 128,
        "replay_capacity": 20000,
        "dual_replay": False,
        "hard_replay_capacity": None,
        "hard_replay_ratio": 0.25,
        "prioritized_replay": False,
        "replay_priority_alpha": 1.0,
        "replay_uniform_ratio": 0.0,
        "replay_priority_epsilon": 1e-3,
        "target_entropy": None,
        "target_entropy_scale": 1.0,
        "alpha_min": None,
        "alpha_max": None,
        "device": None,
    },
    "train": {
        "num_episodes": 100,
        "max_steps_per_episode": 20,
        "warmup_steps": 1024,
        "warmup_action_scale": 0.35,
        "policy_exploration_scale_start": 0.35,
        "policy_exploration_scale_end": 1.0,
        "policy_additive_noise_std_start": 0.05,
        "policy_additive_noise_std_end": 0.0,
        "policy_noise_decay_steps": 1024,
        "collection_deterministic_prob_start": 0.0,
        "collection_deterministic_prob_end": 0.0,
        "collection_deterministic_prob_decay_steps": 1024,
        "collection_deterministic_noise_std": 0.0,
        "late_collection_deterministic_prob_start_episode": None,
        "late_collection_deterministic_prob_value": None,
        "policy_delay": 1,
        "hard_actor_update_gap_threshold": None,
        "hard_actor_policy_delay": None,
        "hard_actor_min_observations": 1,
        "freeze_actor_after_episode": None,
        "rx_block_episodes": 1,
        "rx_block_episodes_start": None,
        "rx_block_episodes_end": None,
        "rx_block_transition_episode": None,
        "hard_rx_focus_enabled": False,
        "hard_rx_focus_start_episode": 1,
        "hard_rx_focus_end_episode": None,
        "hard_rx_focus_zone": "nlos",
        "hard_rx_focus_probability": 0.5,
        "hard_rx_focus_gap_power": 1.0,
        "hard_rx_focus_min_gap": 0.25,
        "hard_rx_focus_warmup_episodes": 0,
        "hard_rx_focus_bootstrap_min_observations": 0,
        "hard_rx_focus_priority_zone": None,
        "hard_rx_focus_priority_candidate": None,
        "hard_rx_focus_priority_start_episode": None,
        "hard_rx_focus_priority_end_episode": None,
        "hard_rx_focus_priority_min_observations": 0,
        "hard_rx_focus_priority_weight_boost": 1.0,
        "hard_rx_priority_scale": 0.0,
        "updates_per_step": 1,
        "checkpoint_every": 10,
        "deterministic_eval": False,
        "eval_every_episodes": 5,
        "eval_num_episodes": 2,
        "eval_zone_name": "all",
        "zone_aware_eval_enabled": False,
        "zone_aware_eval_zones": ["los", "nlos"],
        "zone_aware_eval_num_episodes": None,
        "save_best_eval_checkpoint": False,
        "best_eval_metric": "eval_avg_reward",
        "final_checkpoint_reeval_enabled": False,
        "final_checkpoint_reeval_num_episodes": None,
        "final_checkpoint_reeval_last_k_checkpoints": 0,
        "final_checkpoint_reeval_include_best_eval": True,
        "final_checkpoint_reeval_include_final": True,
        "final_checkpoint_reeval_metric": "eval_avg_reward",
        "reward_margin_weight": 0.0,
        "reward_margin_positive_weight": None,
        "reward_margin_negative_weight": None,
        "reward_margin_positive_weight_los": None,
        "reward_margin_negative_weight_los": None,
        "reward_margin_positive_weight_nlos": None,
        "reward_margin_negative_weight_nlos": None,
        "reward_baseline_key": "phase_gradient_reflector_rate",
        "hard_replay_route_mode": "duplicate",
        "hard_replay_step_gap_threshold": None,
        "hard_replay_block_gap_threshold": None,
        "hard_step_priority_scale": 0.0,
        "hard_block_priority_scale": 0.0,
        "hard_priority_max": 10.0,
    },
    "logging": {
        "output_dir": "runs",
        "experiment_name": "sac_ris",
        "log_file": "training.log",
        "metrics_file": "metrics.jsonl",
        "csv_file": "episode_metrics.csv",
        "save_final_checkpoint": True,
    },
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config and merge it onto project defaults."""
    config_path = Path(config_path)
    raw: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    return _merge_dict(DEFAULT_CONFIG, raw)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_output_dir(logging_cfg: dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(logging_cfg["output_dir"]) / f"{logging_cfg['experiment_name']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(run_dir: Path, logging_cfg: dict[str, Any]) -> logging.Logger:
    logger = logging.getLogger("train_loop")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(run_dir / logging_cfg["log_file"], encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def save_config_snapshot(config: dict[str, Any], run_dir: Path) -> None:
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def maybe_create_tensorboard_writer(run_dir: Path):
    if SummaryWriter is None:
        return None
    return SummaryWriter(log_dir=str(run_dir / "tensorboard"))


def evaluate_baselines(env: SionnaRISEnv) -> dict[str, float | None]:
    """Evaluate the requested baselines at the current RX position."""
    return {
        "no_ris_rate": float(env.evaluate_no_ris_rate()),
        "phase_gradient_reflector_rate": env.evaluate_phase_gradient_reflector_rate(),
    }


def evaluate_fixed_position_baselines(
    env: SionnaRISEnv,
    *,
    zone_name: str | None = None,
    candidate_index: int | None = None,
) -> dict[str, float | None]:
    """Evaluate no-RIS and reflector baselines on one exact RX realization."""
    snapshot = _snapshot_env_state(env)
    try:
        env.reset(
            reuse_rx_position=False,
            candidate_index=candidate_index,
            zone_name=zone_name,
        )
        return {
            "no_ris_rate": float(env.evaluate_no_ris_rate()),
            "phase_gradient_reflector_rate": env.evaluate_phase_gradient_reflector_rate(),
        }
    finally:
        _restore_env_state(env, snapshot)


def collect_formal_baselines(env: SionnaRISEnv) -> dict[str, Any]:
    """Collect exact-position baselines for all exposed zone candidate pools."""
    zone_candidates = getattr(env, "zone_candidates", {})
    payload: dict[str, Any] = {
        "overall": {
            "no_ris_rate_mean": None,
            "phase_gradient_reflector_rate_mean": None,
        },
        "zones": {},
    }
    all_no_ris: list[float] = []
    all_reflector: list[float] = []
    for zone_name, candidates in zone_candidates.items():
        rows: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(np.asarray(candidates, dtype=np.float32)):
            baselines = evaluate_fixed_position_baselines(
                env,
                zone_name=zone_name,
                candidate_index=int(candidate_index),
            )
            rows.append(
                {
                    "candidate_index": int(candidate_index),
                    "rx_x_m": float(candidate[0]),
                    "rx_y_m": float(candidate[1]),
                    "rx_z_m": float(candidate[2]),
                    "no_ris_rate": _maybe_float_value(baselines.get("no_ris_rate")),
                    "phase_gradient_reflector_rate": _maybe_float_value(
                        baselines.get("phase_gradient_reflector_rate")
                    ),
                }
            )
        zone_no_ris = [row["no_ris_rate"] for row in rows if row["no_ris_rate"] is not None]
        zone_reflector = [
            row["phase_gradient_reflector_rate"]
            for row in rows
            if row["phase_gradient_reflector_rate"] is not None
        ]
        all_no_ris.extend(zone_no_ris)
        all_reflector.extend(zone_reflector)
        payload["zones"][zone_name] = {
            "num_candidates": len(rows),
            "no_ris_rate_mean": None if not zone_no_ris else float(np.mean(zone_no_ris)),
            "phase_gradient_reflector_rate_mean": None
            if not zone_reflector
            else float(np.mean(zone_reflector)),
            "candidates": rows,
        }
    payload["overall"] = {
        "no_ris_rate_mean": None if not all_no_ris else float(np.mean(all_no_ris)),
        "phase_gradient_reflector_rate_mean": None
        if not all_reflector
        else float(np.mean(all_reflector)),
    }
    return payload


def state_numpy_to_torch(state: np.ndarray, device: torch.device) -> torch.Tensor:
    """Bridge NumPy environment states into a Torch tensor on the target device."""
    state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32))
    if device.type == "cuda":
        return state_tensor.cuda(non_blocking=True)
    return state_tensor.to(device)


def sample_action_for_env(
    agent: SACAgent,
    state: np.ndarray,
    warmup: bool,
    global_step: int,
    train_cfg: dict[str, Any],
    *,
    episode: int | None = None,
    deterministic: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Follow the required bridge: NumPy state -> Torch action -> NumPy env action."""
    action_info = {
        "collection_deterministic": 0.0,
        "collection_deterministic_prob": 0.0,
        "exploration_scale": 0.0,
        "additive_noise_std": 0.0,
    }
    if deterministic:
        state_tensor = state_numpy_to_torch(state, agent.device)
        grouped_action_tensor = agent.sample_grouped_action_tensor(
            state_tensor,
            deterministic=True,
        )
        action_info["collection_deterministic"] = 1.0
        action_info["collection_deterministic_prob"] = 1.0
    elif warmup:
        warmup_action_scale = float(train_cfg.get("warmup_action_scale", 1.0))
        grouped_action_tensor = torch.empty(
            agent.config.grouped_action_dim,
            device=agent.device,
        ).uniform_(-warmup_action_scale, warmup_action_scale)
        action_info["exploration_scale"] = warmup_action_scale
    else:
        state_tensor = state_numpy_to_torch(state, agent.device)
        schedule_step = max(global_step - int(train_cfg.get("warmup_steps", 0)), 0)
        noise_decay_steps = int(train_cfg.get("policy_noise_decay_steps", 1))
        exploration_scale = _linear_schedule(
            start=float(train_cfg.get("policy_exploration_scale_start", 1.0)),
            end=float(train_cfg.get("policy_exploration_scale_end", 1.0)),
            step=schedule_step,
            decay_steps=noise_decay_steps,
        )
        additive_noise_std = _linear_schedule(
            start=float(train_cfg.get("policy_additive_noise_std_start", 0.0)),
            end=float(train_cfg.get("policy_additive_noise_std_end", 0.0)),
            step=schedule_step,
            decay_steps=noise_decay_steps,
        )
        deterministic_prob = _linear_schedule(
            start=float(train_cfg.get("collection_deterministic_prob_start", 0.0)),
            end=float(train_cfg.get("collection_deterministic_prob_end", 0.0)),
            step=schedule_step,
            decay_steps=int(train_cfg.get("collection_deterministic_prob_decay_steps", 1)),
        )
        late_start_episode = train_cfg.get(
            "late_collection_deterministic_prob_start_episode",
            None,
        )
        late_value = train_cfg.get("late_collection_deterministic_prob_value", None)
        if late_start_episode is not None and late_value is not None:
            current_episode = 0 if episode is None else int(episode)
            if current_episode >= int(late_start_episode):
                deterministic_prob = float(late_value)
        deterministic_prob = float(np.clip(deterministic_prob, 0.0, 1.0))
        use_deterministic_collection = bool(np.random.random() < deterministic_prob)

        if use_deterministic_collection:
            grouped_action_tensor = agent.sample_grouped_action_tensor(
                state_tensor,
                deterministic=True,
            )
            deterministic_noise_std = float(
                train_cfg.get("collection_deterministic_noise_std", 0.0)
            )
            if deterministic_noise_std > 0.0:
                grouped_action_tensor = (
                    grouped_action_tensor
                    + deterministic_noise_std * torch.randn_like(grouped_action_tensor)
                ).clamp(-1.0, 1.0)
                action_info["additive_noise_std"] = deterministic_noise_std
            action_info["collection_deterministic"] = 1.0
        else:
            grouped_action_tensor = agent.sample_grouped_action_tensor(
                state_tensor,
                deterministic=False,
                exploration_scale=exploration_scale,
                additive_noise_std=additive_noise_std,
            )
            action_info["exploration_scale"] = exploration_scale
            action_info["additive_noise_std"] = additive_noise_std

        action_info["collection_deterministic_prob"] = deterministic_prob

    grouped_action_np = grouped_action_tensor.detach().cpu().numpy().astype(np.float32)
    env_action_np = (
        agent.mapper.grouped_to_env_action(grouped_action_np)
        .reshape(-1)
        .astype(np.float32)
    )
    return env_action_np, grouped_action_np, action_info


def _linear_schedule(start: float, end: float, step: int, decay_steps: int) -> float:
    if decay_steps <= 0:
        return float(end)
    ratio = min(max(step, 0) / float(decay_steps), 1.0)
    return float(start + ratio * (end - start))


def _resolve_rx_block_episodes(
    train_cfg: dict[str, Any],
    *,
    episode: int,
    total_episodes: int,
) -> int:
    base_block = max(1, int(train_cfg.get("rx_block_episodes", 1)))
    start_value = train_cfg.get("rx_block_episodes_start", None)
    end_value = train_cfg.get("rx_block_episodes_end", None)

    if start_value is None and end_value is None:
        return base_block

    start_block = base_block if start_value is None else max(1, int(start_value))
    end_block = base_block if end_value is None else max(1, int(end_value))
    transition_episode = train_cfg.get("rx_block_transition_episode", None)
    if transition_episode is None:
        transition_episode = max(1, (int(total_episodes) // 2) + 1)
    transition_episode = max(1, int(transition_episode))

    return start_block if int(episode) < transition_episode else end_block


def initialize_hard_rx_stats(env) -> dict[str, dict[str, float]]:
    """Create per-zone candidate stats for hard-example receiver sampling."""
    hard_rx_stats: dict[str, dict[str, float]] = {}
    zone_candidates = getattr(env, "zone_candidates", {})
    if zone_candidates:
        for zone_name, candidates in zone_candidates.items():
            normalized_zone = _normalize_zone_name(zone_name)
            if normalized_zone is None:
                continue
            for candidate_index in range(len(candidates)):
                key = make_hard_rx_key(normalized_zone, candidate_index)
                hard_rx_stats[key] = {
                    "episodes": 0.0,
                    "avg_gap": 0.0,
                    "last_gap": 0.0,
                }
    if not hard_rx_stats:
        candidate_count = int(getattr(env, "num_available_blind_spot_candidates", 0))
        for candidate_index in range(candidate_count):
            key = make_hard_rx_key("nlos", candidate_index)
            hard_rx_stats[key] = {
                "episodes": 0.0,
                "avg_gap": 0.0,
                "last_gap": 0.0,
            }
    return hard_rx_stats


def make_hard_rx_key(zone_name: str, candidate_index: int) -> str:
    """Build a stable key for one `(zone, candidate)` receiver block."""
    return f"{zone_name}:{int(candidate_index)}"


def select_rx_candidate_index(
    env,
    *,
    train_cfg: dict[str, Any],
    episode: int,
    reuse_rx_position: bool,
    hard_rx_stats: dict[str, dict[str, float]],
) -> tuple[str | None, int | None, float, float]:
    """Optionally bias sampling toward blind-spot candidates with larger historical gaps."""
    if reuse_rx_position:
        return None, None, 0.0, 0.0
    if not bool(train_cfg.get("hard_rx_focus_enabled", False)):
        return None, None, 0.0, 0.0
    if int(episode) < int(train_cfg.get("hard_rx_focus_start_episode", 1)):
        return None, None, 0.0, 0.0
    end_episode = train_cfg.get("hard_rx_focus_end_episode", None)
    if end_episode is not None and int(episode) > int(end_episode):
        return None, None, 0.0, 0.0

    warmup_episodes = max(0, int(train_cfg.get("hard_rx_focus_warmup_episodes", 0)))
    zone_filter = _normalize_zone_name(train_cfg.get("hard_rx_focus_zone", "nlos"))
    eligible = [
        key
        for key, stats in hard_rx_stats.items()
        if stats["episodes"] >= warmup_episodes
        and (zone_filter is None or key.startswith(f"{zone_filter}:"))
    ]
    if not eligible:
        eligible = [
            key
            for key, stats in hard_rx_stats.items()
            if stats["episodes"] >= warmup_episodes
        ]
    if not eligible:
        return None, None, 0.0, 0.0

    eligible_keys = list(eligible)

    priority_zone = _normalize_zone_name(train_cfg.get("hard_rx_focus_priority_zone", None))
    priority_candidate = train_cfg.get("hard_rx_focus_priority_candidate", None)
    priority_key = None
    if priority_zone is not None and priority_candidate is not None:
        try:
            priority_key = make_hard_rx_key(priority_zone, int(priority_candidate))
        except (TypeError, ValueError):
            priority_key = None
        if priority_key not in eligible_keys:
            priority_key = None

    priority_start_episode = train_cfg.get("hard_rx_focus_priority_start_episode", None)
    priority_end_episode = train_cfg.get("hard_rx_focus_priority_end_episode", None)
    priority_active = priority_key is not None
    if priority_active and priority_start_episode is not None:
        priority_active = int(episode) >= int(priority_start_episode)
    if priority_active and priority_end_episode is not None:
        priority_active = int(episode) <= int(priority_end_episode)
    if not priority_active:
        priority_key = None

    priority_min_observations = max(
        0,
        int(train_cfg.get("hard_rx_focus_priority_min_observations", 0)),
    )
    if (
        priority_key is not None
        and priority_min_observations > 0
        and float(hard_rx_stats[priority_key]["episodes"]) < float(priority_min_observations)
    ):
        priority_zone_name, priority_index_str = priority_key.split(":", maxsplit=1)
        return (
            priority_zone_name,
            int(priority_index_str),
            1.0,
            float(hard_rx_stats[priority_key]["avg_gap"]),
        )

    bootstrap_min_observations = max(
        0,
        int(train_cfg.get("hard_rx_focus_bootstrap_min_observations", 0)),
    )
    if bootstrap_min_observations > 0:
        bootstrap_keys = [
            key
            for key in eligible_keys
            if float(hard_rx_stats[key]["episodes"]) < float(bootstrap_min_observations)
        ]
        if bootstrap_keys:
            min_observations = min(float(hard_rx_stats[key]["episodes"]) for key in bootstrap_keys)
            least_seen_keys = [
                key
                for key in bootstrap_keys
                if float(hard_rx_stats[key]["episodes"]) == min_observations
            ]
            chosen_key = str(np.random.choice(least_seen_keys))
            chosen_zone, chosen_index_str = chosen_key.split(":", maxsplit=1)
            chosen_index = int(chosen_index_str)
            return (
                chosen_zone,
                chosen_index,
                1.0 / float(len(least_seen_keys)),
                float(hard_rx_stats[chosen_key]["avg_gap"]),
            )

    focus_probability = float(np.clip(train_cfg.get("hard_rx_focus_probability", 0.5), 0.0, 1.0))
    if np.random.random() >= focus_probability:
        return None, None, 0.0, 0.0

    min_gap = float(max(train_cfg.get("hard_rx_focus_min_gap", 0.0), 0.0))
    gap_power = float(max(train_cfg.get("hard_rx_focus_gap_power", 1.0), 1e-6))
    weights = []
    priority_weight_boost = float(
        max(train_cfg.get("hard_rx_focus_priority_weight_boost", 1.0), 1e-6)
    )
    for key in eligible_keys:
        avg_gap = max(float(hard_rx_stats[key]["avg_gap"]), 0.0)
        weight = max(avg_gap, min_gap) ** gap_power
        if priority_key is not None and key == priority_key:
            weight *= priority_weight_boost
        weights.append(weight)
    weights_np = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights_np).all() or float(np.sum(weights_np)) <= 0.0:
        return None, None, 0.0, 0.0
    probs = weights_np / float(np.sum(weights_np))
    chosen_offset = int(np.random.choice(len(eligible_keys), p=probs))
    chosen_key = str(eligible_keys[chosen_offset])
    chosen_zone, chosen_index_str = chosen_key.split(":", maxsplit=1)
    chosen_index = int(chosen_index_str)
    return (
        chosen_zone,
        chosen_index,
        float(probs[chosen_offset]),
        float(hard_rx_stats[chosen_key]["avg_gap"]),
    )


def update_hard_rx_stats(
    hard_rx_stats: dict[str, dict[str, float]],
    *,
    zone_name: str | None,
    candidate_index: int | None,
    episode_gap: float | None,
) -> None:
    """Update per-candidate average hard gap after one episode block."""
    normalized_zone = _normalize_zone_name(zone_name)
    if normalized_zone is None or candidate_index is None or episode_gap is None:
        return
    stats_key = make_hard_rx_key(normalized_zone, candidate_index)
    if stats_key not in hard_rx_stats:
        return
    gap = max(float(episode_gap), 0.0)
    stats = hard_rx_stats[stats_key]
    episodes = int(stats["episodes"])
    new_count = episodes + 1
    stats["avg_gap"] = (float(stats["avg_gap"]) * episodes + gap) / float(new_count)
    stats["episodes"] = float(new_count)
    stats["last_gap"] = gap


def should_freeze_actor_for_episode(
    train_cfg: dict[str, Any],
    *,
    episode: int,
) -> bool:
    """Return whether actor updates should be disabled for the current episode."""
    freeze_after_episode = train_cfg.get("freeze_actor_after_episode", None)
    if freeze_after_episode is None:
        return False
    return int(episode) >= int(freeze_after_episode)


def _resolve_reward_baseline(
    baselines: dict[str, float | None],
    train_cfg: dict[str, Any],
) -> float | None:
    baseline_key = str(train_cfg.get("reward_baseline_key", "phase_gradient_reflector_rate"))
    baseline = baselines.get(baseline_key)
    if baseline is None:
        baseline = baselines.get("no_ris_rate")
    if baseline is None:
        return None
    return float(baseline)


def _normalize_zone_name(zone_name: str | None) -> str | None:
    if zone_name is None:
        return None
    normalized = str(zone_name).strip().lower()
    if normalized in {"los", "nlos"}:
        return normalized
    return None


def _resolve_zone_margin_weight(
    train_cfg: dict[str, Any],
    *,
    margin_positive: bool,
    current_zone: str | None,
    default_weight: float,
) -> float:
    sign_key = "positive" if margin_positive else "negative"
    normalized_zone = _normalize_zone_name(current_zone)
    if normalized_zone is not None:
        zone_key = f"reward_margin_{sign_key}_weight_{normalized_zone}"
        zone_weight = train_cfg.get(zone_key, None)
        if zone_weight is not None:
            return float(zone_weight)

    generic_key = f"reward_margin_{sign_key}_weight"
    generic_weight = train_cfg.get(generic_key, None)
    if generic_weight is not None:
        return float(generic_weight)
    return float(default_weight)


def shape_reward(
    raw_reward: float,
    *,
    baselines: dict[str, float | None],
    train_cfg: dict[str, Any],
    current_zone: str | None = None,
) -> tuple[float, float | None]:
    reward_margin_weight = float(train_cfg.get("reward_margin_weight", 0.0))
    baseline = _resolve_reward_baseline(baselines, train_cfg)
    if baseline is None:
        return float(raw_reward), None

    margin = float(raw_reward) - baseline
    positive_weight = _resolve_zone_margin_weight(
        train_cfg,
        margin_positive=True,
        current_zone=current_zone,
        default_weight=reward_margin_weight,
    )
    negative_weight = _resolve_zone_margin_weight(
        train_cfg,
        margin_positive=False,
        current_zone=current_zone,
        default_weight=reward_margin_weight,
    )

    if margin >= 0.0:
        shaped_reward = float(raw_reward) + positive_weight * margin
    else:
        shaped_reward = float(raw_reward) + negative_weight * margin
    return shaped_reward, margin


def compute_transition_priority(
    *,
    step_margin: float | None,
    episode_margin: float | None,
    train_cfg: dict[str, Any],
) -> float:
    """Build a replay priority from hard-step and hard-block baseline gaps."""
    step_gap = 0.0 if step_margin is None else max(-float(step_margin), 0.0)
    block_gap = 0.0 if episode_margin is None else max(-float(episode_margin), 0.0)
    priority = 1.0
    priority += float(train_cfg.get("hard_step_priority_scale", 0.0)) * step_gap
    priority += float(train_cfg.get("hard_block_priority_scale", 0.0)) * block_gap
    priority_max = float(train_cfg.get("hard_priority_max", 10.0))
    return float(np.clip(priority, 1e-6, priority_max))


def should_store_hard_transition(
    *,
    step_margin: float | None,
    episode_margin: float | None,
    train_cfg: dict[str, Any],
) -> bool:
    """Decide whether a transition should be duplicated into the hard replay pool."""
    step_gap = 0.0 if step_margin is None else max(-float(step_margin), 0.0)
    block_gap = 0.0 if episode_margin is None else max(-float(episode_margin), 0.0)
    step_threshold = train_cfg.get("hard_replay_step_gap_threshold", None)
    block_threshold = train_cfg.get("hard_replay_block_gap_threshold", None)

    step_hard = False if step_threshold is None else step_gap >= float(step_threshold)
    block_hard = False if block_threshold is None else block_gap >= float(block_threshold)
    return bool(step_hard or block_hard)


def resolve_hard_replay_route_mode(train_cfg: dict[str, Any]) -> str:
    """Normalize the routing mode used by dual replay."""
    route_mode = str(train_cfg.get("hard_replay_route_mode", "duplicate")).strip().lower()
    if route_mode not in {"duplicate", "exclusive"}:
        raise ValueError(
            "`train.hard_replay_route_mode` must be either 'duplicate' or 'exclusive'."
        )
    return route_mode


def resolve_actor_update_gate(
    train_cfg: dict[str, Any],
    *,
    online_hard_gap: float,
    num_observations: int,
) -> tuple[int, bool]:
    """Resolve the effective actor-update delay for the current training context."""
    base_delay = max(1, int(train_cfg.get("policy_delay", 1)))
    hard_delay_value = train_cfg.get("hard_actor_policy_delay", None)
    gap_threshold = train_cfg.get("hard_actor_update_gap_threshold", None)
    min_observations = max(1, int(train_cfg.get("hard_actor_min_observations", 1)))

    if hard_delay_value is None or gap_threshold is None:
        return base_delay, False
    if num_observations < min_observations:
        return base_delay, False

    hard_delay = max(base_delay, int(hard_delay_value))
    gate_active = float(online_hard_gap) >= float(gap_threshold)
    if not gate_active:
        return base_delay, False
    return hard_delay, True


def maybe_evaluate_episode_baselines(
    env,
    *,
    train_cfg: dict[str, Any],
    fallback_baselines: dict[str, float | None],
) -> dict[str, float | None]:
    """Resolve the reward baseline for the current RX realization."""
    try:
        episode_baselines = evaluate_baselines(env)
    except Exception:
        return dict(fallback_baselines)

    merged = dict(fallback_baselines)
    for key, value in episode_baselines.items():
        if value is not None:
            merged[key] = float(value)
    return merged


def _snapshot_env_state(env) -> dict[str, Any]:
    """Capture mutable environment state so evaluation does not perturb training."""
    snapshot: dict[str, Any] = {
        "last_state": None
        if getattr(env, "last_state", None) is None
        else np.array(env.last_state, dtype=np.float32, copy=True),
        "last_reward": getattr(env, "last_reward", None),
    }

    if hasattr(env, "rx"):
        snapshot["rx_position"] = np.asarray(env.rx.position, dtype=np.float32).tolist()
    if getattr(env, "ris", None) is not None and hasattr(env, "_snapshot_ris_state"):
        snapshot["ris_state"] = env._snapshot_ris_state()
    rng = getattr(env, "rng", None)
    if rng is not None and hasattr(rng, "bit_generator"):
        snapshot["rng_state"] = copy.deepcopy(rng.bit_generator.state)
    return snapshot


def _restore_env_state(env, snapshot: dict[str, Any]) -> None:
    """Restore environment state after deterministic evaluation."""
    if "rx_position" in snapshot and hasattr(env, "rx"):
        env.rx.position = snapshot["rx_position"]
    if "ris_state" in snapshot and getattr(env, "ris", None) is not None:
        env._restore_ris_state(snapshot["ris_state"])
    if "rng_state" in snapshot:
        env.rng.bit_generator.state = snapshot["rng_state"]
    env.last_state = snapshot.get("last_state")
    env.last_reward = snapshot.get("last_reward")


def _run_eval_rollouts(
    *,
    env,
    agent: SACAgent,
    train_cfg: dict[str, Any],
    baselines: dict[str, float | None],
    eval_zone_name: str,
    num_episodes: int,
) -> dict[str, float | None]:
    max_steps = int(train_cfg["max_steps_per_episode"])
    raw_episode_rewards: list[float] = []
    shaped_episode_rewards: list[float] = []
    reward_margins: list[float] = []

    for _ in range(max(1, int(num_episodes))):
        state = env.reset(zone_name=eval_zone_name)
        episode_baselines = maybe_evaluate_episode_baselines(
            env,
            train_cfg=train_cfg,
            fallback_baselines=baselines,
        )
        current_zone = _normalize_zone_name(getattr(env, "current_rx_zone", eval_zone_name))

        episode_raw_reward = 0.0
        episode_shaped_reward = 0.0
        for _ in range(max_steps):
            env_action, _, _ = sample_action_for_env(
                agent,
                state,
                warmup=False,
                global_step=0,
                train_cfg=train_cfg,
                deterministic=True,
            )
            next_state, raw_reward = env.step(env_action)
            shaped_reward, _ = shape_reward(
                float(raw_reward),
                baselines=episode_baselines,
                train_cfg=train_cfg,
                current_zone=current_zone,
            )
            state = next_state
            episode_raw_reward += float(raw_reward)
            episode_shaped_reward += float(shaped_reward)

        episode_avg_reward = episode_raw_reward / max(max_steps, 1)
        episode_avg_shaped_reward = episode_shaped_reward / max(max_steps, 1)
        raw_episode_rewards.append(episode_avg_reward)
        shaped_episode_rewards.append(episode_avg_shaped_reward)

        baseline_reference = _resolve_reward_baseline(episode_baselines, train_cfg)
        if baseline_reference is not None:
            reward_margins.append(episode_avg_reward - baseline_reference)

    return {
        "avg_reward": _safe_mean(raw_episode_rewards),
        "avg_shaped_reward": _safe_mean(shaped_episode_rewards),
        "avg_reward_margin": _safe_mean(reward_margins),
    }


def run_deterministic_eval(
    *,
    env,
    agent: SACAgent,
    train_cfg: dict[str, Any],
    baselines: dict[str, float | None],
) -> dict[str, float]:
    eval_num_episodes = max(1, int(train_cfg.get("eval_num_episodes", 1)))
    eval_zone_name = str(train_cfg.get("eval_zone_name", "all")).strip().lower()

    snapshot = _snapshot_env_state(env)
    try:
        global_eval = _run_eval_rollouts(
            env=env,
            agent=agent,
            train_cfg=train_cfg,
            baselines=baselines,
            eval_zone_name=eval_zone_name,
            num_episodes=eval_num_episodes,
        )
        result = {
            "eval_avg_reward": global_eval["avg_reward"],
            "eval_avg_shaped_reward": global_eval["avg_shaped_reward"],
            "eval_avg_reward_margin": global_eval["avg_reward_margin"],
        }
        if bool(train_cfg.get("zone_aware_eval_enabled", False)):
            zone_names = train_cfg.get("zone_aware_eval_zones", ["los", "nlos"])
            zone_eval_num_episodes = train_cfg.get("zone_aware_eval_num_episodes", None)
            zone_eval_num_episodes = (
                eval_num_episodes
                if zone_eval_num_episodes is None
                else max(1, int(zone_eval_num_episodes))
            )
            for zone_name in zone_names:
                normalized_zone = _normalize_zone_name(zone_name)
                if normalized_zone is None:
                    continue
                zone_eval = _run_eval_rollouts(
                    env=env,
                    agent=agent,
                    train_cfg=train_cfg,
                    baselines=baselines,
                    eval_zone_name=normalized_zone,
                    num_episodes=zone_eval_num_episodes,
                )
                result[f"eval_{normalized_zone}_avg_reward"] = zone_eval["avg_reward"]
                result[f"eval_{normalized_zone}_avg_shaped_reward"] = zone_eval[
                    "avg_shaped_reward"
                ]
                result[f"eval_{normalized_zone}_avg_reward_margin"] = zone_eval[
                    "avg_reward_margin"
                ]
        return result
    finally:
        _restore_env_state(env, snapshot)


def append_metrics_jsonl(metrics_path: Path, payload: dict[str, Any]) -> None:
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_metrics_csv(csv_path: Path, payload: dict[str, Any]) -> None:
    fieldnames = list(payload.keys())
    file_exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(payload)


def maybe_write_tensorboard(writer, payload: dict[str, Any], step: int) -> None:
    if writer is None:
        return
    for key, value in payload.items():
        if isinstance(value, (int, float)) and value is not None:
            writer.add_scalar(key, value, global_step=step)


def _record_candidate_checkpoint(
    records: list[dict[str, Any]],
    *,
    path: Path,
    source: str,
    episode: int | None,
) -> None:
    records.append(
        {
            "path": str(path),
            "source": str(source),
            "episode": None if episode is None else int(episode),
        }
    )


def _dedupe_candidate_checkpoints(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        path = str(record["path"])
        if path in seen:
            continue
        seen.add(path)
        deduped.append(record)
    return deduped


def _copy_checkpoint_file(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    destination.write_bytes(source.read_bytes())


def train(config_path: str | Path = "config.yaml") -> Path:
    config = load_config(config_path)

    env_cfg = dict(config["env"])
    env_cfg.setdefault("rng_seed", int(config["seed"]))
    sionna_tf_device = str(env_cfg.get("tf_device", "cpu")).strip().lower()
    os.environ["SIONNA_TF_DEVICE"] = sionna_tf_device
    previous_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    from env_sionna import SionnaRISEnv
    if sionna_tf_device == "cpu":
        if previous_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda_visible_devices

    set_global_seed(int(config["seed"]))

    run_dir = setup_output_dir(config["logging"])
    logger = setup_logger(run_dir, config["logging"])
    writer = maybe_create_tensorboard_writer(run_dir)
    save_config_snapshot(config, run_dir)

    logger.info("Loading environment and agent from %s", Path(config_path))
    env = SionnaRISEnv(**env_cfg)
    initial_state = env.reset()
    state_dim = int(np.asarray(initial_state).size)
    logger.info("Resolved state_dim=%d", state_dim)

    agent = SACAgent(SACConfig(state_dim=state_dim, **config["agent"]))
    logger.info("Using agent device=%s", agent.device)

    baselines = evaluate_baselines(env)
    formal_baselines = collect_formal_baselines(env)
    logger.info(
        "Baselines | no_ris_rate=%.6f | phase_gradient_reflector_rate=%s",
        baselines["no_ris_rate"],
        "unavailable"
        if baselines["phase_gradient_reflector_rate"] is None
        else f"{baselines['phase_gradient_reflector_rate']:.6f}",
    )

    metrics_path = run_dir / config["logging"]["metrics_file"]
    csv_path = run_dir / config["logging"]["csv_file"]

    append_metrics_jsonl(
        metrics_path,
        {
            "type": "baseline",
            "timestamp": datetime.now().isoformat(),
            **baselines,
        },
    )
    append_metrics_jsonl(
        metrics_path,
        {
            "type": "baseline_fixed_positions",
            "timestamp": datetime.now().isoformat(),
            **formal_baselines,
        },
    )

    train_cfg = config["train"]
    dual_replay_enabled = bool(config["agent"].get("dual_replay", False))
    hard_replay_route_mode = resolve_hard_replay_route_mode(train_cfg)
    hard_rx_stats = initialize_hard_rx_stats(env)
    global_step = 0
    gradient_step = 0
    actor_update_wait = 0
    total_episodes = int(train_cfg["num_episodes"])
    best_eval_metric_name = str(train_cfg.get("best_eval_metric", "eval_avg_reward"))
    best_eval_metric_value: float | None = None
    checkpoint_records: list[dict[str, Any]] = []
    best_eval_checkpoint_path: Path | None = None
    final_checkpoint_path: Path | None = None
    recommended_checkpoint_path: Path | None = None
    recommended_checkpoint_source: str | None = None

    for episode in range(1, total_episodes + 1):
        rx_block_episodes = _resolve_rx_block_episodes(
            train_cfg,
            episode=episode,
            total_episodes=total_episodes,
        )
        reuse_rx_position = episode > 1 and ((episode - 1) % rx_block_episodes != 0)
        selected_rx_zone, selected_rx_candidate_index, hard_rx_focus_prob, selected_rx_avg_gap = (
            select_rx_candidate_index(
                env,
                train_cfg=train_cfg,
                episode=episode,
                reuse_rx_position=reuse_rx_position,
                hard_rx_stats=hard_rx_stats,
            )
        )
        state = env.reset(
            reuse_rx_position=reuse_rx_position,
            candidate_index=selected_rx_candidate_index,
            zone_name=selected_rx_zone,
        )
        active_rx_candidate_index = getattr(env, "current_rx_candidate_index", None)
        active_rx_zone = getattr(env, "current_rx_zone", None)
        episode_baselines = maybe_evaluate_episode_baselines(
            env,
            train_cfg=train_cfg,
            fallback_baselines=baselines,
        )
        episode_baseline_reference = _resolve_reward_baseline(episode_baselines, train_cfg)
        episode_reward = 0.0
        episode_shaped_reward = 0.0
        hard_replay_added = 0
        episode_transition_records: list[tuple[int, float | None]] = []
        episode_transition_payloads: list[dict[str, Any]] = []
        episode_hard_gap_sum = 0.0
        episode_hard_gap_count = 0
        loss_accumulator: dict[str, list[float]] = {
            "actor_loss": [],
            "critic_loss": [],
            "alpha_loss": [],
            "alpha": [],
            "mean_q": [],
            "mean_log_prob": [],
            "actor_updated": [],
            "collection_deterministic": [],
            "collection_deterministic_prob": [],
            "exploration_scale": [],
            "action_noise_std": [],
            "effective_policy_delay": [],
            "actor_gate_active": [],
            "online_hard_step_gap": [],
        }

        for step_idx in range(1, int(train_cfg["max_steps_per_episode"]) + 1):
            warmup = global_step < int(train_cfg["warmup_steps"])
            env_action, grouped_action, action_info = sample_action_for_env(
                agent,
                state,
                warmup=warmup,
                global_step=global_step,
                train_cfg=train_cfg,
                episode=episode,
            )
            next_state, raw_reward = env.step(env_action)
            done = step_idx >= int(train_cfg["max_steps_per_episode"])
            shaped_reward, reward_margin = shape_reward(
                float(raw_reward),
                baselines=episode_baselines,
                train_cfg=train_cfg,
                current_zone=active_rx_zone,
            )
            hard_step_gap = 0.0 if reward_margin is None else max(-float(reward_margin), 0.0)
            episode_hard_gap_sum += hard_step_gap
            episode_hard_gap_count += 1
            online_hard_gap = episode_hard_gap_sum / max(episode_hard_gap_count, 1)

            transition_priority = compute_transition_priority(
                step_margin=reward_margin,
                episode_margin=None,
                train_cfg=train_cfg,
            )
            hard_rx_priority_scale = float(train_cfg.get("hard_rx_priority_scale", 0.0))
            if hard_rx_priority_scale > 0.0:
                transition_priority += hard_rx_priority_scale * max(selected_rx_avg_gap, 0.0)
                transition_priority = float(
                    np.clip(
                        transition_priority,
                        1e-6,
                        float(train_cfg.get("hard_priority_max", 10.0)),
                    )
                )
            episode_transition_payloads.append(
                {
                    "state": np.asarray(state, dtype=np.float32).copy(),
                    "grouped_action": np.asarray(grouped_action, dtype=np.float32).copy(),
                    "reward": float(shaped_reward),
                    "next_state": np.asarray(next_state, dtype=np.float32).copy(),
                    "done": bool(done),
                    "step_margin": reward_margin,
                }
            )
            if not (dual_replay_enabled and hard_replay_route_mode == "exclusive"):
                replay_index = agent.store_transition(
                    state,
                    grouped_action,
                    shaped_reward,
                    next_state,
                    done,
                    priority=transition_priority,
                )
                episode_transition_records.append((replay_index, reward_margin))

            if agent.ready():
                for _ in range(int(train_cfg["updates_per_step"])):
                    gradient_step += 1
                    actor_frozen = should_freeze_actor_for_episode(
                        train_cfg,
                        episode=episode,
                    )
                    effective_policy_delay, actor_gate_active = resolve_actor_update_gate(
                        train_cfg,
                        online_hard_gap=online_hard_gap,
                        num_observations=episode_hard_gap_count,
                    )
                    actor_update_wait += 1
                    update_actor = actor_update_wait >= effective_policy_delay
                    if update_actor:
                        actor_update_wait = 0
                    if actor_frozen:
                        update_actor = False
                    losses = agent.update(
                        update_actor=update_actor,
                        update_alpha=update_actor,
                        update_target=update_actor,
                    )
                    for key, value in losses.items():
                        if value is not None:
                            loss_accumulator[key].append(value)
                    loss_accumulator["effective_policy_delay"].append(
                        float(effective_policy_delay)
                    )
                    loss_accumulator["actor_gate_active"].append(float(actor_gate_active))
                    loss_accumulator["online_hard_step_gap"].append(float(online_hard_gap))

            state = next_state
            episode_reward += float(raw_reward)
            episode_shaped_reward += float(shaped_reward)
            loss_accumulator["collection_deterministic"].append(
                float(action_info["collection_deterministic"])
            )
            loss_accumulator["collection_deterministic_prob"].append(
                float(action_info["collection_deterministic_prob"])
            )
            loss_accumulator["exploration_scale"].append(float(action_info["exploration_scale"]))
            loss_accumulator["action_noise_std"].append(float(action_info["additive_noise_std"]))
            global_step += 1

        steps_this_episode = int(train_cfg["max_steps_per_episode"])
        episode_margin = None
        if episode_baseline_reference is not None:
            episode_margin = (episode_reward / max(steps_this_episode, 1)) - episode_baseline_reference
        episode_hard_gap = None if episode_margin is None else max(-episode_margin, 0.0)
        update_hard_rx_stats(
            hard_rx_stats,
            zone_name=active_rx_zone,
            candidate_index=active_rx_candidate_index,
            episode_gap=episode_hard_gap,
        )
        if dual_replay_enabled and episode_transition_payloads:
            for payload in episode_transition_payloads:
                is_hard = should_store_hard_transition(
                    step_margin=payload["step_margin"],
                    episode_margin=episode_margin,
                    train_cfg=train_cfg,
                )
                transition_priority = compute_transition_priority(
                    step_margin=payload["step_margin"],
                    episode_margin=episode_margin,
                    train_cfg=train_cfg,
                )
                if hard_replay_route_mode == "exclusive":
                    if is_hard:
                        agent.store_hard_transition(
                            payload["state"],
                            payload["grouped_action"],
                            payload["reward"],
                            payload["next_state"],
                            payload["done"],
                            priority=transition_priority,
                        )
                        hard_replay_added += 1
                    else:
                        agent.store_transition(
                            payload["state"],
                            payload["grouped_action"],
                            payload["reward"],
                            payload["next_state"],
                            payload["done"],
                            priority=transition_priority,
                        )
                elif is_hard:
                    agent.store_hard_transition(
                        payload["state"],
                        payload["grouped_action"],
                        payload["reward"],
                        payload["next_state"],
                        payload["done"],
                        priority=transition_priority,
                    )
                    hard_replay_added += 1
        elif episode_transition_records and (
            float(train_cfg.get("hard_step_priority_scale", 0.0)) > 0.0
            or float(train_cfg.get("hard_block_priority_scale", 0.0)) > 0.0
        ):
            replay_indices = [index for index, _ in episode_transition_records]
            replay_priorities = [
                compute_transition_priority(
                    step_margin=step_margin,
                    episode_margin=episode_margin,
                    train_cfg=train_cfg,
                )
                for _, step_margin in episode_transition_records
            ]
            agent.update_replay_priorities(replay_indices, replay_priorities)

        eval_summary = {
            "eval_avg_reward": None,
            "eval_avg_shaped_reward": None,
            "eval_avg_reward_margin": None,
        }
        eval_every_episodes = int(train_cfg.get("eval_every_episodes", 0))
        if train_cfg.get("deterministic_eval", False) and eval_every_episodes > 0:
            if episode % eval_every_episodes == 0:
                eval_summary = run_deterministic_eval(
                    env=env,
                    agent=agent,
                    train_cfg=train_cfg,
                    baselines=baselines,
                )

        avg_reward = episode_reward / max(steps_this_episode, 1)
        summary = {
            "episode": episode,
            "global_step": global_step,
            "avg_reward": avg_reward,
            "avg_shaped_reward": episode_shaped_reward / max(steps_this_episode, 1),
            "avg_reward_margin": None
            if episode_baseline_reference is None
            else avg_reward - episode_baseline_reference,
            "avg_hard_block_gap": episode_hard_gap,
            "hard_replay_added": hard_replay_added,
            "episode_no_ris_rate": episode_baselines.get("no_ris_rate"),
            "episode_phase_gradient_reflector_rate": episode_baselines.get(
                "phase_gradient_reflector_rate"
            ),
            "episode_reward_baseline": episode_baseline_reference,
            "rx_candidate_index": active_rx_candidate_index,
            "rx_zone": active_rx_zone,
            "rx_focus_selected": float(selected_rx_candidate_index is not None),
            "rx_focus_zone": selected_rx_zone,
            "rx_focus_selection_prob": hard_rx_focus_prob,
            "rx_focus_candidate_avg_gap": selected_rx_avg_gap,
            "rx_candidate_avg_gap_after_episode": None
            if active_rx_candidate_index is None or _normalize_zone_name(active_rx_zone) is None
            else float(
                hard_rx_stats[
                    make_hard_rx_key(_normalize_zone_name(active_rx_zone), active_rx_candidate_index)
                ]["avg_gap"]
            ),
            "avg_actor_loss": _safe_mean(loss_accumulator["actor_loss"]),
            "avg_critic_loss": _safe_mean(loss_accumulator["critic_loss"]),
            "avg_alpha_loss": _safe_mean(loss_accumulator["alpha_loss"]),
            "avg_alpha": _safe_mean(loss_accumulator["alpha"]),
            "avg_mean_q": _safe_mean(loss_accumulator["mean_q"]),
            "avg_mean_log_prob": _safe_mean(loss_accumulator["mean_log_prob"]),
            "avg_actor_updated": _safe_mean(loss_accumulator["actor_updated"]),
            "avg_collection_deterministic": _safe_mean(
                loss_accumulator["collection_deterministic"]
            ),
            "avg_collection_deterministic_prob": _safe_mean(
                loss_accumulator["collection_deterministic_prob"]
            ),
            "avg_exploration_scale": _safe_mean(loss_accumulator["exploration_scale"]),
            "avg_action_noise_std": _safe_mean(loss_accumulator["action_noise_std"]),
            "avg_effective_policy_delay": _safe_mean(
                loss_accumulator["effective_policy_delay"]
            ),
            "avg_actor_gate_active": _safe_mean(loss_accumulator["actor_gate_active"]),
            "avg_online_hard_step_gap": _safe_mean(loss_accumulator["online_hard_step_gap"]),
            "rx_block_episodes": rx_block_episodes,
            "rx_reused": float(reuse_rx_position),
            **eval_summary,
        }

        logger.info(
            "Episode %03d | avg_reward=%.6f | avg_shaped_reward=%.6f | rx_zone=%s | rx_idx=%s | rx_gap=%s | det_frac=%s | actor_upd=%s | critic_loss=%s | eval_reward=%s",
            summary["episode"],
            summary["avg_reward"],
            summary["avg_shaped_reward"],
            summary.get("rx_zone", "n/a"),
            "n/a" if summary["rx_candidate_index"] is None else int(summary["rx_candidate_index"]),
            _fmt_float(summary["rx_candidate_avg_gap_after_episode"]),
            _fmt_float(summary["avg_collection_deterministic"]),
            _fmt_float(summary["avg_actor_updated"]),
            _fmt_float(summary["avg_critic_loss"]),
            _fmt_float(summary["eval_avg_reward"]),
        )
        append_metrics_jsonl(metrics_path, {"type": "episode", **summary})
        append_metrics_csv(csv_path, summary)
        maybe_write_tensorboard(writer, summary, step=episode)

        if bool(train_cfg.get("save_best_eval_checkpoint", False)):
            candidate_metric = summary.get(best_eval_metric_name)
            if candidate_metric is not None:
                candidate_metric = float(candidate_metric)
                if best_eval_metric_value is None or candidate_metric > best_eval_metric_value:
                    best_eval_metric_value = candidate_metric
                    best_checkpoint_path = run_dir / "best_eval_agent.pt"
                    agent.save(str(best_checkpoint_path))
                    best_eval_checkpoint_path = best_checkpoint_path
                    _record_candidate_checkpoint(
                        checkpoint_records,
                        path=best_checkpoint_path,
                        source="best_eval",
                        episode=episode,
                    )
                    append_metrics_jsonl(
                        metrics_path,
                        {
                            "type": "best_eval_checkpoint",
                            "episode": episode,
                            "metric_name": best_eval_metric_name,
                            "metric_value": best_eval_metric_value,
                            "checkpoint_path": str(best_checkpoint_path),
                        },
                    )

        checkpoint_every = int(train_cfg["checkpoint_every"])
        if checkpoint_every > 0 and episode % checkpoint_every == 0:
            checkpoint_path = run_dir / f"checkpoint_episode_{episode:04d}.pt"
            agent.save(str(checkpoint_path))
            _record_candidate_checkpoint(
                checkpoint_records,
                path=checkpoint_path,
                source="periodic",
                episode=episode,
            )

    if config["logging"].get("save_final_checkpoint", True):
        final_checkpoint_path = run_dir / "final_agent.pt"
        agent.save(str(final_checkpoint_path))
        _record_candidate_checkpoint(
            checkpoint_records,
            path=final_checkpoint_path,
            source="final",
            episode=total_episodes,
        )

    if bool(train_cfg.get("final_checkpoint_reeval_enabled", False)):
        reevaluate_candidates = _dedupe_candidate_checkpoints(checkpoint_records)
        last_k_checkpoints = max(
            0,
            int(train_cfg.get("final_checkpoint_reeval_last_k_checkpoints", 0)),
        )
        include_best_eval = bool(train_cfg.get("final_checkpoint_reeval_include_best_eval", True))
        include_final = bool(train_cfg.get("final_checkpoint_reeval_include_final", True))

        filtered_candidates: list[dict[str, Any]] = []
        periodic_candidates: list[dict[str, Any]] = []
        for record in reevaluate_candidates:
            if record["source"] == "periodic":
                periodic_candidates.append(record)
                continue
            if record["source"] == "best_eval" and not include_best_eval:
                continue
            if record["source"] == "final" and not include_final:
                continue
            filtered_candidates.append(record)

        if last_k_checkpoints > 0 and periodic_candidates:
            periodic_candidates = sorted(
                periodic_candidates,
                key=lambda item: -1 if item["episode"] is None else int(item["episode"]),
            )
            filtered_candidates.extend(periodic_candidates[-last_k_checkpoints:])

        filtered_candidates = _dedupe_candidate_checkpoints(filtered_candidates)
        reevaluate_num_episodes = train_cfg.get("final_checkpoint_reeval_num_episodes", None)
        reevaluate_eval_episodes = (
            int(train_cfg.get("eval_num_episodes", 2))
            if reevaluate_num_episodes is None
            else int(reevaluate_num_episodes)
        )
        reevaluate_metric_name = str(
            train_cfg.get("final_checkpoint_reeval_metric", "eval_avg_reward")
        )

        original_eval_num_episodes = train_cfg.get("eval_num_episodes", 2)
        reevaluate_train_cfg = dict(train_cfg)
        reevaluate_train_cfg["eval_num_episodes"] = reevaluate_eval_episodes

        reevaluate_results: list[dict[str, Any]] = []
        if filtered_candidates:
            current_agent_snapshot = run_dir / "__reeval_restore_agent.pt"
            agent.save(str(current_agent_snapshot))
            try:
                for candidate in filtered_candidates:
                    candidate_path = Path(candidate["path"])
                    if not candidate_path.exists():
                        continue
                    agent.load(str(candidate_path))
                    eval_result = run_deterministic_eval(
                        env=env,
                        agent=agent,
                        train_cfg=reevaluate_train_cfg,
                        baselines=baselines,
                    )
                    metric_value = eval_result.get(reevaluate_metric_name)
                    result = {
                        "type": "final_checkpoint_reeval_candidate",
                        "checkpoint_path": str(candidate_path),
                        "checkpoint_source": candidate["source"],
                        "episode": candidate["episode"],
                        "reeval_num_episodes": reevaluate_eval_episodes,
                        **eval_result,
                        "reeval_metric_name": reevaluate_metric_name,
                        "reeval_metric_value": metric_value,
                    }
                    reevaluate_results.append(result)
                    append_metrics_jsonl(metrics_path, result)

                valid_results = [
                    item
                    for item in reevaluate_results
                    if item.get("reeval_metric_value") is not None
                ]
                if valid_results:
                    best_result = max(
                        valid_results,
                        key=lambda item: float(item["reeval_metric_value"]),
                    )
                    best_source_path = Path(best_result["checkpoint_path"])
                    reevaluate_best_path = run_dir / "best_final_reeval_agent.pt"
                    _copy_checkpoint_file(best_source_path, reevaluate_best_path)
                    recommended_checkpoint_path = reevaluate_best_path
                    recommended_checkpoint_source = "final_checkpoint_reeval"
                    append_metrics_jsonl(
                        metrics_path,
                        {
                            "type": "final_checkpoint_reeval_best",
                            "checkpoint_path": str(reevaluate_best_path),
                            "checkpoint_source": best_result["checkpoint_source"],
                            "episode": best_result["episode"],
                            "reeval_num_episodes": reevaluate_eval_episodes,
                            "reeval_metric_name": reevaluate_metric_name,
                            "reeval_metric_value": best_result["reeval_metric_value"],
                            "eval_avg_reward": best_result.get("eval_avg_reward"),
                            "eval_avg_shaped_reward": best_result.get("eval_avg_shaped_reward"),
                            "eval_avg_reward_margin": best_result.get("eval_avg_reward_margin"),
                        },
                    )
            finally:
                agent.load(str(current_agent_snapshot))
                if current_agent_snapshot.exists():
                    current_agent_snapshot.unlink()

    if recommended_checkpoint_path is None:
        if best_eval_checkpoint_path is not None and best_eval_checkpoint_path.exists():
            recommended_checkpoint_path = run_dir / "recommended_agent.pt"
            _copy_checkpoint_file(best_eval_checkpoint_path, recommended_checkpoint_path)
            recommended_checkpoint_source = "best_eval"
        elif final_checkpoint_path is not None and final_checkpoint_path.exists():
            recommended_checkpoint_path = run_dir / "recommended_agent.pt"
            _copy_checkpoint_file(final_checkpoint_path, recommended_checkpoint_path)
            recommended_checkpoint_source = "final"
    else:
        recommended_copy_path = run_dir / "recommended_agent.pt"
        _copy_checkpoint_file(recommended_checkpoint_path, recommended_copy_path)
        recommended_checkpoint_path = recommended_copy_path

    if recommended_checkpoint_path is not None and recommended_checkpoint_source is not None:
        append_metrics_jsonl(
            metrics_path,
            {
                "type": "recommended_checkpoint",
                "checkpoint_path": str(recommended_checkpoint_path),
                "checkpoint_source": recommended_checkpoint_source,
            },
        )

    if writer is not None:
        writer.flush()
        writer.close()

    logger.info("Training run completed. Artifacts saved to %s", run_dir)
    return run_dir


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _maybe_float_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC for RIS phase optimization.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
