"""Main training loop for the RIS SAC project."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from agent_drl import SACAgent, SACConfig
from env_sionna import SionnaRISEnv

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
        "rx_height_m": 1.5,
        "rx_jitter_xy_m": 2.0,
        "blind_spot_search_center": [0.0, 0.0, 1.5],
        "blind_spot_search_size": [400.0, 400.0],
        "blind_spot_cell_size": [8.0, 8.0],
        "num_blind_spot_candidates": 16,
        "require_native_ris": True,
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
        "target_entropy": None,
        "device": None,
    },
    "train": {
        "num_episodes": 100,
        "max_steps_per_episode": 20,
        "warmup_steps": 1024,
        "updates_per_step": 1,
        "checkpoint_every": 10,
        "deterministic_eval": False,
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
) -> tuple[np.ndarray, np.ndarray]:
    """Follow the required bridge: NumPy state -> Torch action -> NumPy env action."""
    if warmup:
        grouped_action_tensor = torch.empty(
            agent.config.grouped_action_dim,
            device=agent.device,
        ).uniform_(-1.0, 1.0)
    else:
        state_tensor = state_numpy_to_torch(state, agent.device)
        grouped_action_tensor = agent.sample_grouped_action_tensor(
            state_tensor,
            deterministic=False,
        )

    grouped_action_np = grouped_action_tensor.detach().cpu().numpy().astype(np.float32)
    env_action_np = (
        agent.mapper.grouped_to_env_action(grouped_action_np)
        .reshape(-1)
        .astype(np.float32)
    )
    return env_action_np, grouped_action_np


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


def train(config_path: str | Path = "config.yaml") -> Path:
    config = load_config(config_path)
    set_global_seed(int(config["seed"]))

    run_dir = setup_output_dir(config["logging"])
    logger = setup_logger(run_dir, config["logging"])
    writer = maybe_create_tensorboard_writer(run_dir)
    save_config_snapshot(config, run_dir)

    logger.info("Loading environment and agent from %s", Path(config_path))
    env = SionnaRISEnv(**config["env"])
    initial_state = env.reset()
    state_dim = int(np.asarray(initial_state).size)
    logger.info("Resolved state_dim=%d", state_dim)

    agent = SACAgent(SACConfig(state_dim=state_dim, **config["agent"]))
    logger.info("Using agent device=%s", agent.device)

    baselines = evaluate_baselines(env)
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

    train_cfg = config["train"]
    global_step = 0

    for episode in range(1, int(train_cfg["num_episodes"]) + 1):
        state = env.reset()
        episode_reward = 0.0
        loss_accumulator: dict[str, list[float]] = {
            "actor_loss": [],
            "critic_loss": [],
            "alpha_loss": [],
            "alpha": [],
            "mean_q": [],
            "mean_log_prob": [],
        }

        for step_idx in range(1, int(train_cfg["max_steps_per_episode"]) + 1):
            warmup = global_step < int(train_cfg["warmup_steps"])
            env_action, grouped_action = sample_action_for_env(agent, state, warmup=warmup)
            next_state, reward = env.step(env_action)
            done = step_idx >= int(train_cfg["max_steps_per_episode"])

            agent.store_transition(state, grouped_action, reward, next_state, done)

            if agent.ready():
                for _ in range(int(train_cfg["updates_per_step"])):
                    losses = agent.update()
                    for key, value in losses.items():
                        loss_accumulator[key].append(value)

            state = next_state
            episode_reward += float(reward)
            global_step += 1

        steps_this_episode = int(train_cfg["max_steps_per_episode"])
        summary = {
            "episode": episode,
            "global_step": global_step,
            "avg_reward": episode_reward / max(steps_this_episode, 1),
            "avg_actor_loss": _safe_mean(loss_accumulator["actor_loss"]),
            "avg_critic_loss": _safe_mean(loss_accumulator["critic_loss"]),
            "avg_alpha_loss": _safe_mean(loss_accumulator["alpha_loss"]),
            "avg_alpha": _safe_mean(loss_accumulator["alpha"]),
            "avg_mean_q": _safe_mean(loss_accumulator["mean_q"]),
            "avg_mean_log_prob": _safe_mean(loss_accumulator["mean_log_prob"]),
        }

        logger.info(
            "Episode %03d | avg_reward=%.6f | actor_loss=%s | critic_loss=%s",
            summary["episode"],
            summary["avg_reward"],
            _fmt_float(summary["avg_actor_loss"]),
            _fmt_float(summary["avg_critic_loss"]),
        )
        append_metrics_jsonl(metrics_path, {"type": "episode", **summary})
        append_metrics_csv(csv_path, summary)
        maybe_write_tensorboard(writer, summary, step=episode)

        checkpoint_every = int(train_cfg["checkpoint_every"])
        if checkpoint_every > 0 and episode % checkpoint_every == 0:
            checkpoint_path = run_dir / f"checkpoint_episode_{episode:04d}.pt"
            agent.save(str(checkpoint_path))

    if config["logging"].get("save_final_checkpoint", True):
        agent.save(str(run_dir / "final_agent.pt"))

    if writer is not None:
        writer.flush()
        writer.close()

    logger.info("Training run completed. Artifacts saved to %s", run_dir)
    return run_dir


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


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
