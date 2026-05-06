#!/usr/bin/env python3
"""Formal fixed-zone evaluation for a recommended RIS SAC checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

from utils_vis import plot_academic_learning_curve, plot_coverage_comparison, plot_phase_profile


sns.set_theme(style="whitegrid", context="talk")

EPS = 1e-12
ZONE_LOS = "los"
ZONE_NLOS = "nlos"


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
        "device": None,
    },
    "train": {
        "num_episodes": 100,
        "max_steps_per_episode": 20,
    },
    "logging": {
        "output_dir": "runs",
        "experiment_name": "sac_ris",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a stricter post-training evaluation on fixed LOS/NLOS candidate sets "
            "for a recommended SAC checkpoint."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory that contains resolved_config.yaml and recommended_agent.pt.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config path. Defaults to <run-dir>/resolved_config.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path. Defaults to <run-dir>/recommended_agent.pt.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="formal_fixed_zone_eval",
        help="Output subdirectory created under --run-dir.",
    )
    parser.add_argument(
        "--num-rollouts-per-candidate",
        type=int,
        default=4,
        help="Repeated deterministic rollouts per fixed RX candidate.",
    )
    parser.add_argument(
        "--representative-zone",
        type=str,
        default=ZONE_NLOS,
        choices=[ZONE_LOS, ZONE_NLOS],
        help="Zone used for representative phase/coverage plots.",
    )
    parser.add_argument(
        "--representative-candidate-index",
        type=int,
        default=None,
        help=(
            "Optional representative candidate index inside the representative zone. "
            "Defaults to the candidate whose DRL average rate is closest to the zone mean."
        ),
    )
    return parser.parse_args()


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    raw: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    return _merge_dict(DEFAULT_CONFIG, raw)


def set_global_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)


def _instantiate_env_and_agent(
    config: dict[str, Any],
    checkpoint_path: Path,
):
    env_cfg = dict(config["env"])
    env_cfg["rx_jitter_xy_m"] = 0.0
    env_cfg.setdefault("rng_seed", int(config["seed"]))

    os.environ["SIONNA_TF_DEVICE"] = str(env_cfg.get("tf_device", "cpu")).strip().lower()

    from env_sionna import SionnaRISEnv
    from agent_drl import SACAgent, SACConfig

    env = SionnaRISEnv(**env_cfg)
    initial_zone = ZONE_NLOS if len(env.zone_candidates.get(ZONE_NLOS, [])) > 0 else ZONE_LOS
    initial_state = env.reset(zone_name=initial_zone, candidate_index=0)
    state_dim = int(np.asarray(initial_state).size)

    agent = SACAgent(SACConfig(state_dim=state_dim, **config["agent"]))
    agent.load(str(checkpoint_path))
    return env, agent, env_cfg


def _evaluate_one_rollout(
    env,
    agent,
    *,
    zone_name: str,
    candidate_index: int,
    max_steps: int,
) -> dict[str, Any]:
    state = env.reset(zone_name=zone_name, candidate_index=candidate_index)
    rx_position = np.asarray(env.rx.position, dtype=np.float32)

    no_ris_rate = float(env.evaluate_no_ris_rate())
    phase_gradient_reflector_rate = env.evaluate_phase_gradient_reflector_rate()
    if phase_gradient_reflector_rate is None:
        raise RuntimeError("The native Sionna RIS reflector baseline is unavailable.")
    phase_gradient_reflector_rate = float(phase_gradient_reflector_rate)

    step_rewards: list[float] = []
    final_grouped_action: np.ndarray | None = None
    final_env_action: np.ndarray | None = None

    for _ in range(max_steps):
        env_action, grouped_action = agent.select_action(
            state,
            deterministic=True,
            return_grouped_action=True,
        )
        next_state, reward = env.step(env_action)
        step_rewards.append(float(reward))
        final_grouped_action = np.asarray(grouped_action, dtype=np.float32).copy()
        final_env_action = np.asarray(env_action, dtype=np.float32).copy()
        state = next_state

    if final_grouped_action is None or final_env_action is None:
        raise RuntimeError("Deterministic rollout produced no action.")

    drl_avg_rate = float(np.mean(step_rewards))
    drl_final_rate = float(step_rewards[-1])
    drl_best_rate = float(np.max(step_rewards))
    phase_matrix = np.asarray(
        agent.expand_grouped_action(final_grouped_action),
        dtype=np.float32,
    )

    return {
        "zone_name": str(zone_name),
        "candidate_index": int(candidate_index),
        "rx_x_m": float(rx_position[0]),
        "rx_y_m": float(rx_position[1]),
        "rx_z_m": float(rx_position[2]),
        "no_ris_rate": no_ris_rate,
        "phase_gradient_reflector_rate": phase_gradient_reflector_rate,
        "drl_avg_rate": drl_avg_rate,
        "drl_final_rate": drl_final_rate,
        "drl_best_rate": drl_best_rate,
        "avg_minus_no_ris": drl_avg_rate - no_ris_rate,
        "avg_minus_reflector": drl_avg_rate - phase_gradient_reflector_rate,
        "final_minus_no_ris": drl_final_rate - no_ris_rate,
        "final_minus_reflector": drl_final_rate - phase_gradient_reflector_rate,
        "step_rewards_json": json.dumps(step_rewards),
        "final_grouped_action_json": json.dumps(final_grouped_action.tolist()),
        "phase_matrix": phase_matrix,
        "final_env_action": final_env_action,
    }


def _summarize_candidates(
    rollout_rows: list[dict[str, Any]],
    *,
    num_rollouts_per_candidate: int,
) -> list[dict[str, Any]]:
    keys = sorted({(str(row["zone_name"]), int(row["candidate_index"])) for row in rollout_rows})
    summary_rows: list[dict[str, Any]] = []

    metric_names = [
        "no_ris_rate",
        "phase_gradient_reflector_rate",
        "drl_avg_rate",
        "drl_final_rate",
        "drl_best_rate",
        "avg_minus_no_ris",
        "avg_minus_reflector",
        "final_minus_no_ris",
        "final_minus_reflector",
    ]

    for zone_name, candidate_index in keys:
        rows = [
            row
            for row in rollout_rows
            if str(row["zone_name"]) == zone_name and int(row["candidate_index"]) == candidate_index
        ]
        first = rows[0]
        summary: dict[str, Any] = {
            "zone_name": str(zone_name),
            "candidate_index": int(candidate_index),
            "rx_x_m": float(first["rx_x_m"]),
            "rx_y_m": float(first["rx_y_m"]),
            "rx_z_m": float(first["rx_z_m"]),
            "num_rollouts": int(len(rows)),
            "avg_beats_reflector_win_rate": float(
                np.mean([float(row["avg_minus_reflector"] > 0.0) for row in rows])
            ),
            "final_beats_reflector_win_rate": float(
                np.mean([float(row["final_minus_reflector"] > 0.0) for row in rows])
            ),
            "avg_beats_no_ris_win_rate": float(
                np.mean([float(row["avg_minus_no_ris"] > 0.0) for row in rows])
            ),
            "final_beats_no_ris_win_rate": float(
                np.mean([float(row["final_minus_no_ris"] > 0.0) for row in rows])
            ),
        }
        for metric_name in metric_names:
            values = np.asarray([float(row[metric_name]) for row in rows], dtype=np.float64)
            summary[f"{metric_name}_mean"] = float(np.mean(values))
            summary[f"{metric_name}_std"] = float(np.std(values))
            summary[f"{metric_name}_min"] = float(np.min(values))
            summary[f"{metric_name}_max"] = float(np.max(values))
        summary["avg_gain_over_reflector_pct_mean"] = (
            100.0
            * summary["avg_minus_reflector_mean"]
            / max(abs(summary["phase_gradient_reflector_rate_mean"]), EPS)
        )
        summary["final_gain_over_reflector_pct_mean"] = (
            100.0
            * summary["final_minus_reflector_mean"]
            / max(abs(summary["phase_gradient_reflector_rate_mean"]), EPS)
        )
        summary["avg_gain_over_no_ris_pct_mean"] = (
            100.0
            * summary["avg_minus_no_ris_mean"]
            / max(abs(summary["no_ris_rate_mean"]), EPS)
        )
        summary["final_gain_over_no_ris_pct_mean"] = (
            100.0
            * summary["final_minus_no_ris_mean"]
            / max(abs(summary["no_ris_rate_mean"]), EPS)
        )
        if len(rows) != num_rollouts_per_candidate:
            summary["warning"] = (
                f"expected {num_rollouts_per_candidate} rollouts, got {len(rows)}"
            )
        summary_rows.append(summary)

    return summary_rows


def _aggregate_zone_rows(candidate_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    zone_names = sorted({str(row["zone_name"]) for row in candidate_summary_rows})
    zone_rows: list[dict[str, Any]] = []

    for zone_name in zone_names:
        rows = [row for row in candidate_summary_rows if str(row["zone_name"]) == zone_name]
        zone_row = {
            "zone_name": str(zone_name),
            "num_candidates": int(len(rows)),
            "no_ris_rate_mean": float(np.mean([float(row["no_ris_rate_mean"]) for row in rows])),
            "phase_gradient_reflector_rate_mean": float(
                np.mean([float(row["phase_gradient_reflector_rate_mean"]) for row in rows])
            ),
            "drl_avg_rate_mean": float(np.mean([float(row["drl_avg_rate_mean"]) for row in rows])),
            "drl_final_rate_mean": float(np.mean([float(row["drl_final_rate_mean"]) for row in rows])),
            "avg_minus_reflector_mean": float(
                np.mean([float(row["avg_minus_reflector_mean"]) for row in rows])
            ),
            "final_minus_reflector_mean": float(
                np.mean([float(row["final_minus_reflector_mean"]) for row in rows])
            ),
            "avg_beats_reflector_rate": float(
                np.mean([float(row["avg_beats_reflector_win_rate"]) for row in rows])
            ),
            "final_beats_reflector_rate": float(
                np.mean([float(row["final_beats_reflector_win_rate"]) for row in rows])
            ),
        }
        zone_row["avg_gain_over_reflector_pct_mean"] = (
            100.0
            * zone_row["avg_minus_reflector_mean"]
            / max(abs(zone_row["phase_gradient_reflector_rate_mean"]), EPS)
        )
        zone_row["final_gain_over_reflector_pct_mean"] = (
            100.0
            * zone_row["final_minus_reflector_mean"]
            / max(abs(zone_row["phase_gradient_reflector_rate_mean"]), EPS)
        )
        zone_rows.append(zone_row)

    return zone_rows


def _aggregate_summary(
    candidate_summary_rows: list[dict[str, Any]],
    *,
    run_dir: Path,
    checkpoint_path: Path,
    config_path: Path,
    output_dir: Path,
    num_rollouts_per_candidate: int,
    representative_zone: str,
    representative_candidate_index: int,
    representative_rx_position: list[float],
) -> dict[str, Any]:
    def mean_std(key: str) -> tuple[float, float]:
        values = np.asarray([float(row[key]) for row in candidate_summary_rows], dtype=np.float64)
        return float(np.mean(values)), float(np.std(values))

    no_ris_mean, no_ris_std = mean_std("no_ris_rate_mean")
    reflector_mean, reflector_std = mean_std("phase_gradient_reflector_rate_mean")
    drl_avg_mean, drl_avg_std = mean_std("drl_avg_rate_mean")
    drl_final_mean, drl_final_std = mean_std("drl_final_rate_mean")
    drl_best_mean, drl_best_std = mean_std("drl_best_rate_mean")
    avg_minus_reflector_mean, avg_minus_reflector_std = mean_std("avg_minus_reflector_mean")
    final_minus_reflector_mean, final_minus_reflector_std = mean_std("final_minus_reflector_mean")
    avg_minus_no_ris_mean, avg_minus_no_ris_std = mean_std("avg_minus_no_ris_mean")
    final_minus_no_ris_mean, final_minus_no_ris_std = mean_std("final_minus_no_ris_mean")

    return {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "num_candidates": int(len(candidate_summary_rows)),
        "num_rollouts_per_candidate": int(num_rollouts_per_candidate),
        "evaluation_mode": "deterministic_actor_fixed_los_nlos_zone_set",
        "rx_jitter_xy_m": 0.0,
        "representative_zone": str(representative_zone),
        "representative_candidate_index": int(representative_candidate_index),
        "representative_rx_position": [float(value) for value in representative_rx_position],
        "no_ris_rate_mean_over_candidates": no_ris_mean,
        "no_ris_rate_std_over_candidates": no_ris_std,
        "phase_gradient_reflector_rate_mean_over_candidates": reflector_mean,
        "phase_gradient_reflector_rate_std_over_candidates": reflector_std,
        "drl_avg_rate_mean_over_candidates": drl_avg_mean,
        "drl_avg_rate_std_over_candidates": drl_avg_std,
        "drl_final_rate_mean_over_candidates": drl_final_mean,
        "drl_final_rate_std_over_candidates": drl_final_std,
        "drl_best_rate_mean_over_candidates": drl_best_mean,
        "drl_best_rate_std_over_candidates": drl_best_std,
        "drl_avg_minus_reflector_mean_over_candidates": avg_minus_reflector_mean,
        "drl_avg_minus_reflector_std_over_candidates": avg_minus_reflector_std,
        "drl_final_minus_reflector_mean_over_candidates": final_minus_reflector_mean,
        "drl_final_minus_reflector_std_over_candidates": final_minus_reflector_std,
        "drl_avg_minus_no_ris_mean_over_candidates": avg_minus_no_ris_mean,
        "drl_avg_minus_no_ris_std_over_candidates": avg_minus_no_ris_std,
        "drl_final_minus_no_ris_mean_over_candidates": final_minus_no_ris_mean,
        "drl_final_minus_no_ris_std_over_candidates": final_minus_no_ris_std,
        "candidate_mean_avg_beats_reflector_rate": float(
            np.mean(
                [float(row["avg_minus_reflector_mean"] > 0.0) for row in candidate_summary_rows]
            )
        ),
        "candidate_mean_final_beats_reflector_rate": float(
            np.mean(
                [float(row["final_minus_reflector_mean"] > 0.0) for row in candidate_summary_rows]
            )
        ),
        "rollout_avg_beats_reflector_rate": float(
            np.mean([float(row["avg_beats_reflector_win_rate"]) for row in candidate_summary_rows])
        ),
        "rollout_final_beats_reflector_rate": float(
            np.mean([float(row["final_beats_reflector_win_rate"]) for row in candidate_summary_rows])
        ),
    }


def _select_representative_candidate(
    candidate_summary_rows: list[dict[str, Any]],
    zone_name: str,
    explicit_index: int | None,
) -> int:
    zone_rows = [row for row in candidate_summary_rows if str(row["zone_name"]) == zone_name]
    if not zone_rows:
        raise RuntimeError(f"No candidate summaries are available for zone `{zone_name}`.")
    if explicit_index is not None:
        return int(explicit_index)

    set_mean = float(
        np.mean([float(row["drl_avg_rate_mean"]) for row in zone_rows], dtype=np.float64)
    )
    representative_row = min(
        zone_rows,
        key=lambda row: abs(float(row["drl_avg_rate_mean"]) - set_mean),
    )
    return int(representative_row["candidate_index"])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for CSV export: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_rate_by_candidate_zone(
    candidate_summary_rows: list[dict[str, Any]],
    figure_path: Path,
) -> None:
    zone_names = sorted({str(row["zone_name"]) for row in candidate_summary_rows})
    fig, axes = plt.subplots(1, len(zone_names), figsize=(8 * len(zone_names), 6), constrained_layout=True)
    if len(zone_names) == 1:
        axes = [axes]

    for ax, zone_name in zip(axes, zone_names):
        rows = [row for row in candidate_summary_rows if str(row["zone_name"]) == zone_name]
        rows = sorted(rows, key=lambda row: int(row["candidate_index"]))
        indices = np.asarray([int(row["candidate_index"]) for row in rows], dtype=np.int32)
        no_ris = np.asarray([float(row["no_ris_rate_mean"]) for row in rows], dtype=np.float64)
        reflector = np.asarray(
            [float(row["phase_gradient_reflector_rate_mean"]) for row in rows],
            dtype=np.float64,
        )
        drl_avg = np.asarray([float(row["drl_avg_rate_mean"]) for row in rows], dtype=np.float64)
        drl_final = np.asarray([float(row["drl_final_rate_mean"]) for row in rows], dtype=np.float64)

        ax.plot(indices, no_ris, marker="o", linewidth=2.0, color="#64748b", label="No RIS")
        ax.plot(indices, reflector, marker="s", linewidth=2.0, color="#111827", linestyle="--", label="Reflector")
        ax.plot(indices, drl_avg, marker="D", linewidth=2.4, color="#1d4ed8", label="DRL avg")
        ax.plot(indices, drl_final, marker="^", linewidth=2.2, color="#059669", label="DRL final")
        ax.set_title(f"{zone_name.upper()} fixed-set rate comparison")
        ax.set_xlabel("Candidate index")
        ax.set_ylabel("Rate [bit/s/Hz]")
        ax.set_xticks(indices)
        ax.legend(loc="best")

    sns.despine(fig=fig)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_zone_aggregate_bar(
    zone_summary_rows: list[dict[str, Any]],
    figure_path: Path,
) -> None:
    zone_names = [str(row["zone_name"]).upper() for row in zone_summary_rows]
    no_ris = [float(row["no_ris_rate_mean"]) for row in zone_summary_rows]
    reflector = [float(row["phase_gradient_reflector_rate_mean"]) for row in zone_summary_rows]
    drl_avg = [float(row["drl_avg_rate_mean"]) for row in zone_summary_rows]
    drl_final = [float(row["drl_final_rate_mean"]) for row in zone_summary_rows]

    x = np.arange(len(zone_names), dtype=np.float64)
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.bar(x - 1.5 * width, no_ris, width=width, color="#94a3b8", label="No RIS")
    ax.bar(x - 0.5 * width, reflector, width=width, color="#111827", label="Reflector")
    ax.bar(x + 0.5 * width, drl_avg, width=width, color="#2563eb", label="DRL avg")
    ax.bar(x + 1.5 * width, drl_final, width=width, color="#10b981", label="DRL final")
    ax.set_title("Aggregate LOS/NLOS fixed-set performance")
    ax.set_xlabel("Zone")
    ax.set_ylabel("Rate [bit/s/Hz]")
    ax.set_xticks(x)
    ax.set_xticklabels(zone_names)
    ax.legend(loc="best")
    sns.despine(fig=fig)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _rows_to_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def _format_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_format_value(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def _write_report(
    report_path: Path,
    *,
    aggregate_summary: dict[str, Any],
    zone_summary_rows: list[dict[str, Any]],
    candidate_summary_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    zone_columns = [
        "zone_name",
        "num_candidates",
        "no_ris_rate_mean",
        "phase_gradient_reflector_rate_mean",
        "drl_avg_rate_mean",
        "drl_final_rate_mean",
        "avg_minus_reflector_mean",
        "final_minus_reflector_mean",
    ]
    candidate_columns = [
        "zone_name",
        "candidate_index",
        "rx_x_m",
        "rx_y_m",
        "no_ris_rate_mean",
        "phase_gradient_reflector_rate_mean",
        "drl_avg_rate_mean",
        "drl_final_rate_mean",
        "avg_minus_reflector_mean",
        "final_minus_reflector_mean",
    ]

    lines = [
        "# Formal Fixed-Zone Evaluation",
        "",
        "## Protocol",
        "",
        f"- Run directory: `{aggregate_summary['run_dir']}`",
        f"- Config snapshot: `{aggregate_summary['config_path']}`",
        f"- Checkpoint: `{aggregate_summary['checkpoint_path']}`",
        "- TensorFlow runs on CPU and PyTorch runs on GPU when available.",
        "- RX jitter is forced to `0.0 m` so every LOS/NLOS candidate is evaluated at a fixed position.",
        (
            f"- Each candidate is re-evaluated with "
            f"`{aggregate_summary['num_rollouts_per_candidate']}` deterministic rollouts."
        ),
        "",
        "## Zone-Level Results",
        "",
        _rows_to_markdown_table(zone_summary_rows, zone_columns),
        "",
        "## Candidate-Level Results",
        "",
        _rows_to_markdown_table(candidate_summary_rows, candidate_columns),
        "",
        "## Representative Candidate",
        "",
        f"- Zone: `{aggregate_summary['representative_zone']}`",
        f"- Candidate index: `{aggregate_summary['representative_candidate_index']}`",
        f"- RX position: `{aggregate_summary['representative_rx_position']}`",
        "",
        "## Exported Figures",
        "",
        "- `formal_zone_learning_curve.png`",
        "- `formal_fixed_zone_rate_by_candidate.png`",
        "- `formal_fixed_zone_aggregate_bar.png`",
        "- `formal_fixed_zone_phase_profile_representative.png`",
        "- `formal_fixed_zone_coverage_representative.png`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.resolve()
    config_path = (args.config or (run_dir / "resolved_config.yaml")).resolve()
    checkpoint_path = (args.checkpoint or (run_dir / "recommended_agent.pt")).resolve()
    output_dir = (run_dir / args.output_subdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    set_global_seed(int(config["seed"]))

    env, agent, env_cfg = _instantiate_env_and_agent(config, checkpoint_path)
    max_steps = int(config["train"]["max_steps_per_episode"])
    zone_candidates = env.zone_candidates

    rollout_rows: list[dict[str, Any]] = []
    for zone_name in (ZONE_LOS, ZONE_NLOS):
        candidates = np.asarray(zone_candidates.get(zone_name, []), dtype=np.float32)
        for candidate_index in range(int(len(candidates))):
            for rollout_index in range(int(args.num_rollouts_per_candidate)):
                result = _evaluate_one_rollout(
                    env,
                    agent,
                    zone_name=zone_name,
                    candidate_index=candidate_index,
                    max_steps=max_steps,
                )
                result.pop("phase_matrix")
                result.pop("final_env_action")
                result["rollout_index"] = int(rollout_index)
                rollout_rows.append(result)

    if not rollout_rows:
        raise RuntimeError("No zone candidates were available for fixed-set evaluation.")

    candidate_summary_rows = _summarize_candidates(
        rollout_rows,
        num_rollouts_per_candidate=int(args.num_rollouts_per_candidate),
    )
    zone_summary_rows = _aggregate_zone_rows(candidate_summary_rows)

    representative_zone = str(args.representative_zone)
    representative_candidate_index = _select_representative_candidate(
        candidate_summary_rows,
        representative_zone,
        args.representative_candidate_index,
    )
    representative_candidates = np.asarray(zone_candidates[representative_zone], dtype=np.float32)
    representative_rx_position = representative_candidates[representative_candidate_index].tolist()

    aggregate_summary = _aggregate_summary(
        candidate_summary_rows,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        output_dir=output_dir,
        num_rollouts_per_candidate=int(args.num_rollouts_per_candidate),
        representative_zone=representative_zone,
        representative_candidate_index=representative_candidate_index,
        representative_rx_position=representative_rx_position,
    )
    aggregate_summary["zone_summaries"] = zone_summary_rows

    rollout_csv_path = output_dir / "formal_fixed_zone_rollout_metrics.csv"
    candidate_csv_path = output_dir / "formal_fixed_zone_candidate_metrics.csv"
    zone_csv_path = output_dir / "formal_fixed_zone_zone_summary.csv"
    summary_yaml_path = output_dir / "formal_fixed_zone_summary.yaml"
    report_path = output_dir / "formal_fixed_zone_report.md"

    _write_csv(rollout_csv_path, rollout_rows)
    _write_csv(candidate_csv_path, candidate_summary_rows)
    _write_csv(zone_csv_path, zone_summary_rows)
    summary_yaml_path.write_text(
        yaml.safe_dump(aggregate_summary, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    plot_academic_learning_curve(
        run_dir,
        figure_path=output_dir / "formal_zone_learning_curve.png",
    )
    _plot_rate_by_candidate_zone(
        candidate_summary_rows,
        output_dir / "formal_fixed_zone_rate_by_candidate.png",
    )
    _plot_zone_aggregate_bar(
        zone_summary_rows,
        output_dir / "formal_fixed_zone_aggregate_bar.png",
    )

    representative_rollout = _evaluate_one_rollout(
        env,
        agent,
        zone_name=representative_zone,
        candidate_index=representative_candidate_index,
        max_steps=max_steps,
    )
    representative_phase_matrix = representative_rollout["phase_matrix"]

    plot_phase_profile(
        representative_phase_matrix,
        physical_shape=(agent.config.physical_rows, agent.config.physical_cols),
        figure_path=output_dir / "formal_fixed_zone_phase_profile_representative.png",
        title=(
            "Representative Fixed-Zone RIS Phase Profile "
            f"({representative_zone} candidate {representative_candidate_index})"
        ),
    )

    env.reset(zone_name=representative_zone, candidate_index=representative_candidate_index)
    coverage_center = (
        tuple(float(v) for v in env_cfg["zone_nlos_search_center"])
        if representative_zone == ZONE_NLOS
        else tuple(float(v) for v in env_cfg["zone_los_search_center"])
    )
    coverage_size = (
        tuple(float(v) for v in env_cfg["zone_nlos_search_size"])
        if representative_zone == ZONE_NLOS
        else tuple(float(v) for v in env_cfg["zone_los_search_size"])
    )
    coverage_cell_size = (
        tuple(float(v) for v in env_cfg["zone_nlos_cell_size"])
        if representative_zone == ZONE_NLOS
        else tuple(float(v) for v in env_cfg["zone_los_cell_size"])
    )
    plot_coverage_comparison(
        env,
        representative_phase_matrix,
        metric="sinr",
        cm_center=coverage_center,
        cm_size=coverage_size,
        cm_cell_size=coverage_cell_size,
        max_depth=int(env_cfg["max_depth"]),
        num_samples=int(env_cfg["coverage_num_samples"]),
        show_native_sionna_figures=False,
        figure_path=output_dir / "formal_fixed_zone_coverage_representative.png",
    )

    _write_report(
        report_path,
        aggregate_summary=aggregate_summary,
        zone_summary_rows=zone_summary_rows,
        candidate_summary_rows=candidate_summary_rows,
        output_dir=output_dir,
    )

    print(f"Formal fixed-zone evaluation completed: {output_dir}")


if __name__ == "__main__":
    main()
