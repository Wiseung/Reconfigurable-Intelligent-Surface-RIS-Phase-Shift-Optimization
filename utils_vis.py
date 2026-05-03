"""Visualization utilities for RIS optimization evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")

EPS = 1e-12


def plot_coverage_comparison(
    env: Any,
    drl_phase_profile: np.ndarray,
    *,
    metric: str = "sinr",
    tx: int | str | None = 0,
    cm_center: tuple[float, float, float] | None = None,
    cm_orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cm_size: tuple[float, float] = (400.0, 400.0),
    cm_cell_size: tuple[float, float] = (8.0, 8.0),
    max_depth: int | None = None,
    num_samples: int | None = None,
    show_native_sionna_figures: bool = True,
    show_rx_marker: bool = True,
    show_ris_marker: bool = True,
    figure_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare no-RIS and DRL-optimized RIS coverage maps in the same scene.

    This function uses Sionna's native coverage-map API to compute the maps and
    also calls the built-in `CoverageMap.show()` plotting API for traceability.
    A cleaner side-by-side Matplotlib figure is then generated for reports.
    """
    if metric not in {"path_gain", "rss", "sinr"}:
        raise ValueError("`metric` must be one of {'path_gain', 'rss', 'sinr'}.")
    if not hasattr(env, "scene") or not hasattr(env.scene, "coverage_map"):
        raise RuntimeError(
            "The provided environment does not expose `scene.coverage_map()`, "
            "which is required for coverage-map visualization."
        )
    if getattr(env, "ris", None) is None:
        raise RuntimeError(
            "plot_coverage_comparison() requires an environment with a native RIS object."
        )

    max_depth = int(max_depth if max_depth is not None else getattr(env, "max_depth", 3))
    num_samples = int(
        num_samples if num_samples is not None else getattr(env, "num_samples", int(1e6))
    )
    if cm_center is None:
        rx_height = float(getattr(env, "rx_height_m", 1.5))
        cm_center = (0.0, 0.0, rx_height)

    phase_matrix = _as_phase_matrix(drl_phase_profile, env=env)
    original_phase_state = _snapshot_phase_profile(env.ris)

    try:
        no_ris_map = _compute_coverage_map(
            env=env,
            include_ris=False,
            cm_center=cm_center,
            cm_orientation=cm_orientation,
            cm_size=cm_size,
            cm_cell_size=cm_cell_size,
            max_depth=max_depth,
            num_samples=num_samples,
        )

        _set_ris_phase_profile(env.ris, phase_matrix)
        drl_map = _compute_coverage_map(
            env=env,
            include_ris=True,
            cm_center=cm_center,
            cm_orientation=cm_orientation,
            cm_size=cm_size,
            cm_cell_size=cm_cell_size,
            max_depth=max_depth,
            num_samples=num_samples,
        )
    finally:
        _restore_phase_profile(env.ris, original_phase_state)

    native_figures: dict[str, Any] = {}
    if show_native_sionna_figures:
        native_figures["no_ris"] = no_ris_map.show(
            metric=metric,
            tx=tx,
            show_tx=True,
            show_rx=show_rx_marker,
            show_ris=show_ris_marker,
        )
        native_figures["drl_optimized"] = drl_map.show(
            metric=metric,
            tx=tx,
            show_tx=True,
            show_rx=show_rx_marker,
            show_ris=show_ris_marker,
        )

    no_ris_db = _extract_metric_db(no_ris_map, metric=metric, tx=tx)
    drl_db = _extract_metric_db(drl_map, metric=metric, tx=tx)
    extent = _coverage_extent(no_ris_map)

    finite_values = np.concatenate(
        [
            no_ris_db[np.isfinite(no_ris_db)],
            drl_db[np.isfinite(drl_db)],
        ]
    )
    if finite_values.size == 0:
        raise RuntimeError("Coverage maps contain no finite values to visualize.")

    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    cmap = sns.color_palette("mako", as_cmap=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    panels = [
        (axes[0], no_ris_db, "No RIS"),
        (axes[1], drl_db, "DRL-Optimized RIS"),
    ]

    image = None
    for ax, data_db, title in panels:
        image = ax.imshow(
            data_db,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        _overlay_scene_markers(
            ax=ax,
            env=env,
            show_rx_marker=show_rx_marker,
            show_ris_marker=show_ris_marker,
        )
        ax.set_title(f"{title} {metric.upper()} Coverage")
        ax.set_xlabel("X position [m]")
        ax.set_ylabel("Y position [m]")

    assert image is not None
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label(_metric_label(metric))
    _add_shared_legend(fig, axes[1])
    sns.despine(fig=fig)

    if figure_path is not None:
        fig.savefig(Path(figure_path), dpi=300, bbox_inches="tight")

    return {
        "figure": fig,
        "axes": axes,
        "coverage_maps": {
            "no_ris": no_ris_map,
            "drl_optimized": drl_map,
        },
        "native_figures": native_figures,
    }


def plot_learning_curve(
    run_path: str | Path,
    *,
    smoothing_window: int = 10,
    baseline_key: str = "phase_gradient_reflector_rate",
    figure_path: str | Path | None = None,
) -> dict[str, Any]:
    """Plot a smoothed reward curve and compare it with the reflector baseline."""
    metrics_path, csv_path = _resolve_training_artifacts(run_path)
    baseline_metrics, episode_metrics = _load_training_metrics(metrics_path, csv_path)

    episodes = np.asarray([row["episode"] for row in episode_metrics], dtype=np.int32)
    rewards = np.asarray([row["avg_reward"] for row in episode_metrics], dtype=np.float64)
    if rewards.size == 0:
        raise RuntimeError("No episode rewards were found in the training artifacts.")

    window = max(1, min(int(smoothing_window), rewards.size))
    smoothed_rewards = _moving_average(rewards, window=window)
    baseline = baseline_metrics.get(baseline_key)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.lineplot(
        x=episodes,
        y=rewards,
        ax=ax,
        label="Episode average reward",
        color="#94a3b8",
        linewidth=1.5,
        alpha=0.55,
    )
    sns.lineplot(
        x=episodes,
        y=smoothed_rewards,
        ax=ax,
        label=f"Smoothed reward (window={window})",
        color="#1d4ed8",
        linewidth=2.5,
    )

    improvement_percent = None
    best_smoothed_reward = float(np.max(smoothed_rewards))
    if baseline is not None:
        ax.axhline(
            float(baseline),
            color="#111827",
            linestyle="--",
            linewidth=2.0,
            label="Phase-gradient reflector baseline",
        )
        improvement_percent = 100.0 * (best_smoothed_reward - float(baseline)) / max(
            abs(float(baseline)),
            EPS,
        )
        anchor_x = int(episodes[-1])
        anchor_y = best_smoothed_reward
        relation = "above" if improvement_percent >= 0.0 else "below"
        ax.annotate(
            f"Best smoothed reward is {abs(improvement_percent):.2f}% {relation} baseline",
            xy=(anchor_x, anchor_y),
            xytext=(0.48, 0.12),
            textcoords="axes fraction",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#1f2937", "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "color": "#1f2937", "lw": 1.5},
        )

    ax.set_title("RIS DRL Training Reward Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average reward [bit/s/Hz]")
    ax.legend(loc="best")
    sns.despine(fig=fig)

    if figure_path is not None:
        fig.savefig(Path(figure_path), dpi=300, bbox_inches="tight")

    return {
        "figure": fig,
        "ax": ax,
        "episodes": episodes,
        "rewards": rewards,
        "smoothed_rewards": smoothed_rewards,
        "baseline": baseline,
        "best_smoothed_reward": best_smoothed_reward,
        "improvement_percent": improvement_percent,
        "metrics_path": metrics_path,
        "csv_path": csv_path,
    }


def plot_phase_profile(
    phase_profile: np.ndarray,
    *,
    physical_shape: tuple[int, int] = (100, 100),
    figure_path: str | Path | None = None,
    title: str = "Final RIS Phase Profile",
) -> dict[str, Any]:
    """Plot the final 100x100 RIS phase matrix as a 2D heatmap."""
    phase_matrix = _as_phase_matrix(phase_profile, physical_shape=physical_shape)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    heatmap = sns.heatmap(
        phase_matrix,
        ax=ax,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        cbar_kws={"label": "Phase shift [rad]"},
    )

    _set_sparse_heatmap_ticks(ax, phase_matrix.shape)
    ax.set_title(title)
    ax.set_xlabel("RIS column index")
    ax.set_ylabel("RIS row index")
    sns.despine(fig=fig, left=False, bottom=False)

    if figure_path is not None:
        fig.savefig(Path(figure_path), dpi=300, bbox_inches="tight")

    return {
        "figure": fig,
        "ax": ax,
        "heatmap": heatmap,
        "phase_matrix": phase_matrix,
    }


def _compute_coverage_map(
    *,
    env: Any,
    include_ris: bool,
    cm_center: tuple[float, float, float],
    cm_orientation: tuple[float, float, float],
    cm_size: tuple[float, float],
    cm_cell_size: tuple[float, float],
    max_depth: int,
    num_samples: int,
):
    return env.scene.coverage_map(
        rx_orientation=(0.0, 0.0, 0.0),
        max_depth=max_depth,
        cm_center=list(cm_center),
        cm_orientation=list(cm_orientation),
        cm_size=list(cm_size),
        cm_cell_size=list(cm_cell_size),
        num_samples=num_samples,
        ris=include_ris,
    )


def _extract_metric_db(coverage_map: Any, metric: str, tx: int | str | None) -> np.ndarray:
    values = getattr(coverage_map, metric)
    if hasattr(values, "numpy"):
        values = values.numpy()
    values = np.asarray(values, dtype=np.float64)

    if tx is None:
        values = np.max(values, axis=0)
    else:
        tx_index = _resolve_tx_index(coverage_map, tx)
        values = values[tx_index]

    safe_values = np.maximum(values, EPS)
    if metric == "rss":
        return 10.0 * np.log10(safe_values) + 30.0
    return 10.0 * np.log10(safe_values)


def _resolve_tx_index(coverage_map: Any, tx: int | str) -> int:
    if isinstance(tx, int):
        return tx
    mapping = getattr(coverage_map, "_tx_name_2_ind", {})
    if tx not in mapping:
        raise ValueError(f"Unknown transmitter name: {tx}")
    return int(mapping[tx])


def _coverage_extent(coverage_map: Any) -> tuple[float, float, float, float]:
    centers = coverage_map.cell_centers
    cell_size = coverage_map.cell_size
    if hasattr(centers, "numpy"):
        centers = centers.numpy()
    if hasattr(cell_size, "numpy"):
        cell_size = cell_size.numpy()

    centers = np.asarray(centers, dtype=np.float64)
    cell_size = np.asarray(cell_size, dtype=np.float64)

    x_min = float(centers[0, 0, 0] - 0.5 * cell_size[0])
    x_max = float(centers[0, -1, 0] + 0.5 * cell_size[0])
    y_min = float(centers[0, 0, 1] - 0.5 * cell_size[1])
    y_max = float(centers[-1, 0, 1] + 0.5 * cell_size[1])
    return x_min, x_max, y_min, y_max


def _overlay_scene_markers(
    *,
    ax: Any,
    env: Any,
    show_rx_marker: bool,
    show_ris_marker: bool,
) -> None:
    tx_position = np.asarray(env.tx.position, dtype=np.float64)
    ax.scatter(
        tx_position[0],
        tx_position[1],
        marker="^",
        s=90,
        c="#dc2626",
        edgecolors="white",
        linewidths=0.8,
        label="TX",
    )

    if show_rx_marker and hasattr(env, "rx"):
        rx_position = np.asarray(env.rx.position, dtype=np.float64)
        ax.scatter(
            rx_position[0],
            rx_position[1],
            marker="X",
            s=90,
            c="#2563eb",
            edgecolors="white",
            linewidths=0.8,
            label="RX",
        )

    if show_ris_marker and getattr(env, "ris", None) is not None:
        ris_position = np.asarray(env.ris.position, dtype=np.float64)
        ax.scatter(
            ris_position[0],
            ris_position[1],
            marker="s",
            s=85,
            c="#111827",
            edgecolors="white",
            linewidths=0.8,
            label="RIS",
        )


def _add_shared_legend(fig: Any, ax: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique[label] = handle
    if unique:
        fig.legend(
            unique.values(),
            unique.keys(),
            loc="upper center",
            ncol=len(unique),
            bbox_to_anchor=(0.5, 1.03),
            frameon=True,
        )


def _metric_label(metric: str) -> str:
    if metric == "sinr":
        return "SINR [dB]"
    if metric == "rss":
        return "RSS [dBm]"
    return "Path gain [dB]"


def _snapshot_phase_profile(ris: Any) -> dict[str, Any]:
    phase_profile = getattr(ris, "phase_profile", None)
    if phase_profile is None:
        raise RuntimeError("RIS object does not expose `phase_profile`.")

    if hasattr(phase_profile, "values"):
        values = phase_profile.values
        if hasattr(values, "numpy"):
            values = values.numpy()
        return {"mode": "values", "data": np.array(values, dtype=np.float32, copy=True)}

    if hasattr(phase_profile, "numpy"):
        return {"mode": "attribute", "data": np.array(phase_profile.numpy(), dtype=np.float32)}

    return {"mode": "attribute", "data": np.array(phase_profile, dtype=np.float32, copy=True)}


def _restore_phase_profile(ris: Any, snapshot: dict[str, Any]) -> None:
    _set_ris_phase_profile(ris, snapshot["data"])


def _set_ris_phase_profile(ris: Any, phase_matrix: np.ndarray) -> None:
    phase_matrix = np.asarray(phase_matrix, dtype=np.float32)
    profile = getattr(ris, "phase_profile", None)
    if profile is None:
        raise RuntimeError("RIS object does not expose `phase_profile`.")

    if phase_matrix.ndim == 2:
        phase_values = np.expand_dims(phase_matrix, axis=0)
    else:
        phase_values = phase_matrix

    if hasattr(profile, "values"):
        try:
            profile.values = phase_values
            return
        except Exception:
            import tensorflow as tf

            profile.values = tf.convert_to_tensor(phase_values, dtype=tf.float32)
            return

    if hasattr(profile, "assign"):
        try:
            profile.assign(phase_values)
            return
        except Exception:
            import tensorflow as tf

            profile.assign(tf.convert_to_tensor(phase_values, dtype=tf.float32))
            return

    setattr(ris, "phase_profile", phase_values)


def _as_phase_matrix(
    phase_profile: np.ndarray,
    *,
    env: Any | None = None,
    physical_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    phase_array = np.asarray(phase_profile, dtype=np.float32)

    if phase_array.ndim == 2:
        return phase_array

    if phase_array.ndim != 1:
        raise ValueError("Phase profile must be a flat vector or a 2D matrix.")

    if physical_shape is None and env is not None:
        physical_shape = (int(getattr(env, "ris_rows", 100)), int(getattr(env, "ris_cols", 100)))
    if physical_shape is None:
        total = int(np.sqrt(phase_array.size))
        if total * total != phase_array.size:
            raise ValueError(
                "Cannot infer a square phase matrix from the provided flat vector."
            )
        physical_shape = (total, total)

    expected_size = int(np.prod(physical_shape))
    if phase_array.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} phase entries for shape {physical_shape}, "
            f"but received {phase_array.size}."
        )
    return phase_array.reshape(physical_shape)


def _resolve_training_artifacts(run_path: str | Path) -> tuple[Path | None, Path | None]:
    path = Path(run_path)
    if path.is_dir():
        metrics_path = path / "metrics.jsonl"
        csv_path = path / "episode_metrics.csv"
        return (
            metrics_path if metrics_path.exists() else None,
            csv_path if csv_path.exists() else None,
        )

    if path.suffix == ".jsonl":
        csv_path = path.with_name("episode_metrics.csv")
        return path, csv_path if csv_path.exists() else None

    if path.suffix == ".csv":
        metrics_path = path.with_name("metrics.jsonl")
        return metrics_path if metrics_path.exists() else None, path

    raise ValueError(
        "run_path must point to a training run directory, `metrics.jsonl`, or `episode_metrics.csv`."
    )


def _load_training_metrics(
    metrics_path: Path | None,
    csv_path: Path | None,
) -> tuple[dict[str, float | None], list[dict[str, float]]]:
    baseline_metrics: dict[str, float | None] = {
        "no_ris_rate": None,
        "phase_gradient_reflector_rate": None,
    }
    episode_metrics: list[dict[str, float]] = []

    if metrics_path is not None and metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get("type") == "baseline":
                    baseline_metrics["no_ris_rate"] = _maybe_float(payload.get("no_ris_rate"))
                    baseline_metrics["phase_gradient_reflector_rate"] = _maybe_float(
                        payload.get("phase_gradient_reflector_rate")
                    )
                elif payload.get("type") == "episode":
                    episode_metrics.append(
                        {
                            "episode": int(payload["episode"]),
                            "avg_reward": float(payload["avg_reward"]),
                        }
                    )

    if not episode_metrics and csv_path is not None and csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                episode_metrics.append(
                    {
                        "episode": int(float(row["episode"])),
                        "avg_reward": float(row["avg_reward"]),
                    }
                )

    episode_metrics.sort(key=lambda item: item["episode"])
    return baseline_metrics, episode_metrics


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float64, copy=True)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _set_sparse_heatmap_ticks(ax: Any, shape: tuple[int, int]) -> None:
    num_rows, num_cols = shape
    row_step = max(1, num_rows // 10)
    col_step = max(1, num_cols // 10)
    ax.set_xticks(np.arange(0, num_cols + 1, col_step) + 0.5)
    ax.set_xticklabels([str(i) for i in range(0, num_cols + 1, col_step)], rotation=0)
    ax.set_yticks(np.arange(0, num_rows + 1, row_step) + 0.5)
    ax.set_yticklabels([str(i) for i in range(0, num_rows + 1, row_step)], rotation=0)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


__all__ = [
    "plot_coverage_comparison",
    "plot_learning_curve",
    "plot_phase_profile",
]
