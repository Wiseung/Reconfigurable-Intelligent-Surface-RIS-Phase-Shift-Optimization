"""Sionna environment for RIS phase-shift optimization.

This implementation is aligned with the locally installed Sionna RT API, which
exposes `Scene.compute_paths()` and `Scene.coverage_map()` directly on the
scene object. To avoid PTX JIT on TensorFlow 2.15 with an RTX 5090, the
default runtime configuration keeps TensorFlow on CPU and leaves the GPU to
PyTorch. If GPU-backed TensorFlow is explicitly re-enabled, the 14 GB logical
device cap is still enforced so the remaining memory can be used by PyTorch in
the same process.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

DEFAULT_TF_DEVICE = "cpu"

_MODULE_TF_DEVICE = os.environ.get("SIONNA_TF_DEVICE", DEFAULT_TF_DEVICE).strip().lower()
if _MODULE_TF_DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf

TF_MEMORY_LIMIT_MB = 1024 * 14
DEFAULT_MAX_DEPTH = 2
DEFAULT_PATH_NUM_SAMPLES = int(2e4)
DEFAULT_COVERAGE_NUM_SAMPLES = int(1e5)
STATE_NUM_PATHS = 64
DEFAULT_PROBE_NUM_SAMPLES = 64
RIS_SHAPE = (100, 100)
TX_POSITION = np.array([-150.0, 21.0, 42.0], dtype=np.float32)
DEFAULT_RX_HEIGHT_M = 1.5
DEFAULT_CARRIER_FREQUENCY_HZ = 3.5e9
DEFAULT_BANDWIDTH_HZ = 10e6
DEFAULT_TX_POWER_DBM = 30.0
DEFAULT_NOISE_TEMPERATURE_K = 290.0
DEFAULT_TX_BORESIGHT_TARGET = np.array([0.0, 90.0, DEFAULT_RX_HEIGHT_M], dtype=np.float32)
_BOLTZMANN = 1.380649e-23
_TF_RUNTIME_CONFIG: tuple[str, int] | None = None


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Return a unit-norm copy of `vec` when possible."""
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return vec.copy()
    return vec / norm


def _configure_tensorflow_runtime(
    tf_device: str = DEFAULT_TF_DEVICE,
    memory_limit_mb: int = TF_MEMORY_LIMIT_MB,
) -> None:
    """Configure TensorFlow exactly once before Sionna creates any runtime state."""
    global _TF_RUNTIME_CONFIG

    tf_device = str(tf_device).strip().lower()
    memory_limit_mb = int(memory_limit_mb)
    requested = (tf_device, memory_limit_mb)

    if tf_device != _MODULE_TF_DEVICE:
        raise RuntimeError(
            "The requested TensorFlow device does not match the mode that was active "
            "when `env_sionna.py` was imported. Set `SIONNA_TF_DEVICE` before importing "
            f"this module. Imported mode: `{_MODULE_TF_DEVICE}`, requested: `{tf_device}`."
        )

    if _TF_RUNTIME_CONFIG is not None:
        if _TF_RUNTIME_CONFIG != requested:
            raise RuntimeError(
                "TensorFlow runtime was already configured as "
                f"{_TF_RUNTIME_CONFIG}, so it cannot be reconfigured to {requested} "
                "inside the same process."
            )
        return

    try:
        if tf_device == "cpu":
            tf.config.set_visible_devices([], "GPU")
        elif tf_device == "gpu":
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                raise RuntimeError("`tf_device='gpu'` was requested, but no TensorFlow GPU is visible.")
            tf.config.set_visible_devices(gpus[0], "GPU")
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)],
            )
        else:
            raise ValueError("`tf_device` must be either 'cpu' or 'gpu'.")
    except RuntimeError as exc:
        raise RuntimeError(
            "TensorFlow runtime was initialized before SionnaRISEnv could apply "
            f"`tf_device={tf_device}`. Create the environment before any TensorFlow "
            "GPU work in this process."
        ) from exc

    _TF_RUNTIME_CONFIG = requested

try:
    import sionna.rt as rt
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise ImportError(
        "env_sionna.py requires a Sionna installation that provides "
        "`sionna.rt`. Install Sionna with RT support before using this module."
    ) from exc


class SionnaRISEnv:
    """RIS phase-shift optimization environment backed by Sionna RT."""

    def __init__(
        self,
        carrier_frequency_hz: float = DEFAULT_CARRIER_FREQUENCY_HZ,
        bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ,
        tx_power_dbm: float = DEFAULT_TX_POWER_DBM,
        noise_temperature_k: float = DEFAULT_NOISE_TEMPERATURE_K,
        tx_boresight_target: Iterable[float] = DEFAULT_TX_BORESIGHT_TARGET,
        rx_height_m: float = DEFAULT_RX_HEIGHT_M,
        rx_jitter_xy_m: float = 2.0,
        rng_seed: int = 7,
        blind_spot_search_center: Iterable[float] = (0.0, 0.0, DEFAULT_RX_HEIGHT_M),
        blind_spot_search_size: Iterable[float] = (400.0, 400.0),
        blind_spot_cell_size: Iterable[float] = (8.0, 8.0),
        num_blind_spot_candidates: int = 16,
        ris_position: Iterable[float] | None = None,
        ris_search_along_offsets_m: Iterable[float] = (-140.0, -100.0, -60.0, -20.0, 20.0),
        ris_search_lateral_offsets_m: Iterable[float] = (-100.0, -50.0, 0.0, 50.0, 100.0),
        ris_search_heights_m: Iterable[float] = (20.0, 30.0, 40.0),
        require_native_ris: bool = True,
        tf_device: str = DEFAULT_TF_DEVICE,
        tf_memory_limit_mb: int = TF_MEMORY_LIMIT_MB,
        max_depth: int = DEFAULT_MAX_DEPTH,
        path_num_samples: int = DEFAULT_PATH_NUM_SAMPLES,
        coverage_num_samples: int = DEFAULT_COVERAGE_NUM_SAMPLES,
        probe_num_samples: int = DEFAULT_PROBE_NUM_SAMPLES,
        state_num_paths: int = STATE_NUM_PATHS,
    ) -> None:
        self.tf_device = str(tf_device).strip().lower()
        self.tf_memory_limit_mb = int(tf_memory_limit_mb)
        _configure_tensorflow_runtime(
            tf_device=self.tf_device,
            memory_limit_mb=self.tf_memory_limit_mb,
        )

        self.carrier_frequency_hz = float(carrier_frequency_hz)
        self.bandwidth_hz = float(bandwidth_hz)
        self.tx_power_dbm = float(tx_power_dbm)
        self.noise_temperature_k = float(noise_temperature_k)
        self.tx_boresight_target = np.asarray(tx_boresight_target, dtype=np.float32)
        self.rx_height_m = float(rx_height_m)
        self.rx_jitter_xy_m = float(rx_jitter_xy_m)
        self.num_blind_spot_candidates = int(num_blind_spot_candidates)
        self.max_depth = int(max_depth)
        self.path_num_samples = int(path_num_samples)
        self.coverage_num_samples = int(coverage_num_samples)
        self.probe_num_samples = int(probe_num_samples)
        self.num_samples = self.coverage_num_samples
        self.state_num_paths = int(state_num_paths)
        self.ris_rows, self.ris_cols = RIS_SHAPE
        self.action_dim = self.ris_rows * self.ris_cols
        self.rng = np.random.default_rng(rng_seed)
        self.require_native_ris = bool(require_native_ris)
        self.blind_spot_search_center = np.asarray(blind_spot_search_center, dtype=np.float32)
        self.blind_spot_search_size = np.asarray(blind_spot_search_size, dtype=np.float32)
        self.blind_spot_cell_size = np.asarray(blind_spot_cell_size, dtype=np.float32)
        self.ris_search_along_offsets_m = np.asarray(ris_search_along_offsets_m, dtype=np.float32)
        self.ris_search_lateral_offsets_m = np.asarray(
            ris_search_lateral_offsets_m, dtype=np.float32
        )
        self.ris_search_heights_m = np.asarray(ris_search_heights_m, dtype=np.float32)

        self.scene = rt.load_scene(rt.scene.etoile)
        self.scene.frequency = self.carrier_frequency_hz
        self.scene.bandwidth = self.bandwidth_hz
        self.scene.temperature = self.noise_temperature_k

        self.scene.tx_array = rt.PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization="V",
        )
        self.scene.rx_array = rt.PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )

        self.tx = rt.Transmitter(
            name="tx",
            position=TX_POSITION.tolist(),
            power_dbm=self.tx_power_dbm,
        )
        self.scene.add(self.tx)
        self.tx.look_at(self.tx_boresight_target.tolist())

        self._blind_spot_candidates = self._find_blind_spot_candidates(
            center=self.blind_spot_search_center,
            size=self.blind_spot_search_size,
            cell_size=self.blind_spot_cell_size,
        )
        self._base_rx_position = self._blind_spot_candidates[0].copy()

        self.rx = rt.Receiver(
            name="rx",
            position=self._base_rx_position.tolist(),
        )
        self.scene.add(self.rx)

        self.ris_position = (
            np.asarray(ris_position, dtype=np.float32)
            if ris_position is not None
            else self._suggest_ris_position()
        )
        self.ris = self._maybe_build_native_ris(
            position=self.ris_position,
            rx_target=self._base_rx_position,
        )

        self.current_rx_candidate_index: int | None = 0
        self.last_state: np.ndarray | None = None
        self.last_reward: float | None = None

    @property
    def blind_spot_candidates(self) -> np.ndarray:
        """Return a defensive copy of the cached blind-spot candidates."""
        return np.array(self._blind_spot_candidates, dtype=np.float32, copy=True)

    @property
    def num_available_blind_spot_candidates(self) -> int:
        """Return the number of blind-spot candidates available for reset control."""
        return int(len(self._blind_spot_candidates))

    def reset(
        self,
        reuse_rx_position: bool = False,
        candidate_index: int | None = None,
    ) -> np.ndarray:
        """Reset the RIS and optionally resample the blind-zone RX position."""
        if not reuse_rx_position:
            if candidate_index is None:
                candidate_index = int(self.rng.integers(0, len(self._blind_spot_candidates)))
            if candidate_index < 0 or candidate_index >= len(self._blind_spot_candidates):
                raise ValueError(
                    f"`candidate_index` must be in [0, {len(self._blind_spot_candidates) - 1}], "
                    f"received {candidate_index}."
                )
            base = self._blind_spot_candidates[int(candidate_index)].copy()
            jitter = self.rng.uniform(
                low=[-self.rx_jitter_xy_m, -self.rx_jitter_xy_m, 0.0],
                high=[self.rx_jitter_xy_m, self.rx_jitter_xy_m, 0.0],
            ).astype(np.float32)
            position = base + jitter
            position[2] = self.rx_height_m
            self.rx.position = position.tolist()
            self.current_rx_candidate_index = int(candidate_index)
        else:
            position = np.array(self.rx.position, dtype=np.float32, copy=True)
            position[2] = self.rx_height_m
            self.rx.position = position.tolist()

        if self.ris is not None:
            self._apply_action(np.zeros(self.action_dim, dtype=np.float32))

        self.last_state = self._compute_state()
        return self.last_state

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        """Apply a RIS phase profile and return `(next_state, reward)`."""
        if self.ris is None:
            raise RuntimeError(
                "The installed Sionna RT package does not expose a native RIS API, "
                "so `step()` cannot control a phase profile."
            )

        self._apply_action(action)
        paths = self._compute_paths(include_ris=True)
        next_state = self._state_from_paths(paths)
        reward = self._rate_from_paths(paths, include_ris=True)

        self.last_state = next_state
        self.last_reward = reward
        return next_state, reward

    def evaluate_current_rate(self) -> float:
        """Evaluate the current scene/rx configuration with the active RIS state."""
        return self._rate_from_paths(self._compute_paths(include_ris=True), include_ris=True)

    def evaluate_no_ris_rate(self) -> float:
        """Evaluate the current TX/RX pair without RIS contribution."""
        return self._rate_from_paths(self._compute_paths(include_ris=False), include_ris=False)

    def evaluate_phase_gradient_reflector_rate(self) -> float | None:
        """Evaluate the built-in phase-gradient reflector baseline."""
        if self.ris is None:
            return None

        snapshot = self._snapshot_ris_state()
        try:
            if not self._assign_phase_gradient_reflector():
                return None
            return self._rate_from_paths(self._compute_paths(include_ris=True), include_ris=True)
        finally:
            self._restore_ris_state(snapshot)

    def compute_coverage_map(
        self,
        *,
        include_ris: bool = True,
        center: Iterable[float] | None = None,
        size: Iterable[float] | None = None,
        cell_size: Iterable[float] | None = None,
        orientation: Iterable[float] = (0.0, 0.0, 0.0),
    ):
        """Compute a Sionna coverage map with the project's safety limits."""
        cm_center = (
            np.asarray(center, dtype=np.float32)
            if center is not None
            else np.array([0.0, 0.0, self.rx_height_m], dtype=np.float32)
        )
        cm_size = (
            np.asarray(size, dtype=np.float32)
            if size is not None
            else self.blind_spot_search_size
        )
        cm_cell_size = (
            np.asarray(cell_size, dtype=np.float32)
            if cell_size is not None
            else self.blind_spot_cell_size
        )
        cm_orientation = np.asarray(orientation, dtype=np.float32)

        return self.scene.coverage_map(
            rx_orientation=(0.0, 0.0, 0.0),
            max_depth=self.max_depth,
            cm_center=cm_center.tolist(),
            cm_orientation=cm_orientation.tolist(),
            cm_size=cm_size.tolist(),
            cm_cell_size=cm_cell_size.tolist(),
            num_samples=self.coverage_num_samples,
            los=True,
            reflection=True,
            diffraction=False,
            scattering=False,
            ris=include_ris,
            edge_diffraction=False,
            num_runs=1,
        )

    def _find_blind_spot_candidates(
        self,
        center: np.ndarray,
        size: np.ndarray,
        cell_size: np.ndarray,
    ) -> np.ndarray:
        """Find low-power receiver candidates from a protected coverage-map solve."""
        coverage_map = self.compute_coverage_map(
            include_ris=False,
            center=center,
            size=size,
            cell_size=cell_size,
        )

        path_gain = coverage_map.path_gain
        cell_centers = coverage_map.cell_centers
        if hasattr(path_gain, "numpy"):
            path_gain = path_gain.numpy()
        if hasattr(cell_centers, "numpy"):
            cell_centers = cell_centers.numpy()

        path_gain = np.asarray(path_gain, dtype=np.float64)
        cell_centers = np.asarray(cell_centers, dtype=np.float32)
        if path_gain.ndim == 3:
            path_gain = path_gain[0]

        distances = np.linalg.norm(cell_centers - TX_POSITION.reshape(1, 1, 3), axis=-1)
        valid_mask = np.isfinite(path_gain) & (distances >= 50.0) & (distances <= 260.0)
        positive_mask = valid_mask & (path_gain > 0.0)

        if not np.any(positive_mask):
            return np.array([[-20.0, 65.0, self.rx_height_m]], dtype=np.float32)

        positive_indices = np.argwhere(positive_mask)
        positive_gains = path_gain[positive_mask]
        order = np.argsort(positive_gains)

        preselect_count = min(len(order), max(self.num_blind_spot_candidates * 8, 16))
        weakest_positive_indices = positive_indices[order[:preselect_count]]
        weakest_positive_candidates = cell_centers[
            weakest_positive_indices[:, 0], weakest_positive_indices[:, 1]
        ].copy()
        weakest_positive_candidates[:, 2] = self.rx_height_m

        refined_candidates = self._refine_blind_spot_candidates(weakest_positive_candidates)
        if refined_candidates.size > 0:
            return refined_candidates.astype(np.float32)

        fallback_count = min(self.num_blind_spot_candidates, len(order))
        fallback_indices = positive_indices[order[:fallback_count]]
        fallback_candidates = cell_centers[fallback_indices[:, 0], fallback_indices[:, 1]].copy()
        fallback_candidates[:, 2] = self.rx_height_m
        return fallback_candidates.astype(np.float32)

    def _refine_blind_spot_candidates(self, candidates: np.ndarray) -> np.ndarray:
        """Keep the weakest candidates that still have a non-zero pointwise rate."""
        scored_candidates: list[tuple[float, np.ndarray]] = []
        for candidate in candidates:
            rate = self._estimate_no_ris_rate_at_position(candidate)
            if rate > 0.0 and np.isfinite(rate):
                scored_candidates.append((rate, candidate.copy()))

        if not scored_candidates:
            return np.empty((0, 3), dtype=np.float32)

        scored_candidates.sort(key=lambda item: item[0])
        keep = min(self.num_blind_spot_candidates, len(scored_candidates))
        return np.stack([candidate for _, candidate in scored_candidates[:keep]], axis=0)

    def _estimate_no_ris_rate_at_position(self, position: np.ndarray) -> float:
        """Estimate the non-RIS rate for one receiver position during scene setup."""
        probe_rx = rt.Receiver(name="blind_probe_rx", position=np.asarray(position).tolist())
        self.scene.add(probe_rx)
        try:
            paths = self.scene.compute_paths(
                max_depth=self.max_depth,
                num_samples=self.path_num_samples,
                los=True,
                reflection=True,
                diffraction=False,
                scattering=False,
                ris=False,
                edge_diffraction=False,
            )
            return self._rate_from_paths(paths, include_ris=False)
        finally:
            self.scene.remove("blind_probe_rx")

    def _suggest_ris_position(self) -> np.ndarray:
        """Select a static RIS position from a stable geometry heuristic."""
        tx = TX_POSITION
        rx = self._base_rx_position
        midpoint_xy = 0.5 * (tx[:2] + rx[:2])
        direction_xy = _normalize_vector(rx[:2] - tx[:2])
        if float(np.linalg.norm(direction_xy)) < 1e-6:
            direction_xy = np.array([1.0, 0.0], dtype=np.float32)
        lateral_xy = np.array([-direction_xy[1], direction_xy[0]], dtype=np.float32)

        if self.ris_search_along_offsets_m.size >= 2:
            along = float(np.mean(self.ris_search_along_offsets_m[:2]))
        elif self.ris_search_along_offsets_m.size == 1:
            along = float(self.ris_search_along_offsets_m[0])
        else:
            along = -100.0

        if self.ris_search_lateral_offsets_m.size > 0:
            lateral = float(np.max(np.abs(self.ris_search_lateral_offsets_m)))
        else:
            lateral = 100.0

        if self.ris_search_heights_m.size > 0:
            height = float(np.median(self.ris_search_heights_m))
        else:
            height = 30.0

        candidate_xy = midpoint_xy + along * direction_xy + lateral * lateral_xy
        return np.array([candidate_xy[0], candidate_xy[1], height], dtype=np.float32)

    def _generate_ris_candidates(self, tx: np.ndarray, rx: np.ndarray) -> list[np.ndarray]:
        """Generate a coarse candidate set around the TX-RX midpoint corridor."""
        midpoint_xy = 0.5 * (tx[:2] + rx[:2])
        direction_xy = _normalize_vector(rx[:2] - tx[:2])
        if float(np.linalg.norm(direction_xy)) < 1e-6:
            direction_xy = np.array([1.0, 0.0], dtype=np.float32)
        lateral_xy = np.array([-direction_xy[1], direction_xy[0]], dtype=np.float32)

        candidates: list[np.ndarray] = []
        for along in self.ris_search_along_offsets_m:
            for lateral in self.ris_search_lateral_offsets_m:
                xy = midpoint_xy + along * direction_xy + lateral * lateral_xy
                for z in self.ris_search_heights_m:
                    candidates.append(np.array([xy[0], xy[1], z], dtype=np.float32))
        return candidates

    def _evaluate_ris_position_candidate(self, position: np.ndarray, rx_target: np.ndarray) -> float:
        """Score one candidate RIS location using the built-in phase-gradient baseline."""
        temp_ris = self._create_ris_object(position=position, rx_target=rx_target, name="ris")
        self.scene.add(temp_ris)
        try:
            self._initialize_ris_profiles(temp_ris)
            if not self._assign_phase_gradient_reflector_for_ris(temp_ris):
                return -np.inf
            return self._rate_from_paths(self._compute_paths(include_ris=True), include_ris=True)
        finally:
            self.scene.remove("ris")

    def _specular_look_at_target(self, position: np.ndarray, rx_target: np.ndarray) -> np.ndarray:
        """Follow the Sionna RIS tutorial and point the RIS between TX and RX."""
        del position
        return 0.5 * (TX_POSITION + np.asarray(rx_target, dtype=np.float32))

    def _create_ris_object(
        self,
        *,
        position: np.ndarray,
        rx_target: np.ndarray,
        name: str = "ris",
    ):
        """Create a fresh native RIS object at the requested geometry."""
        ris_cls = getattr(rt, "RIS", None)
        if ris_cls is None:
            if self.require_native_ris:
                raise RuntimeError(
                    "Requested native RIS support, but the installed `sionna.rt` "
                    "package does not expose `RIS`."
                )
            return None

        errors: list[str] = []
        look_at_target = self._specular_look_at_target(position=position, rx_target=rx_target)
        constructor_candidates = [
            {
                "name": name,
                "position": np.asarray(position, dtype=np.float32).tolist(),
                "num_rows": self.ris_rows,
                "num_cols": self.ris_cols,
                "look_at": look_at_target.tolist(),
            },
            {
                "name": name,
                "position": np.asarray(position, dtype=np.float32).tolist(),
                "num_rows": self.ris_rows,
                "num_cols": self.ris_cols,
            },
        ]

        for kwargs in constructor_candidates:
            try:
                ris = ris_cls(**kwargs)
                return ris
            except (AttributeError, TypeError, ValueError) as exc:
                errors.append(str(exc))

        raise RuntimeError(
            "A native `sionna.rt.RIS` object exists, but its constructor does not "
            f"match the interface assumed by this project. Constructor errors: {errors}"
        )

    def _initialize_ris_profiles(self, ris: Any) -> None:
        """Reset the native RIS to a neutral all-ones amplitude / zero-phase profile."""
        phase_values = np.zeros((1, self.ris_rows, self.ris_cols), dtype=np.float32)
        self._assign_phase_values(ris, phase_values)

        amplitude_profile = getattr(ris, "amplitude_profile", None)
        if amplitude_profile is not None and hasattr(amplitude_profile, "values"):
            amplitude_profile.values = tf.ones_like(
                tf.convert_to_tensor(phase_values, dtype=tf.float32)
            )
        if amplitude_profile is not None and hasattr(amplitude_profile, "mode_powers"):
            amplitude_profile.mode_powers = tf.ones((1,), dtype=tf.float32)

    def _maybe_build_native_ris(
        self,
        *,
        position: np.ndarray,
        rx_target: np.ndarray,
    ):
        """Instantiate a fresh native Sionna RIS object if the API is available."""
        ris = self._create_ris_object(position=position, rx_target=rx_target, name="ris")
        if ris is None:
            return None
        self.scene.add(ris)
        self._initialize_ris_profiles(ris)
        return ris

    def _apply_action(self, action: np.ndarray) -> None:
        """Reshape and assign the RIS phase profile."""
        if self.ris is None:
            return

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(
                f"Expected an action with {self.action_dim} entries for a "
                f"{self.ris_rows}x{self.ris_cols} RIS, but got {action.size}."
            )

        phase_values = action.reshape(1, self.ris_rows, self.ris_cols)
        self._assign_phase_values(self.ris, phase_values)

    def _assign_phase_gradient_reflector(self) -> bool:
        """Configure the native Sionna RIS baseline if available."""
        if self.ris is None:
            return False
        return self._assign_phase_gradient_reflector_for_ris(self.ris)

    def _assign_phase_gradient_reflector_for_ris(self, ris: Any) -> bool:
        """Configure a specific native RIS object as a phase-gradient reflector."""
        reflector = getattr(ris, "phase_gradient_reflector", None)
        if reflector is None:
            return False

        try:
            reflector(
                sources=np.asarray(self.tx.position).tolist(),
                targets=np.asarray(self.rx.position).tolist(),
            )
            return True
        except (TypeError, ValueError):
            try:
                reflector(
                    sources=[np.asarray(self.tx.position).tolist()],
                    targets=[np.asarray(self.rx.position).tolist()],
                )
                return True
            except (TypeError, ValueError):
                return False

    def _snapshot_ris_state(self) -> dict[str, np.ndarray]:
        """Copy the RIS phase/amplitude state so temporary baselines can restore it."""
        snapshot: dict[str, np.ndarray] = {}

        if self.ris is None:
            return snapshot

        phase_values = getattr(getattr(self.ris, "phase_profile", None), "values", None)
        if phase_values is not None:
            snapshot["phase_values"] = np.array(phase_values.numpy(), copy=True)

        amplitude_profile = getattr(self.ris, "amplitude_profile", None)
        amplitude_values = getattr(amplitude_profile, "values", None)
        if amplitude_values is not None:
            snapshot["amplitude_values"] = np.array(amplitude_values.numpy(), copy=True)

        mode_powers = getattr(amplitude_profile, "mode_powers", None)
        if mode_powers is not None:
            if hasattr(mode_powers, "numpy"):
                snapshot["mode_powers"] = np.array(mode_powers.numpy(), copy=True)
            else:
                snapshot["mode_powers"] = np.array(mode_powers, copy=True)

        return snapshot

    def _restore_ris_state(self, snapshot: dict[str, np.ndarray]) -> None:
        """Restore the RIS state after a temporary baseline evaluation."""
        if self.ris is None:
            return

        if "phase_values" in snapshot:
            self.ris.phase_profile.values = tf.convert_to_tensor(
                snapshot["phase_values"], dtype=tf.float32
            )

        amplitude_profile = getattr(self.ris, "amplitude_profile", None)
        if amplitude_profile is not None and "amplitude_values" in snapshot:
            amplitude_profile.values = tf.convert_to_tensor(
                snapshot["amplitude_values"], dtype=tf.float32
            )
        if amplitude_profile is not None and "mode_powers" in snapshot:
            amplitude_profile.mode_powers = tf.convert_to_tensor(
                snapshot["mode_powers"], dtype=tf.float32
            )

    def _assign_phase_values(self, ris: Any, phase_values: np.ndarray) -> None:
        """Assign a `[num_modes, num_rows, num_cols]` phase tensor to an RIS."""
        phase_profile = getattr(ris, "phase_profile", None)
        if phase_profile is None or not hasattr(phase_profile, "values"):
            raise RuntimeError(
                "The native RIS object does not expose `phase_profile.values`. "
                "Please adapt `_assign_phase_values` to the installed Sionna API."
            )
        phase_profile.values = tf.convert_to_tensor(phase_values, dtype=tf.float32)

    def _compute_paths(self, include_ris: bool) -> Any:
        """Run a protected CIR solve with the requested depth and sample budget."""
        return self.scene.compute_paths(
            max_depth=self.max_depth,
            num_samples=self.path_num_samples,
            los=True,
            reflection=True,
            diffraction=False,
            scattering=False,
            ris=include_ris,
            edge_diffraction=False,
        )

    def _compute_state(self) -> np.ndarray:
        """Return the current complex-channel state as concatenated real/imag parts."""
        return self._state_from_paths(self._compute_paths(include_ris=True))

    def _state_from_paths(self, paths: Any) -> np.ndarray:
        """Convert the CIR tensor into a fixed-length real NumPy state vector."""
        coefficients = self._extract_valid_coefficients(paths, include_ris=True)
        if coefficients.size > self.state_num_paths:
            strongest = np.argsort(-np.abs(coefficients))[: self.state_num_paths]
            coefficients = coefficients[strongest]
        elif coefficients.size < self.state_num_paths:
            coefficients = np.pad(
                coefficients,
                (0, self.state_num_paths - coefficients.size),
                mode="constant",
                constant_values=0.0,
            )

        real = coefficients.real.astype(np.float32, copy=False)
        imag = coefficients.imag.astype(np.float32, copy=False)
        return np.concatenate([real, imag], axis=0)

    def _rate_from_paths(self, paths: Any, include_ris: bool = True) -> float:
        """Compute the single-user Shannon rate from the effective SISO CIR."""
        coefficients = self._extract_valid_coefficients(paths, include_ris=include_ris)
        if coefficients.size == 0:
            channel_gain = 0.0
        else:
            effective_channel = np.sum(coefficients, dtype=np.complex64)
            channel_gain = float(np.abs(effective_channel) ** 2)

        tx_power_w = 10.0 ** ((self.tx_power_dbm - 30.0) / 10.0)
        noise_power_w = _BOLTZMANN * self.noise_temperature_k * self.bandwidth_hz
        snr = (tx_power_w * channel_gain) / noise_power_w
        return float(np.log2(1.0 + snr))

    def _extract_valid_coefficients(self, paths: Any, include_ris: bool) -> np.ndarray:
        """Return all finite CIR coefficients for the requested path family."""
        coefficients, delays = paths.cir(
            los=True,
            reflection=True,
            diffraction=False,
            scattering=False,
            ris=include_ris,
            cluster_ris_paths=False,
            num_paths=None,
        )
        coeff_np = np.asarray(coefficients.numpy(), dtype=np.complex64).reshape(-1)
        delay_np = np.asarray(delays.numpy(), dtype=np.float32).reshape(-1)
        valid = (
            np.isfinite(coeff_np.real)
            & np.isfinite(coeff_np.imag)
            & np.isfinite(delay_np)
            & (delay_np >= 0.0)
        )
        return coeff_np[valid]

    def _has_nonzero_channel(self, tx_position: np.ndarray, rx_position: np.ndarray) -> bool:
        """Check whether Sionna reports any non-zero LoS path between two points."""
        probe_tx = rt.Transmitter(name="probe_tx", position=np.asarray(tx_position).tolist())
        probe_rx = rt.Receiver(name="probe_rx", position=np.asarray(rx_position).tolist())

        self.scene.add(probe_tx)
        self.scene.add(probe_rx)
        try:
            paths = self.scene.compute_paths(
                max_depth=0,
                num_samples=self.probe_num_samples,
                los=True,
                reflection=False,
                diffraction=False,
                scattering=False,
                ris=False,
                edge_diffraction=False,
            )
            coefficients, _ = paths.cir(
                los=True,
                reflection=False,
                diffraction=False,
                scattering=False,
                ris=False,
                num_paths=1,
            )
            probe_pair = coefficients[0, -1, :, -1, :, :, :]
            return bool(np.any(np.abs(probe_pair.numpy()) > 0.0))
        finally:
            self.scene.remove("probe_tx")
            self.scene.remove("probe_rx")
