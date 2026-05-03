"""Sionna environment for RIS phase-shift optimization.

This implementation is aligned with the locally installed Sionna RT API, which
exposes `Scene.compute_paths()` and `Scene.coverage_map()` directly on the
scene object. The environment keeps TensorFlow on a strict 14 GB logical GPU
budget so the remaining memory can be used by PyTorch in the same process.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import tensorflow as tf

TF_MEMORY_LIMIT_MB = 1024 * 14
MAX_DEPTH = 3
NUM_SAMPLES = int(1e6)
STATE_NUM_PATHS = 64
PROBE_NUM_SAMPLES = 256
RIS_SHAPE = (100, 100)
TX_POSITION = np.array([-150.0, 21.0, 42.0], dtype=np.float32)
DEFAULT_RX_HEIGHT_M = 1.5
DEFAULT_CARRIER_FREQUENCY_HZ = 3.5e9
DEFAULT_BANDWIDTH_HZ = 10e6
DEFAULT_TX_POWER_DBM = 30.0
DEFAULT_NOISE_TEMPERATURE_K = 290.0
_BOLTZMANN = 1.380649e-23


def _configure_tensorflow_memory(memory_limit_mb: int = TF_MEMORY_LIMIT_MB) -> None:
    """Hard-cap TensorFlow GPU memory before any other framework starts."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return

    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)],
        )
    except RuntimeError:
        # TensorFlow was already initialized by the caller. Keep going so the
        # environment remains importable, but the memory cap might not take effect.
        pass


_configure_tensorflow_memory()

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
        rx_height_m: float = DEFAULT_RX_HEIGHT_M,
        rx_jitter_xy_m: float = 2.0,
        rng_seed: int = 7,
        blind_spot_search_center: Iterable[float] = (0.0, 0.0, DEFAULT_RX_HEIGHT_M),
        blind_spot_search_size: Iterable[float] = (400.0, 400.0),
        blind_spot_cell_size: Iterable[float] = (8.0, 8.0),
        num_blind_spot_candidates: int = 16,
        ris_position: Iterable[float] | None = None,
        require_native_ris: bool = True,
        state_num_paths: int = STATE_NUM_PATHS,
    ) -> None:
        self.carrier_frequency_hz = float(carrier_frequency_hz)
        self.bandwidth_hz = float(bandwidth_hz)
        self.tx_power_dbm = float(tx_power_dbm)
        self.noise_temperature_k = float(noise_temperature_k)
        self.rx_height_m = float(rx_height_m)
        self.rx_jitter_xy_m = float(rx_jitter_xy_m)
        self.num_blind_spot_candidates = int(num_blind_spot_candidates)
        self.max_depth = MAX_DEPTH
        self.num_samples = NUM_SAMPLES
        self.state_num_paths = int(state_num_paths)
        self.ris_rows, self.ris_cols = RIS_SHAPE
        self.action_dim = self.ris_rows * self.ris_cols
        self.rng = np.random.default_rng(rng_seed)
        self.require_native_ris = bool(require_native_ris)
        self.blind_spot_search_center = np.asarray(blind_spot_search_center, dtype=np.float32)
        self.blind_spot_search_size = np.asarray(blind_spot_search_size, dtype=np.float32)
        self.blind_spot_cell_size = np.asarray(blind_spot_cell_size, dtype=np.float32)

        self.scene = rt.load_scene(rt.scene.etoile)
        self.scene.frequency = self.carrier_frequency_hz
        self.scene.bandwidth = self.bandwidth_hz
        self.scene.temperature = self.noise_temperature_k

        self.scene.tx_array = rt.PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
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
        self.tx.look_at(self.rx.position)

        self.ris_position = (
            np.asarray(ris_position, dtype=np.float32)
            if ris_position is not None
            else self._suggest_ris_position()
        )
        self.ris = self._maybe_build_native_ris()

        self.last_state: np.ndarray | None = None
        self.last_reward: float | None = None

    def reset(self) -> np.ndarray:
        """Randomly perturb the blind-zone RX position and return the channel state."""
        base = self._blind_spot_candidates[
            int(self.rng.integers(0, len(self._blind_spot_candidates)))
        ].copy()
        jitter = self.rng.uniform(
            low=[-self.rx_jitter_xy_m, -self.rx_jitter_xy_m, 0.0],
            high=[self.rx_jitter_xy_m, self.rx_jitter_xy_m, 0.0],
        ).astype(np.float32)
        position = base + jitter
        position[2] = self.rx_height_m

        self.rx.position = position.tolist()
        self.tx.look_at(position.tolist())

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
        reward = self._rate_from_paths(paths)

        self.last_state = next_state
        self.last_reward = reward
        return next_state, reward

    def evaluate_current_rate(self) -> float:
        """Evaluate the current scene/rx configuration with the active RIS state."""
        return self._rate_from_paths(self._compute_paths(include_ris=True))

    def evaluate_no_ris_rate(self) -> float:
        """Evaluate the current TX/RX pair without RIS contribution."""
        return self._rate_from_paths(self._compute_paths(include_ris=False))

    def evaluate_phase_gradient_reflector_rate(self) -> float | None:
        """Evaluate the built-in phase-gradient reflector baseline."""
        if self.ris is None:
            return None

        snapshot = self._snapshot_ris_state()
        try:
            if not self._assign_phase_gradient_reflector():
                return None
            return self._rate_from_paths(self._compute_paths(include_ris=True))
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
            num_samples=self.num_samples,
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

        if not np.any(valid_mask):
            return np.array([[-20.0, 65.0, self.rx_height_m]], dtype=np.float32)

        valid_indices = np.argwhere(valid_mask)
        valid_gains = path_gain[valid_mask]
        order = np.argsort(valid_gains)
        num_keep = min(self.num_blind_spot_candidates, len(order))
        worst_indices = valid_indices[order[:num_keep]]

        candidates = cell_centers[worst_indices[:, 0], worst_indices[:, 1]].copy()
        candidates[:, 2] = self.rx_height_m
        return candidates.astype(np.float32)

    def _suggest_ris_position(self) -> np.ndarray:
        """Place the RIS near the TX-RX midpoint while preserving line-of-sight."""
        tx = TX_POSITION
        rx = self._base_rx_position
        midpoint = 0.5 * (tx + rx)
        direction = rx[:2] - tx[:2]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = direction / norm

        lateral = np.array([-direction[1], direction[0]], dtype=np.float32)
        candidate_xy = [
            midpoint[:2] + 15.0 * lateral,
            midpoint[:2] - 15.0 * lateral,
            midpoint[:2] + 30.0 * lateral,
            midpoint[:2] - 30.0 * lateral,
            midpoint[:2],
        ]
        candidate_z = [25.0, 30.0, 35.0, 40.0]

        for xy in candidate_xy:
            for z in candidate_z:
                candidate = np.array([xy[0], xy[1], z], dtype=np.float32)
                if self._has_nonzero_channel(TX_POSITION, candidate) and self._has_nonzero_channel(
                    candidate, rx
                ):
                    return candidate

        return np.array([midpoint[0], midpoint[1], 30.0], dtype=np.float32)

    def _maybe_build_native_ris(self):
        """Instantiate a native Sionna RIS object if the installation exposes it."""
        ris_cls = getattr(rt, "RIS", None)
        if ris_cls is None:
            if self.require_native_ris:
                raise RuntimeError(
                    "Requested native RIS support, but the installed `sionna.rt` "
                    "package does not expose `RIS`."
                )
            return None

        errors: list[str] = []
        constructor_candidates = [
            {
                "name": "ris",
                "position": self.ris_position.tolist(),
                "num_rows": self.ris_rows,
                "num_cols": self.ris_cols,
            },
            {
                "name": "ris",
                "position": self.ris_position.tolist(),
                "num_rows": self.ris_rows,
                "num_cols": self.ris_cols,
                "look_at": np.asarray(self.rx.position).tolist(),
            },
        ]

        for kwargs in constructor_candidates:
            ris = None
            try:
                ris = ris_cls(**kwargs)
                self.scene.add(ris)
                if hasattr(ris, "look_at"):
                    ris.look_at(np.asarray(self.rx.position).tolist())
                self._assign_phase_values(
                    ris,
                    np.zeros((1, self.ris_rows, self.ris_cols), dtype=np.float32),
                )
                return ris
            except (AttributeError, TypeError, ValueError) as exc:
                if ris is not None and "ris" in getattr(self.scene, "ris", {}):
                    self.scene.remove("ris")
                errors.append(str(exc))

        raise RuntimeError(
            "A native `sionna.rt.RIS` object exists, but its constructor does not "
            f"match the interface assumed by this project. Constructor errors: {errors}"
        )

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
        reflector = getattr(self.ris, "phase_gradient_reflector", None)
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
            num_samples=self.num_samples,
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
        coefficients, _ = paths.cir(
            los=True,
            reflection=True,
            diffraction=False,
            scattering=False,
            ris=True,
            cluster_ris_paths=True,
            num_paths=self.state_num_paths,
        )
        real_coeff = tf.math.real(coefficients)
        imag_coeff = tf.math.imag(coefficients)
        real_coeff = tf.where(tf.math.is_finite(real_coeff), real_coeff, tf.zeros_like(real_coeff))
        imag_coeff = tf.where(tf.math.is_finite(imag_coeff), imag_coeff, tf.zeros_like(imag_coeff))
        real = tf.reshape(real_coeff, [-1])
        imag = tf.reshape(imag_coeff, [-1])
        state = tf.concat([real, imag], axis=0)
        return state.numpy().astype(np.float32)

    def _rate_from_paths(self, paths: Any) -> float:
        """Compute the single-user Shannon rate from the effective SISO CIR."""
        coefficients, _ = paths.cir(
            los=True,
            reflection=True,
            diffraction=False,
            scattering=False,
            ris=True,
            cluster_ris_paths=True,
            num_paths=self.state_num_paths,
        )
        real_coeff = tf.math.real(coefficients)
        imag_coeff = tf.math.imag(coefficients)
        real_coeff = tf.where(tf.math.is_finite(real_coeff), real_coeff, tf.zeros_like(real_coeff))
        imag_coeff = tf.where(tf.math.is_finite(imag_coeff), imag_coeff, tf.zeros_like(imag_coeff))
        coefficients = tf.complex(real_coeff, imag_coeff)
        effective_channel = tf.reduce_sum(coefficients, axis=-2)
        channel_gain = tf.reduce_sum(tf.abs(effective_channel) ** 2)
        channel_gain = tf.where(
            tf.math.is_finite(channel_gain),
            channel_gain,
            tf.zeros_like(channel_gain),
        )

        tx_power_w = 10.0 ** ((self.tx_power_dbm - 30.0) / 10.0)
        noise_power_w = _BOLTZMANN * self.noise_temperature_k * self.bandwidth_hz
        snr = (tx_power_w * tf.cast(channel_gain, tf.float32)) / tf.constant(
            noise_power_w, dtype=tf.float32
        )
        rate = tf.math.log1p(snr) / tf.math.log(tf.constant(2.0, dtype=tf.float32))
        return float(rate.numpy())

    def _has_nonzero_channel(self, tx_position: np.ndarray, rx_position: np.ndarray) -> bool:
        """Check whether Sionna reports any non-zero LoS path between two points."""
        probe_tx = rt.Transmitter(name="probe_tx", position=np.asarray(tx_position).tolist())
        probe_rx = rt.Receiver(name="probe_rx", position=np.asarray(rx_position).tolist())

        self.scene.add(probe_tx)
        self.scene.add(probe_rx)
        try:
            paths = self.scene.compute_paths(
                max_depth=0,
                num_samples=PROBE_NUM_SAMPLES,
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
