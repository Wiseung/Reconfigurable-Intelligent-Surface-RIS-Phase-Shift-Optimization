"""PyTorch SAC agent for grouped RIS phase-shift optimization.

Design choices driven by the project constraints:

1. The physical RIS has 100x100 elements, but the policy operates on a grouped
   10x10 control grid to keep optimization tractable on a 24 GB GPU.
2. Replay data is stored strictly as CPU-side NumPy arrays. Only sampled mini
   batches are transferred to the target device.
3. State tensors can contain very small channel coefficients. A LayerNorm is
   therefore applied at the input of both actor and critics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


@dataclass(slots=True)
class SACConfig:
    """Configuration for the grouped RIS SAC agent."""

    state_dim: int
    grouped_rows: int = 10
    grouped_cols: int = 10
    physical_rows: int = 100
    physical_cols: int = 100
    hidden_dim: int = 512
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.2
    batch_size: int = 128
    replay_capacity: int = 20000
    dual_replay: bool = False
    hard_replay_capacity: int | None = None
    hard_replay_ratio: float = 0.25
    hard_replay_prioritized: bool = False
    hard_replay_priority_alpha: float = 1.0
    hard_replay_uniform_ratio: float = 0.0
    hard_replay_priority_epsilon: float = 1e-3
    prioritized_replay: bool = False
    replay_priority_alpha: float = 1.0
    replay_uniform_ratio: float = 0.0
    replay_priority_epsilon: float = 1e-3
    use_cnn_feature_extractor: bool = False
    state_channels: int = 2
    state_antennas: int = 1
    state_delay_taps: int | None = None
    state_feature_dim: int = 256
    cnn_conv1_channels: int = 32
    cnn_conv2_channels: int = 64
    cnn_kernel_size_1: int = 5
    cnn_kernel_size_2: int = 3
    cnn_pool_size: int = 2
    target_entropy: float | None = None
    target_entropy_scale: float | None = 1.0
    alpha_min: float | None = None
    alpha_max: float | None = None
    device: str | None = None

    @property
    def grouped_action_dim(self) -> int:
        return self.grouped_rows * self.grouped_cols

    @property
    def physical_action_dim(self) -> int:
        return self.physical_rows * self.physical_cols


class ReplayBuffer:
    """CPU-only replay buffer backed by NumPy arrays."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int,
        dtype: np.dtype = np.float32,
        prioritized: bool = False,
        priority_alpha: float = 1.0,
        uniform_ratio: float = 0.0,
        priority_epsilon: float = 1e-3,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.dtype = dtype
        self.prioritized = bool(prioritized)
        self.priority_alpha = float(priority_alpha)
        self.uniform_ratio = float(np.clip(uniform_ratio, 0.0, 1.0))
        self.priority_epsilon = float(max(priority_epsilon, 1e-8))

        self.states = np.empty((self.capacity, self.state_dim), dtype=self.dtype)
        self.actions = np.empty((self.capacity, self.action_dim), dtype=self.dtype)
        self.rewards = np.empty((self.capacity, 1), dtype=self.dtype)
        self.next_states = np.empty((self.capacity, self.state_dim), dtype=self.dtype)
        self.dones = np.empty((self.capacity, 1), dtype=self.dtype)
        self.priorities = np.ones((self.capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float = 1.0,
    ) -> int:
        """Store one transition strictly on CPU."""
        state = np.asarray(state, dtype=self.dtype).reshape(-1)
        action = np.asarray(action, dtype=self.dtype).reshape(-1)
        next_state = np.asarray(next_state, dtype=self.dtype).reshape(-1)

        if state.size != self.state_dim:
            raise ValueError(
                f"Expected state_dim={self.state_dim}, but received {state.size}."
            )
        if action.size != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, but received {action.size}."
            )
        if next_state.size != self.state_dim:
            raise ValueError(
                f"Expected next_state_dim={self.state_dim}, but received {next_state.size}."
            )

        insert_index = self.ptr
        self.states[insert_index] = state
        self.actions[insert_index] = action
        self.rewards[insert_index, 0] = float(reward)
        self.next_states[insert_index] = next_state
        self.dones[insert_index, 0] = float(done)
        self.priorities[insert_index] = self._sanitize_priority(priority)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return insert_index

    def sample(self, batch_size: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        """Move only the sampled mini-batch to the target device."""
        if self.size < batch_size:
            raise ValueError(
                f"Not enough samples in buffer: size={self.size}, batch_size={batch_size}."
            )

        indices = self._sample_indices(batch_size)
        batch = {
            "states": torch.from_numpy(self.states[indices]),
            "actions": torch.from_numpy(self.actions[indices]),
            "rewards": torch.from_numpy(self.rewards[indices]),
            "next_states": torch.from_numpy(self.next_states[indices]),
            "dones": torch.from_numpy(self.dones[indices]),
        }
        return {key: value.to(device) for key, value in batch.items()}

    def update_priorities(
        self,
        indices: list[int] | np.ndarray,
        priorities: list[float] | np.ndarray,
    ) -> None:
        """Update replay priorities for an existing set of CPU-side samples."""
        if len(indices) != len(priorities):
            raise ValueError("`indices` and `priorities` must have the same length.")
        indices_np = np.asarray(indices, dtype=np.int64).reshape(-1)
        priorities_np = np.asarray(priorities, dtype=np.float32).reshape(-1)
        if np.any(indices_np < 0) or np.any(indices_np >= self.capacity):
            raise ValueError("Replay priority indices are out of range.")
        self.priorities[indices_np] = np.asarray(
            [self._sanitize_priority(value) for value in priorities_np],
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return self.size

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        if not self.prioritized:
            return np.random.randint(0, self.size, size=batch_size)

        active_priorities = self.priorities[: self.size].astype(np.float64, copy=False)
        active_priorities = np.maximum(active_priorities, self.priority_epsilon)
        weighted = active_priorities ** self.priority_alpha
        weighted_sum = float(np.sum(weighted))
        if not np.isfinite(weighted_sum) or weighted_sum <= 0.0:
            return np.random.randint(0, self.size, size=batch_size)

        probs = weighted / weighted_sum
        if self.uniform_ratio > 0.0:
            uniform_probs = np.full_like(probs, 1.0 / float(self.size))
            probs = (1.0 - self.uniform_ratio) * probs + self.uniform_ratio * uniform_probs

        return np.random.choice(self.size, size=batch_size, replace=True, p=probs)

    def _sanitize_priority(self, priority: float) -> float:
        priority = float(priority)
        if not np.isfinite(priority):
            priority = 1.0
        return float(max(priority, self.priority_epsilon))


class GroupedRISMapper:
    """Map grouped 10x10 policy actions back to the physical 100x100 RIS."""

    def __init__(
        self,
        grouped_rows: int = 10,
        grouped_cols: int = 10,
        physical_rows: int = 100,
        physical_cols: int = 100,
        phase_scale: float = float(np.pi),
    ) -> None:
        if physical_rows % grouped_rows != 0 or physical_cols % grouped_cols != 0:
            raise ValueError(
                "Physical RIS dimensions must be divisible by grouped dimensions."
            )

        self.grouped_rows = int(grouped_rows)
        self.grouped_cols = int(grouped_cols)
        self.physical_rows = int(physical_rows)
        self.physical_cols = int(physical_cols)
        self.phase_scale = float(phase_scale)
        self.tile_rows = self.physical_rows // self.grouped_rows
        self.tile_cols = self.physical_cols // self.grouped_cols
        self.grouped_action_dim = self.grouped_rows * self.grouped_cols
        self.physical_action_dim = self.physical_rows * self.physical_cols

    def grouped_to_phase_matrix(
        self,
        grouped_action: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        """Convert grouped actions in [-1, 1] to a tiled phase matrix in [-pi, pi]."""
        if isinstance(grouped_action, torch.Tensor):
            return self._grouped_to_phase_matrix_torch(grouped_action)
        return self._grouped_to_phase_matrix_numpy(grouped_action)

    def grouped_to_env_action(
        self,
        grouped_action: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        """Flatten the tiled physical phase matrix for env.step(action)."""
        phase_matrix = self.grouped_to_phase_matrix(grouped_action)
        return phase_matrix.reshape(*phase_matrix.shape[:-2], -1)

    def _grouped_to_phase_matrix_numpy(self, grouped_action: np.ndarray) -> np.ndarray:
        grouped_action = np.asarray(grouped_action, dtype=np.float32)
        if grouped_action.shape[-1] != self.grouped_action_dim:
            raise ValueError(
                f"Expected grouped action dim {self.grouped_action_dim}, "
                f"received {grouped_action.shape[-1]}."
            )

        grouped_grid = grouped_action.reshape(
            *grouped_action.shape[:-1], self.grouped_rows, self.grouped_cols
        )
        phase_grid = grouped_grid * self.phase_scale
        phase_grid = np.repeat(phase_grid, self.tile_rows, axis=-2)
        phase_grid = np.repeat(phase_grid, self.tile_cols, axis=-1)
        return phase_grid.astype(np.float32)

    def _grouped_to_phase_matrix_torch(self, grouped_action: torch.Tensor) -> torch.Tensor:
        if grouped_action.shape[-1] != self.grouped_action_dim:
            raise ValueError(
                f"Expected grouped action dim {self.grouped_action_dim}, "
                f"received {grouped_action.shape[-1]}."
            )

        grouped_grid = grouped_action.reshape(
            *grouped_action.shape[:-1], self.grouped_rows, self.grouped_cols
        )
        phase_grid = grouped_grid * self.phase_scale
        phase_grid = torch.repeat_interleave(phase_grid, self.tile_rows, dim=-2)
        phase_grid = torch.repeat_interleave(phase_grid, self.tile_cols, dim=-1)
        return phase_grid


class ChannelFeatureExtractor(nn.Module):
    """Compress a structured CIR tensor into a compact latent state vector.

    The input contract is `[batch, real_imag_channels, antennas, delay_taps]`.
    For replay and NumPy bridge compatibility, callers may still pass a flat
    `[batch, state_dim]` tensor. The extractor reshapes it internally.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        state_channels: int = 2,
        state_antennas: int = 1,
        state_delay_taps: int | None = None,
        output_dim: int = 256,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        kernel_size_1: int = 5,
        kernel_size_2: int = 3,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.state_channels = int(state_channels)
        self.state_antennas = int(state_antennas)
        self.state_delay_taps = self._resolve_delay_taps(state_delay_taps)
        self.output_dim = int(output_dim)
        self.pool_size = int(pool_size)

        kernel_size_1 = int(max(kernel_size_1, 1))
        kernel_size_2 = int(max(kernel_size_2, 1))
        padding_1 = kernel_size_1 // 2
        padding_2 = kernel_size_2 // 2

        self.conv1 = nn.Conv2d(
            in_channels=self.state_channels,
            out_channels=int(conv1_channels),
            kernel_size=(1, kernel_size_1),
            padding=(0, padding_1),
        )
        self.conv2 = nn.Conv2d(
            in_channels=int(conv1_channels),
            out_channels=int(conv2_channels),
            kernel_size=(1, kernel_size_2),
            padding=(0, padding_2),
        )
        self.pool = nn.MaxPool2d(
            kernel_size=(1, self.pool_size),
            stride=(1, self.pool_size),
        )

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                self.state_channels,
                self.state_antennas,
                self.state_delay_taps,
                dtype=torch.float32,
            )
            flattened_dim = int(self._forward_conv(dummy).reshape(1, -1).shape[-1])
        self.projection = nn.Linear(flattened_dim, self.output_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self._reshape_state(state)
        x = self._forward_conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.projection(x)

    def _forward_conv(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(state))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

    def _reshape_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim == 2:
            if state.shape[-1] != self.state_dim:
                raise ValueError(
                    f"Expected flattened state dim {self.state_dim}, "
                    f"received {state.shape[-1]}."
                )
            return state.reshape(
                state.shape[0],
                self.state_channels,
                self.state_antennas,
                self.state_delay_taps,
            )
        if state.ndim == 4:
            expected_shape = (
                self.state_channels,
                self.state_antennas,
                self.state_delay_taps,
            )
            if tuple(state.shape[-3:]) != expected_shape:
                raise ValueError(
                    "Structured CIR state shape mismatch. Expected trailing "
                    f"shape {expected_shape}, received {tuple(state.shape[-3:])}."
                )
            return state
        raise ValueError(
            "ChannelFeatureExtractor expects a flattened [batch, state_dim] or "
            "structured [batch, channels, antennas, delay_taps] tensor."
        )

    def _resolve_delay_taps(self, state_delay_taps: int | None) -> int:
        base = self.state_channels * self.state_antennas
        if base <= 0:
            raise ValueError("`state_channels` and `state_antennas` must be positive.")
        if state_delay_taps is None:
            if self.state_dim % base != 0:
                raise ValueError(
                    f"state_dim={self.state_dim} is not divisible by "
                    f"state_channels*state_antennas={base}."
                )
            delay_taps = self.state_dim // base
        else:
            delay_taps = int(state_delay_taps)
        if delay_taps <= 0:
            raise ValueError("`state_delay_taps` must be positive.")
        expected_dim = base * delay_taps
        if expected_dim != self.state_dim:
            raise ValueError(
                f"Expected state_dim={expected_dim} from channels={self.state_channels}, "
                f"antennas={self.state_antennas}, delay_taps={delay_taps}, "
                f"but got state_dim={self.state_dim}."
            )
        return delay_taps


class StateEncoder(nn.Module):
    """Encode either flat legacy states or structured CIR states for SAC."""

    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        self.use_cnn_feature_extractor = bool(config.use_cnn_feature_extractor)
        self.state_dim = int(config.state_dim)
        if self.use_cnn_feature_extractor:
            self.feature_extractor = ChannelFeatureExtractor(
                state_dim=config.state_dim,
                state_channels=config.state_channels,
                state_antennas=config.state_antennas,
                state_delay_taps=config.state_delay_taps,
                output_dim=config.state_feature_dim,
                conv1_channels=config.cnn_conv1_channels,
                conv2_channels=config.cnn_conv2_channels,
                kernel_size_1=config.cnn_kernel_size_1,
                kernel_size_2=config.cnn_kernel_size_2,
                pool_size=config.cnn_pool_size,
            )
            self.output_dim = int(config.state_feature_dim)
        else:
            self.feature_extractor = None
            self.output_dim = self.state_dim
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.feature_extractor is None:
            if state.ndim == 1:
                state = state.unsqueeze(0)
            x = state.reshape(state.shape[0], -1)
            if x.shape[-1] != self.state_dim:
                raise ValueError(
                    f"Expected flattened state dim {self.state_dim}, "
                    f"received {x.shape[-1]}."
                )
        else:
            x = self.feature_extractor(state)
        return self.output_norm(x)


class GaussianActor(nn.Module):
    """SAC actor with optional CNN CIR compression and LayerNorm."""

    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        self.state_encoder = StateEncoder(config)
        self.fc1 = nn.Linear(self.state_encoder.output_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.mean_head = nn.Linear(config.hidden_dim, config.grouped_action_dim)
        self.log_std_head = nn.Linear(config.hidden_dim, config.grouped_action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.state_encoder(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        raw_action = normal.rsample()
        squashed_action = torch.tanh(raw_action)

        log_prob = normal.log_prob(raw_action)
        log_prob = log_prob - torch.log(1.0 - squashed_action.pow(2) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        deterministic_action = torch.tanh(mean)
        return squashed_action, log_prob, deterministic_action


class QNetwork(nn.Module):
    """Single critic network with optional CNN CIR compression."""

    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        self.state_encoder = StateEncoder(config)
        self.fc1 = nn.Linear(
            self.state_encoder.output_dim + config.grouped_action_dim,
            config.hidden_dim,
        )
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc3 = nn.Linear(config.hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        encoded_state = self.state_encoder(state)
        x = torch.cat([encoded_state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TwinCritic(nn.Module):
    """Twin critics used by SAC."""

    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        self.q1 = QNetwork(config)
        self.q2 = QNetwork(config)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action), self.q2(state, action)


class SACAgent:
    """Soft Actor-Critic agent for grouped RIS control."""

    def __init__(self, config: SACConfig) -> None:
        self.config = config
        self.device = torch.device(
            config.device
            if config.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.mapper = GroupedRISMapper(
            grouped_rows=config.grouped_rows,
            grouped_cols=config.grouped_cols,
            physical_rows=config.physical_rows,
            physical_cols=config.physical_cols,
        )

        self.actor = GaussianActor(config).to(self.device)
        self.critic = TwinCritic(config).to(self.device)
        self.critic_target = TwinCritic(config).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for parameter in self.critic_target.parameters():
            parameter.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr,
        )

        init_alpha = float(config.init_alpha)
        if config.alpha_min is not None:
            init_alpha = max(init_alpha, float(config.alpha_min))
        if config.alpha_max is not None:
            init_alpha = min(init_alpha, float(config.alpha_max))

        init_log_alpha = np.log(init_alpha)
        self.log_alpha = torch.tensor(
            [init_log_alpha],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = (
            float(config.target_entropy)
            if config.target_entropy is not None
            else -float(
                config.grouped_action_dim
                if config.target_entropy_scale is None
                else config.target_entropy_scale * config.grouped_action_dim
            )
        )
        self.log_alpha_min = (
            None
            if config.alpha_min is None
            else float(np.log(max(float(config.alpha_min), 1e-8)))
        )
        self.log_alpha_max = (
            None
            if config.alpha_max is None
            else float(np.log(max(float(config.alpha_max), 1e-8)))
        )

        self.replay_buffer = ReplayBuffer(
            state_dim=config.state_dim,
            action_dim=config.grouped_action_dim,
            capacity=config.replay_capacity,
            prioritized=config.prioritized_replay,
            priority_alpha=config.replay_priority_alpha,
            uniform_ratio=config.replay_uniform_ratio,
            priority_epsilon=config.replay_priority_epsilon,
        )
        self.dual_replay = bool(config.dual_replay)
        self.hard_replay_ratio = float(np.clip(config.hard_replay_ratio, 0.0, 1.0))
        self.hard_replay_buffer: ReplayBuffer | None = None
        if self.dual_replay:
            hard_replay_capacity = (
                int(config.hard_replay_capacity)
                if config.hard_replay_capacity is not None
                else max(config.batch_size * 8, config.replay_capacity // 2)
            )
            self.hard_replay_buffer = ReplayBuffer(
                state_dim=config.state_dim,
                action_dim=config.grouped_action_dim,
                capacity=hard_replay_capacity,
                prioritized=config.hard_replay_prioritized,
                priority_alpha=config.hard_replay_priority_alpha,
                uniform_ratio=config.hard_replay_uniform_ratio,
                priority_epsilon=config.hard_replay_priority_epsilon,
            )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        return_grouped_action: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Return the flattened physical action and optionally the grouped action."""
        grouped_action = self.sample_grouped_action(
            state,
            deterministic=deterministic,
        )
        env_action = (
            self.mapper.grouped_to_env_action(grouped_action)
            .reshape(-1)
            .astype(np.float32)
        )

        if return_grouped_action:
            return env_action, grouped_action
        return env_action

    def sample_grouped_action_tensor(
        self,
        state_tensor: torch.Tensor,
        deterministic: bool = False,
        exploration_scale: float = 1.0,
        additive_noise_std: float = 0.0,
    ) -> torch.Tensor:
        """Sample a grouped action tensor on the agent device."""
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        state_tensor = state_tensor.to(self.device)
        self.actor.eval()
        with torch.no_grad():
            sampled_action, _, deterministic_action = self.actor.sample(state_tensor)
            if deterministic:
                action = deterministic_action
            else:
                action = deterministic_action + float(exploration_scale) * (
                    sampled_action - deterministic_action
                )
                if additive_noise_std > 0.0:
                    action = action + float(additive_noise_std) * torch.randn_like(action)
                action = action.clamp(-1.0, 1.0)
        self.actor.train()
        return action.squeeze(0)

    def sample_grouped_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Sample a 100-D grouped action in [-1, 1]."""
        state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0)
        action = self.sample_grouped_action_tensor(
            state_tensor,
            deterministic=deterministic,
        )
        return action.cpu().numpy().astype(np.float32)

    def expand_grouped_action(
        self,
        grouped_action: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        """Expand a grouped 100-D action into a physical 100x100 phase matrix."""
        return self.mapper.grouped_to_phase_matrix(grouped_action)

    def store_transition(
        self,
        state: np.ndarray,
        grouped_action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float = 1.0,
    ) -> int:
        """Store grouped actions only, never the expanded 10000-D physical action."""
        return self.replay_buffer.add(
            state,
            grouped_action,
            reward,
            next_state,
            done,
            priority=priority,
        )

    def update_replay_priorities(
        self,
        indices: list[int] | np.ndarray,
        priorities: list[float] | np.ndarray,
    ) -> None:
        self.replay_buffer.update_priorities(indices, priorities)

    def store_hard_transition(
        self,
        state: np.ndarray,
        grouped_action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float = 1.0,
    ) -> int | None:
        if self.hard_replay_buffer is None:
            return None
        return self.hard_replay_buffer.add(
            state,
            grouped_action,
            reward,
            next_state,
            done,
            priority=priority,
        )

    def update_hard_replay_priorities(
        self,
        indices: list[int] | np.ndarray,
        priorities: list[float] | np.ndarray,
    ) -> None:
        if self.hard_replay_buffer is None:
            return
        self.hard_replay_buffer.update_priorities(indices, priorities)

    def ready(self) -> bool:
        return len(self.replay_buffer) >= self.config.batch_size

    def update(
        self,
        *,
        update_actor: bool = True,
        update_alpha: bool | None = None,
        update_target: bool | None = None,
    ) -> dict[str, float | None]:
        """Perform one SAC update step from a CPU-resident replay buffer."""
        if not self.ready():
            raise ValueError(
                f"Replay buffer has {len(self.replay_buffer)} samples, "
                f"need at least {self.config.batch_size}."
            )
        if update_alpha is None:
            update_alpha = update_actor
        if update_target is None:
            update_target = update_actor

        batch = self._sample_training_batch()
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target_value = rewards + (1.0 - dones) * self.config.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        alpha_loss = None
        alpha_value = float(self.alpha.detach().cpu().item())
        mean_q = None
        mean_log_prob = None
        actor_updated = 0.0

        if update_actor:
            new_actions, log_prob, _ = self.actor.sample(states)
            q1_pi, q2_pi = self.critic(states, new_actions)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha.detach() * log_prob - min_q_pi).mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_updated = 1.0

            mean_q = float(min_q_pi.detach().mean().cpu().item())
            mean_log_prob = float(log_prob.detach().mean().cpu().item())

            if update_alpha:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self._clamp_log_alpha()
                alpha_value = float(self.alpha.detach().cpu().item())

        if update_target:
            self._soft_update_target_network()

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": None if actor_loss is None else float(actor_loss.detach().cpu().item()),
            "alpha_loss": None if alpha_loss is None else float(alpha_loss.detach().cpu().item()),
            "alpha": alpha_value,
            "mean_q": mean_q,
            "mean_log_prob": mean_log_prob,
            "actor_updated": actor_updated,
        }

    def save(self, path: str) -> None:
        """Save the trainable SAC state."""
        payload = {
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path: str, strict: bool = True) -> None:
        """Load a saved SAC state."""
        payload: dict[str, Any] = torch.load(path, map_location=self.device)
        try:
            self.actor.load_state_dict(payload["actor"], strict=strict)
            self.critic.load_state_dict(payload["critic"], strict=strict)
            self.critic_target.load_state_dict(payload["critic_target"], strict=strict)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint architecture is incompatible with the current SACAgent "
                "configuration. This can happen when enabling the CNN CIR feature "
                "extractor or changing the grouped/physical state contract."
            ) from exc
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
        self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])

    def _soft_update_target_network(self) -> None:
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic.parameters(),
        ):
            target_param.data.mul_(1.0 - self.config.tau)
            target_param.data.add_(self.config.tau * param.data)

    def _clamp_log_alpha(self) -> None:
        with torch.no_grad():
            if self.log_alpha_min is not None:
                self.log_alpha.data.clamp_(min=self.log_alpha_min)
            if self.log_alpha_max is not None:
                self.log_alpha.data.clamp_(max=self.log_alpha_max)

    def _sample_training_batch(self) -> dict[str, torch.Tensor]:
        if not self.dual_replay or self.hard_replay_buffer is None or len(self.hard_replay_buffer) == 0:
            return self.replay_buffer.sample(self.config.batch_size, self.device)

        hard_batch_size = int(round(self.config.batch_size * self.hard_replay_ratio))
        hard_batch_size = min(hard_batch_size, len(self.hard_replay_buffer), self.config.batch_size)
        normal_batch_size = self.config.batch_size - hard_batch_size

        if normal_batch_size <= 0:
            return self.hard_replay_buffer.sample(self.config.batch_size, self.device)

        normal_batch = self.replay_buffer.sample(normal_batch_size, self.device)
        if hard_batch_size == 0:
            return normal_batch

        hard_batch = self.hard_replay_buffer.sample(hard_batch_size, self.device)
        merged = {
            key: torch.cat([normal_batch[key], hard_batch[key]], dim=0)
            for key in normal_batch
        }
        permutation = torch.randperm(self.config.batch_size, device=self.device)
        return {key: value[permutation] for key, value in merged.items()}
