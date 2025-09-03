from __future__ import annotations

from typing import Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .block_puzzle_env import _compute_action_mask


class FlattenDiscreteActionWrapper(gym.ActionWrapper):
    """Flattens MultiDiscrete (pieces, y, x, r) -> Discrete(N) for PPO.

    Also exposes `get_action_mask()` returning a 1D boolean mask of shape (N,).
    Order: piece, y, x, r (C-order flattening).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.MultiDiscrete)
        k, size_y, size_x, rotations = map(int, env.action_space.nvec)
        assert size_x == size_y, "Expected square grid"
        self.k = k
        self.size = size_x
        self.rot = rotations
        self.n = int(k * self.size * self.size * self.rot)
        self.action_space = spaces.Discrete(self.n)

    def _unflatten(self, idx: int) -> tuple[int, int, int, int]:
        r = idx % self.rot
        idx //= self.rot
        x = idx % self.size
        idx //= self.size
        y = idx % self.size
        piece = idx // self.size
        return int(piece), int(x), int(y), int(r)

    def action(self, action: int):  # type: ignore[override]
        return np.array(self._unflatten(int(action)), dtype=np.int64)

    def get_action_mask(self) -> np.ndarray:
        mask4d = _compute_action_mask(self.env.unwrapped.game)
        return mask4d.reshape(-1)


class ResampleInvalidActionWrapper(gym.Wrapper):
    """If a sampled action is invalid, resample uniformly among valid ones.

    Useful when training with vanilla PPO (no action masking).
    """

    def step(self, action):  # type: ignore[override]
        if isinstance(self.action_space, spaces.Discrete) and hasattr(self, "get_action_mask"):
            mask = self.get_action_mask()  # type: ignore[attr-defined]
            if 0 <= int(action) < mask.shape[0] and not bool(mask[int(action)]):
                valid_idxs = np.flatnonzero(mask)
                if valid_idxs.size > 0:
                    action = int(np.random.choice(valid_idxs))
        return self.env.step(action)

    # Delegate mask access if the wrapped env provides it
    def get_action_mask(self) -> np.ndarray:  # type: ignore[override]
        if hasattr(self.env, "get_action_mask"):
            return getattr(self.env, "get_action_mask")()
        raise AttributeError("Underlying env does not provide get_action_mask")


