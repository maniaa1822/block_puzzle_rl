from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from block_puzzle_rl.blockpuzzle import BlockPuzzleGame, GameConfig, GameAnalytics


def _compute_action_mask(game: BlockPuzzleGame) -> np.ndarray:
    size = game.grid.size
    k = game.config.pieces_per_set
    mask = np.zeros((k, size, size, 4), dtype=np.bool_)
    valid = game.get_valid_actions()  # list of (piece_idx, x, y, rotation)
    for piece_idx, x, y, r in valid:
        if 0 <= piece_idx < k and 0 <= x < size and 0 <= y < size and 0 <= r < 4:
            mask[piece_idx, y, x, r] = True
    return mask


class BlockPuzzleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Optional[GameConfig] = None, render_mode: Optional[str] = None,
                 reward_weights: Optional[Dict[str, float]] = None,
                 invalid_action_penalty: float = -0.1,
                 step_penalty: float = 0.0,
                 terminal_penalty: float = 0.0) -> None:
        super().__init__()
        self.game = BlockPuzzleGame(config)
        self.render_mode = render_mode

        # Reward shaping parameters
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.step_penalty = float(step_penalty)
        self.terminal_penalty = float(terminal_penalty)
        self.reward_weights: Dict[str, float] = {
            # Positive components
            "cells": 0.05,            # reward per cell placed
            "lines": 10.0,           # reward per line cleared
            "lines_sq": 5.0,         # extra for multiple lines (quadratic)
            # Negative components (penalize increases)
            "holes": 0.1,            # penalize holes created
            "bumpiness": 0.01,       # penalize bumpiness increase
            "height": 0.02,          # penalize max height increase
        }
        if reward_weights:
            self.reward_weights.update({k: float(v) for k, v in reward_weights.items()})

        size = self.game.config.grid_size
        k = self.game.config.pieces_per_set

        # Observation space: grid (0/1) and current pieces (indices, -1 for empty)
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0, high=1, shape=(size, size), dtype=np.int8),
                "pieces": spaces.Box(low=-1, high=6, shape=(k,), dtype=np.int8),
                "pieces_remaining": spaces.Discrete(k + 1),
            }
        )

        # Action: (piece_idx, x, y, rotation)
        self.action_space = spaces.MultiDiscrete((k, size, size, 4))

        self._last_obs: Optional[Dict[str, Any]] = None
        self._steps = 0

        # Rendering state (lazy)
        self._surface = None

    def _get_obs(self) -> Dict[str, Any]:
        size = self.game.config.grid_size
        k = self.game.config.pieces_per_set
        grid = self.game.grid.grid.astype(np.int8)
        pieces = np.full((k,), -1, dtype=np.int8)
        types = self.game.get_current_piece_types()
        for i, t in enumerate(types[:k]):
            pieces[i] = int(t)
        obs: Dict[str, Any] = {
            "grid": grid,
            "pieces": pieces,
            "pieces_remaining": len(self.game.current_pieces),
        }
        return obs

    def _get_info(self) -> Dict[str, Any]:
        mask = _compute_action_mask(self.game)
        valid_actions = self.game.get_valid_actions()
        info: Dict[str, Any] = {
            "action_mask": mask,
            "valid_actions": valid_actions,
            "score": self.game.score,
            "steps": self.game.step_count,
        }
        return info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self.game.reset(seed)
        self._steps = 0
        obs = self._get_obs()
        info = self._get_info()
        self._last_obs = obs
        return obs, info

    def step(self, action: np.ndarray | Tuple[int, int, int, int]):
        piece_idx, x, y, r = map(int, action)

        terminated = False
        truncated = False

        # Features before placement
        features_before = GameAnalytics.get_board_features(self.game.grid.grid)

        # Estimate number of cells in the chosen piece at rotation r (for shaping)
        cells_in_piece = 0
        if 0 <= piece_idx < len(self.game.current_pieces):
            shape = self.game.current_pieces[piece_idx].get_shape(r)
            cells_in_piece = int(np.sum(shape))

        # Attempt placement
        success, gained, lines = self.game.place_piece(piece_idx, x, y, r)

        # Features after
        features_after = GameAnalytics.get_board_features(self.game.grid.grid)

        # Base reward components
        reward_components: Dict[str, float] = {}
        if success:
            reward_components["cells"] = self.reward_weights["cells"] * float(cells_in_piece)
            reward_components["lines"] = self.reward_weights["lines"] * float(lines)
            reward_components["lines_sq"] = self.reward_weights["lines_sq"] * float(lines * lines)
            # Penalize increases in undesirable features
            reward_components["holes"] = -self.reward_weights["holes"] * float(
                max(0, features_after["holes"] - features_before["holes"]) )
            reward_components["bumpiness"] = -self.reward_weights["bumpiness"] * float(
                max(0, features_after["bumpiness"] - features_before["bumpiness"]) )
            reward_components["height"] = -self.reward_weights["height"] * float(
                max(0, features_after["max_height"] - features_before["max_height"]) )
        else:
            reward_components["invalid"] = self.invalid_action_penalty

        # Step and terminal shaping
        reward_components["step"] = self.step_penalty
        terminated = bool(self.game.game_over)
        self._steps += 1
        if self.game.step_count >= self.game.config.max_episode_steps:
            truncated = True
        if terminated:
            reward_components["terminal"] = self.terminal_penalty

        reward = float(sum(reward_components.values()))

        obs = self._get_obs()
        info = self._get_info()
        info["reward_components"] = reward_components
        info["engine_score_delta"] = float(gained if success else 0.0)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            # Create a simple RGB image from the grid
            grid = self._last_obs["grid"] if self._last_obs is not None else self.game.grid.grid
            cell = 12
            h, w = grid.shape
            img = np.zeros((h * cell, w * cell, 3), dtype=np.uint8)
            for y in range(h):
                for x in range(w):
                    color = (70, 200, 120) if grid[y, x] else (30, 30, 36)
                    img[y * cell : (y + 1) * cell, x * cell : (x + 1) * cell, :] = color
            return img
        # human rendering delegated to external UI; noop
        return None

    def close(self) -> None:
        pass


