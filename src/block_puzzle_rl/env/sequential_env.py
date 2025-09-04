from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from block_puzzle_rl.blockpuzzle import BlockPuzzleGame, GameConfig, GameAnalytics


class SequentialPlacementEnv(gym.Env):
    """
    Sequential piece placement environment with a small, fixed discrete action space.

    Actions (11 total):
      0: Select Piece 0
      1: Select Piece 1
      2: Select Piece 2
      3: Move Left
      4: Move Right
      5: Move Up
      6: Move Down
      7: Rotate CW
      8: Rotate CCW
      9: Place
     10: Reset (return ghost to default spawn position)

    Notes:
    - A "ghost" placement state is maintained after selecting a piece. All moves/rotations keep
      the ghost position valid; invalid actions are masked out and penalized if taken.
    - Selecting a piece initializes the ghost at the top-left-most valid placement at rotation 0.
    - Reset repositions the ghost back to that default spawn for the currently selected piece.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Action indices
    ACT_SELECT_0 = 0
    ACT_SELECT_1 = 1
    ACT_SELECT_2 = 2
    ACT_MOVE_LEFT = 3
    ACT_MOVE_RIGHT = 4
    ACT_MOVE_UP = 5
    ACT_MOVE_DOWN = 6
    ACT_ROTATE_CW = 7
    ACT_ROTATE_CCW = 8
    ACT_PLACE = 9
    ACT_RESET = 10

    def __init__(
        self,
        config: Optional[GameConfig] = None,
        render_mode: Optional[str] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        invalid_action_penalty: float = -0.1,
        step_penalty: float = 0.0,
        terminal_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        self.game = BlockPuzzleGame(config)
        self.render_mode = render_mode

        # Reward shaping parameters
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.step_penalty = float(step_penalty)
        self.terminal_penalty = float(terminal_penalty)
        self.reward_weights: Dict[str, float] = {
            # Positive components
            "cells": 0.05,
            "lines": 10.0,
            "lines_sq": 5.0,
            # Negative components (penalize increases)
            "holes": 0.1,
            "bumpiness": 0.01,
            "height": 0.02,
        }
        if reward_weights:
            self.reward_weights.update({k: float(v) for k, v in reward_weights.items()})

        size = self.game.config.grid_size
        k = self.game.config.pieces_per_set

        # Observation space: base + ghost selection state
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0, high=1, shape=(size, size), dtype=np.int8),
                "pieces": spaces.Box(low=-1, high=6, shape=(k,), dtype=np.int8),
                "pieces_remaining": spaces.Discrete(k + 1),
                # Ghost information; when not selected, indices take sentinel values
                # ghost_piece_idx in [0..k-1] or k if none
                "ghost_piece_idx": spaces.Discrete(k + 1),
                # x,y in [0..size-1] or size if none
                "ghost_x": spaces.Discrete(size + 1),
                "ghost_y": spaces.Discrete(size + 1),
                # rotation in [0..3]
                "ghost_rot": spaces.Discrete(4),
                # whether the ghost is currently valid
                "ghost_valid": spaces.Discrete(2),
            }
        )

        # Discrete action space (fixed size)
        self.action_space = spaces.Discrete(11)

        # Internal state
        self._last_obs: Optional[Dict[str, Any]] = None
        self._steps = 0  # environment-level steps (includes moves)

        # Ghost state
        self._ghost_piece_idx: Optional[int] = None
        self._ghost_x: Optional[int] = None
        self._ghost_y: Optional[int] = None
        self._ghost_rot: int = 0
        # Default spawn cache for the currently selected piece (x,y,r=0)
        self._default_spawn: Optional[Tuple[int, int]] = None

        # Rendering state (lazy)
        self._surface = None

    # ---------- Helpers ----------
    def _get_obs(self) -> Dict[str, Any]:
        size = self.game.config.grid_size
        k = self.game.config.pieces_per_set
        grid = self.game.grid.grid.astype(np.int8)
        pieces = np.full((k,), -1, dtype=np.int8)
        types = self.game.get_current_piece_types()
        for i, t in enumerate(types[:k]):
            pieces[i] = int(t)

        ghost_idx = self._ghost_piece_idx if self._ghost_piece_idx is not None else k
        ghost_x = self._ghost_x if self._ghost_x is not None else size
        ghost_y = self._ghost_y if self._ghost_y is not None else size
        ghost_rot = self._ghost_rot
        ghost_valid = 0
        if self._ghost_piece_idx is not None and self._ghost_x is not None and self._ghost_y is not None:
            ghost_valid = int(self._is_valid(self._ghost_piece_idx, self._ghost_x, self._ghost_y, self._ghost_rot))

        obs: Dict[str, Any] = {
            "grid": grid,
            "pieces": pieces,
            "pieces_remaining": len(self.game.current_pieces),
            "ghost_piece_idx": int(ghost_idx),
            "ghost_x": int(ghost_x),
            "ghost_y": int(ghost_y),
            "ghost_rot": int(ghost_rot),
            "ghost_valid": int(ghost_valid),
        }
        return obs

    def _compute_action_mask(self) -> np.ndarray:
        mask = np.zeros((self.action_space.n,), dtype=np.bool_)

        k = self.game.config.pieces_per_set
        size = self.game.config.grid_size

        # Selection actions: only allow selecting indices that exist and have a valid spawn
        for idx, action_idx in enumerate([self.ACT_SELECT_0, self.ACT_SELECT_1, self.ACT_SELECT_2]):
            if idx < len(self.game.current_pieces):
                spawn = self._find_default_spawn(idx)
                mask[action_idx] = spawn is not None
            else:
                mask[action_idx] = False

        # If no ghost, movement/rotate/place/reset are disabled
        if self._ghost_piece_idx is None or self._ghost_x is None or self._ghost_y is None:
            return mask

        # Movement validity
        cur_idx = int(self._ghost_piece_idx)
        cur_x = int(self._ghost_x)
        cur_y = int(self._ghost_y)
        cur_rot = int(self._ghost_rot)

        mask[self.ACT_MOVE_LEFT] = self._is_valid(cur_idx, cur_x - 1, cur_y, cur_rot)
        mask[self.ACT_MOVE_RIGHT] = self._is_valid(cur_idx, cur_x + 1, cur_y, cur_rot)
        mask[self.ACT_MOVE_UP] = self._is_valid(cur_idx, cur_x, cur_y - 1, cur_rot)
        mask[self.ACT_MOVE_DOWN] = self._is_valid(cur_idx, cur_x, cur_y + 1, cur_rot)

        # Rotations
        mask[self.ACT_ROTATE_CW] = self._is_valid(cur_idx, cur_x, cur_y, (cur_rot + 1) % 4)
        mask[self.ACT_ROTATE_CCW] = self._is_valid(cur_idx, cur_x, cur_y, (cur_rot - 1) % 4)

        # Place: valid if current ghost placement is valid
        mask[self.ACT_PLACE] = self._is_valid(cur_idx, cur_x, cur_y, cur_rot)

        # Reset: allowed if a default spawn exists for the current piece
        mask[self.ACT_RESET] = self._find_default_spawn(cur_idx) is not None

        return mask

    def _get_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "action_mask": self._compute_action_mask(),
            "score": self.game.score,
            "steps": self.game.step_count,
        }
        return info

    def _is_valid(self, piece_idx: int, x: int, y: int, r: int) -> bool:
        if piece_idx < 0 or piece_idx >= len(self.game.current_pieces):
            return False
        shape = self.game.current_pieces[piece_idx].get_shape(r)
        return bool(self.game.grid.is_valid_placement(shape, x, y))

    def _find_default_spawn(self, piece_idx: int) -> Optional[Tuple[int, int]]:
        if piece_idx < 0 or piece_idx >= len(self.game.current_pieces):
            return None
        shape0 = self.game.current_pieces[piece_idx].get_shape(0)
        # Top-left-most valid: prioritize lower y, then lower x
        best: Optional[Tuple[int, int]] = None
        for x in range(self.game.grid.size):
            for y in range(self.game.grid.size):
                if self.game.grid.is_valid_placement(shape0, x, y):
                    if best is None or (y < best[1] or (y == best[1] and x < best[0])):
                        best = (x, y)
        return best

    def _select_piece(self, piece_idx: int) -> bool:
        spawn = self._find_default_spawn(piece_idx)
        if spawn is None:
            return False
        self._ghost_piece_idx = int(piece_idx)
        self._ghost_rot = 0
        self._ghost_x, self._ghost_y = int(spawn[0]), int(spawn[1])
        self._default_spawn = (self._ghost_x, self._ghost_y)
        return True

    # ---------- Gym API ----------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self.game.reset(seed)
        self._steps = 0
        # Clear ghost state until a piece is selected
        self._ghost_piece_idx = None
        self._ghost_x = None
        self._ghost_y = None
        self._ghost_rot = 0
        self._default_spawn = None

        obs = self._get_obs()
        info = self._get_info()
        self._last_obs = obs
        return obs, info

    def step(self, action: int):
        action = int(action)

        terminated = False
        truncated = False

        reward_components: Dict[str, float] = {}

        # Non-placement actions receive step penalty; invalid actions receive invalid penalty
        took_valid_action = False

        # Handle selection actions
        if action in (self.ACT_SELECT_0, self.ACT_SELECT_1, self.ACT_SELECT_2):
            select_idx = action - self.ACT_SELECT_0
            took_valid_action = self._select_piece(select_idx)
            if not took_valid_action:
                reward_components["invalid"] = self.invalid_action_penalty
            else:
                reward_components["step"] = self.step_penalty

        # Movement actions
        elif action in (self.ACT_MOVE_LEFT, self.ACT_MOVE_RIGHT, self.ACT_MOVE_UP, self.ACT_MOVE_DOWN):
            if self._ghost_piece_idx is not None and self._ghost_x is not None and self._ghost_y is not None:
                dx = -1 if action == self.ACT_MOVE_LEFT else 1 if action == self.ACT_MOVE_RIGHT else 0
                dy = -1 if action == self.ACT_MOVE_UP else 1 if action == self.ACT_MOVE_DOWN else 0
                new_x = int(self._ghost_x + dx)
                new_y = int(self._ghost_y + dy)
                if self._is_valid(self._ghost_piece_idx, new_x, new_y, self._ghost_rot):
                    self._ghost_x, self._ghost_y = new_x, new_y
                    took_valid_action = True
                if took_valid_action:
                    reward_components["step"] = self.step_penalty
                else:
                    reward_components["invalid"] = self.invalid_action_penalty
            else:
                reward_components["invalid"] = self.invalid_action_penalty

        # Rotation actions
        elif action in (self.ACT_ROTATE_CW, self.ACT_ROTATE_CCW):
            if self._ghost_piece_idx is not None and self._ghost_x is not None and self._ghost_y is not None:
                new_rot = (self._ghost_rot + (1 if action == self.ACT_ROTATE_CW else -1)) % 4
                if self._is_valid(self._ghost_piece_idx, int(self._ghost_x), int(self._ghost_y), new_rot):
                    self._ghost_rot = int(new_rot)
                    took_valid_action = True
                if took_valid_action:
                    reward_components["step"] = self.step_penalty
                else:
                    reward_components["invalid"] = self.invalid_action_penalty
            else:
                reward_components["invalid"] = self.invalid_action_penalty

        # Place action
        elif action == self.ACT_PLACE:
            if self._ghost_piece_idx is not None and self._ghost_x is not None and self._ghost_y is not None:
                piece_idx = int(self._ghost_piece_idx)
                x = int(self._ghost_x)
                y = int(self._ghost_y)
                r = int(self._ghost_rot)

                # Features before placement
                features_before = GameAnalytics.get_board_features(self.game.grid.grid)

                # Estimate number of cells for shaping
                cells_in_piece = 0
                if 0 <= piece_idx < len(self.game.current_pieces):
                    shape = self.game.current_pieces[piece_idx].get_shape(r)
                    cells_in_piece = int(np.sum(shape))

                success, gained, lines = self.game.place_piece(piece_idx, x, y, r)
                if success:
                    took_valid_action = True

                    # Features after
                    features_after = GameAnalytics.get_board_features(self.game.grid.grid)

                    # Base reward components
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

                    # After placement, clear ghost selection
                    self._ghost_piece_idx = None
                    self._ghost_x = None
                    self._ghost_y = None
                    self._ghost_rot = 0
                    self._default_spawn = None
                else:
                    reward_components["invalid"] = self.invalid_action_penalty
            else:
                reward_components["invalid"] = self.invalid_action_penalty

        # Reset action
        elif action == self.ACT_RESET:
            if self._ghost_piece_idx is not None:
                piece_idx = int(self._ghost_piece_idx)
                spawn = self._find_default_spawn(piece_idx)
                if spawn is not None:
                    self._ghost_rot = 0
                    self._ghost_x, self._ghost_y = int(spawn[0]), int(spawn[1])
                    self._default_spawn = (self._ghost_x, self._ghost_y)
                    took_valid_action = True
                    reward_components["step"] = self.step_penalty
                else:
                    reward_components["invalid"] = self.invalid_action_penalty
            else:
                reward_components["invalid"] = self.invalid_action_penalty

        else:
            # Unknown action index
            reward_components["invalid"] = self.invalid_action_penalty

        # Terminal shaping
        terminated = bool(self.game.game_over)
        self._steps += 1
        if self._steps >= self.game.config.max_episode_steps:
            truncated = True
        if terminated:
            reward_components["terminal"] = self.terminal_penalty

        reward = float(sum(reward_components.values()))

        obs = self._get_obs()
        info = self._get_info()
        info["reward_components"] = reward_components
        # For placement steps, engine_score_delta is reflected via game.score increment; expose delta on success
        info["engine_score_delta"] = float(gained if "gained" in locals() and took_valid_action and action == self.ACT_PLACE else 0.0)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    # Mask exposure for wrappers/MaskablePPO
    def get_action_mask(self) -> np.ndarray:
        return self._compute_action_mask().astype(np.bool_)

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

