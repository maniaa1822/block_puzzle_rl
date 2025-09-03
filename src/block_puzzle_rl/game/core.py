from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

import numpy as np

from .grid import GameGrid
from .pieces import Piece, TetrominoType
from .rules import ScoringRules


class Action(IntEnum):
    LEFT = 0
    RIGHT = 1
    ROTATE_CW = 2
    ROTATE_CCW = 3
    SOFT_DROP = 4
    HARD_DROP = 5
    NONE = 6


@dataclass
class GameConfig:
    width: int = 10
    height: int = 20
    random_seed: Optional[int] = None
    spawn_y: int = 0


class BlockPuzzleGame:
    def __init__(self, config: Optional[GameConfig] = None, rules: Optional[ScoringRules] = None) -> None:
        self.config = config or GameConfig()
        self.rules = rules or ScoringRules()
        self.rng = random.Random(self.config.random_seed)
        self.grid = GameGrid(self.config.width, self.config.height)
        self.score = 0
        self.lines_cleared_total = 0
        self.game_over = False
        self.current_piece: Optional[Piece] = None
        self.current_x = 0
        self.current_y = 0
        self.reset()

    def reset(self) -> None:
        self.grid.reset()
        self.score = 0
        self.lines_cleared_total = 0
        self.game_over = False
        self._spawn_piece()

    def _random_piece(self) -> Piece:
        kind = self.rng.choice(list(TetrominoType))
        return Piece(kind=kind, rotation=0)

    def _spawn_piece(self) -> None:
        self.current_piece = self._random_piece()
        s = self.current_piece.shape()
        h, w = s.shape
        self.current_x = (self.grid.width - w) // 2
        self.current_y = self.config.spawn_y
        # Immediate collision check: if overlaps, game over
        if not self.grid.can_place(self.current_piece.cells_at(self.current_x, self.current_y)):
            self.game_over = True

    def _move(self, dx: int, dy: int) -> None:
        if self.current_piece is None:
            return
        new_x = self.current_x + dx
        new_y = self.current_y + dy
        if self.grid.can_place(self.current_piece.cells_at(new_x, new_y)):
            self.current_x = new_x
            self.current_y = new_y

    def _rotate(self, delta: int) -> None:
        if self.current_piece is None:
            return
        rotated = self.current_piece.rotated(delta)
        if self.grid.can_place(rotated.cells_at(self.current_x, self.current_y)):
            self.current_piece = rotated

    def _lock_piece(self) -> Tuple[int, bool]:
        assert self.current_piece is not None
        value = int(self.current_piece.kind)
        result = self.grid.place(self.current_piece.cells_at(self.current_x, self.current_y), value)
        lines = result.lines_cleared
        self.lines_cleared_total += lines
        self.score += self.rules.score_for_lines(lines) + self.rules.placement_score
        if result.game_over:
            self.score -= self.rules.game_over_penalty
            self.game_over = True
        return lines, result.game_over

    def hard_drop(self) -> None:
        if self.current_piece is None:
            return
        # Drop until collision
        while True:
            next_y = self.current_y + 1
            if self.grid.can_place(self.current_piece.cells_at(self.current_x, next_y)):
                self.current_y = next_y
            else:
                break
        self._lock_piece()
        if not self.game_over:
            self._spawn_piece()

    def step(self, action: Action) -> Tuple[np.ndarray, int, bool, dict]:
        if self.game_over:
            return self.get_state(), 0, True, {}

        if action == Action.LEFT:
            self._move(-1, 0)
        elif action == Action.RIGHT:
            self._move(1, 0)
        elif action == Action.ROTATE_CW:
            self._rotate(1)
        elif action == Action.ROTATE_CCW:
            self._rotate(-1)
        elif action == Action.SOFT_DROP:
            # Try move down, otherwise lock
            next_y = self.current_y + 1
            if self.grid.can_place(self.current_piece.cells_at(self.current_x, next_y)):
                self.current_y = next_y
            else:
                self._lock_piece()
                if not self.game_over:
                    self._spawn_piece()
        elif action == Action.HARD_DROP:
            self.hard_drop()
        elif action == Action.NONE:
            pass

        obs = self.get_state()
        done = self.game_over
        reward = 0  # Reward shaping will be in env wrapper; keep engine score as info
        info = {
            "score": self.score,
            "lines_cleared_total": self.lines_cleared_total,
        }
        return obs, reward, done, info

    def get_state(self) -> np.ndarray:
        # Overlay current piece on a copy of the grid for observation
        state = self.grid.clone_state()
        if self.current_piece is not None and not self.game_over:
            for x, y in self.current_piece.cells_at(self.current_x, self.current_y):
                if 0 <= y < self.grid.height and 0 <= x < self.grid.width:
                    # Use negative to indicate falling piece overlay
                    state[y, x] = -int(self.current_piece.kind)
        return state



