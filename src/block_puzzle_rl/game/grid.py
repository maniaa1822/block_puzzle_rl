from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


Coordinate = Tuple[int, int]


@dataclass
class PlacementResult:
    lines_cleared: int
    game_over: bool


class GameGrid:
    """Discrete 2D grid for block placement.

    The grid uses 0 for empty cells and positive integers for filled cells.
    Integer values correspond to tetromino indices for optional coloring.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)

    def reset(self) -> None:
        self.grid.fill(0)

    def is_inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def can_place(self, cells: Iterable[Coordinate]) -> bool:
        for x, y in cells:
            if not self.is_inside(x, y):
                return False
            if self.grid[y, x] != 0:
                return False
        return True

    def place(self, cells: Iterable[Coordinate], value: int) -> PlacementResult:
        """Place cells with `value`, clear lines, and return result."""
        for x, y in cells:
            if not self.is_inside(x, y) or self.grid[y, x] != 0:
                # Illegal placement implies game over when spawning overlaps
                return PlacementResult(lines_cleared=0, game_over=True)
        for x, y in cells:
            self.grid[y, x] = value
        lines = self._clear_full_lines()
        return PlacementResult(lines_cleared=lines, game_over=False)

    def _clear_full_lines(self) -> int:
        full_rows = np.where(np.all(self.grid != 0, axis=1))[0]
        if full_rows.size == 0:
            return 0
        num = int(full_rows.size)
        # Remove full rows and add empty rows at the top
        self.grid = np.delete(self.grid, full_rows, axis=0)
        new_rows = np.zeros((num, self.width), dtype=np.int8)
        self.grid = np.vstack((new_rows, self.grid))
        # Ensure height remains constant
        if self.grid.shape[0] != self.height:
            # Pad or trim if numerical issues occur
            self.grid = self.grid[-self.height :]
        return num

    def get_max_height(self) -> int:
        # y=0 is top; find first non-empty from top
        non_empty_rows = np.where(np.any(self.grid != 0, axis=1))[0]
        if non_empty_rows.size == 0:
            return 0
        top_index = int(non_empty_rows[0])
        return self.height - top_index

    def count_holes(self) -> int:
        holes = 0
        for x in range(self.width):
            column = self.grid[:, x]
            seen_block = False
            for cell in column:
                if cell != 0:
                    seen_block = True
                elif seen_block:
                    holes += 1
        return holes

    def clone_state(self) -> np.ndarray:
        return self.grid.copy()



