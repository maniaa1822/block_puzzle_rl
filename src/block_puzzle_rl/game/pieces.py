from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple

import numpy as np


class TetrominoType(IntEnum):
    I = 1
    O = 2
    T = 3
    S = 4
    Z = 5
    J = 6
    L = 7


Shape = np.ndarray


def _rot90(shape: Shape, k: int) -> Shape:
    k = k % 4
    if k == 0:
        return shape
    return np.rot90(shape, k, axes=(1, 0))  # rotate clockwise when k>0


BASE_SHAPES = {
    TetrominoType.I: np.array([[1, 1, 1, 1]], dtype=np.int8),
    TetrominoType.O: np.array([[1, 1], [1, 1]], dtype=np.int8),
    TetrominoType.T: np.array([[1, 1, 1], [0, 1, 0]], dtype=np.int8),
    TetrominoType.S: np.array([[0, 1, 1], [1, 1, 0]], dtype=np.int8),
    TetrominoType.Z: np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8),
    TetrominoType.J: np.array([[1, 0, 0], [1, 1, 1]], dtype=np.int8),
    TetrominoType.L: np.array([[0, 0, 1], [1, 1, 1]], dtype=np.int8),
}


@dataclass
class Piece:
    kind: TetrominoType
    rotation: int = 0  # 0..3

    def shape(self) -> Shape:
        base = BASE_SHAPES[self.kind]
        return _rot90(base, self.rotation)

    def rotated(self, delta: int) -> "Piece":
        return Piece(self.kind, (self.rotation + delta) % 4)

    def cells_at(self, origin_x: int, origin_y: int) -> List[Tuple[int, int]]:
        s = self.shape()
        h, w = s.shape
        cells: List[Tuple[int, int]] = []
        for dy in range(h):
            for dx in range(w):
                if s[dy, dx]:
                    cells.append((origin_x + dx, origin_y + dy))
        return cells



