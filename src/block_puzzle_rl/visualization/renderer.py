from __future__ import annotations

from typing import Tuple

import numpy as np
import pygame


def _color_for_value(v: int) -> Tuple[int, int, int]:
    palette = {
        0: (20, 20, 26),
        1: (0, 240, 240),  # I
        2: (240, 240, 0),  # O
        3: (160, 0, 240),  # T
        4: (0, 240, 0),    # S
        5: (240, 0, 0),    # Z
        6: (0, 0, 240),    # J
        7: (240, 160, 0),  # L
    }
    return palette.get(abs(v), (200, 200, 200))


class Renderer:
    def __init__(self, cell_size: int = 30, margin: int = 20) -> None:
        self.cell_size = cell_size
        self.margin = margin

    def _grid_surface(self, state: np.ndarray) -> pygame.Surface:
        h, w = state.shape
        width = w * self.cell_size
        height = h * self.cell_size
        surf = pygame.Surface((width, height))
        surf.fill((30, 30, 36))
        for y in range(h):
            for x in range(w):
                v = int(state[y, x])
                color = _color_for_value(v)
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size - 1,
                    self.cell_size - 1,
                )
                pygame.draw.rect(surf, color, rect)
        return surf

    def draw(self, screen: pygame.Surface, state: np.ndarray) -> None:
        grid_surf = self._grid_surface(state)
        screen.fill((10, 10, 14))
        screen.blit(grid_surf, (self.margin, self.margin))
        pygame.display.flip()



