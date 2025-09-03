from __future__ import annotations

import os
import sys
import time
from typing import Dict

import numpy as np
import pygame

from block_puzzle_rl.game import BlockPuzzleGame, Action
from .renderer import Renderer


KEY_TO_ACTION: Dict[int, Action] = {
    pygame.K_LEFT: Action.LEFT,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_UP: Action.ROTATE_CW,
    pygame.K_z: Action.ROTATE_CCW,
    pygame.K_DOWN: Action.SOFT_DROP,
    pygame.K_SPACE: Action.HARD_DROP,
}


def run() -> None:
    pygame.init()
    try:
        clock = pygame.time.Clock()
        game = BlockPuzzleGame()
        renderer = Renderer(cell_size=28)

        # Window size based on grid
        state = game.get_state()
        h, w = state.shape
        margin = 20
        screen = pygame.display.set_mode((w * 28 + margin * 2, h * 28 + margin * 2))
        pygame.display.set_caption("Block Puzzle - Human Play")

        gravity_ms = 600
        last_fall = pygame.time.get_ticks()

        running = True
        while running:
            # Input handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    else:
                        action = KEY_TO_ACTION.get(event.key)
                        if action is not None:
                            game.step(action)

            # Gravity
            now = pygame.time.get_ticks()
            if now - last_fall >= gravity_ms:
                game.step(Action.SOFT_DROP)
                last_fall = now

            # Render
            renderer.draw(screen, game.get_state())

            # Game over reset prompt
            if game.game_over:
                # Simple overlay
                font = pygame.font.SysFont(None, 36)
                text = font.render("Game Over - Press R to restart, ESC to quit", True, (255, 255, 255))
                rect = text.get_rect(center=(screen.get_width() // 2, 30))
                screen.blit(text, rect)
                pygame.display.flip()
                # Poll for restart or quit
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_r:
                            game.reset()

            clock.tick(60)
    finally:
        pygame.quit()


if __name__ == "__main__":  # pragma: no cover
    run()



