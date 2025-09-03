from __future__ import annotations

import argparse
from typing import Optional

import gymnasium as gym
import pygame

import block_puzzle_rl.env  # ensure registration
from block_puzzle_rl.env.wrappers import FlattenDiscreteActionWrapper, ResampleInvalidActionWrapper
from block_puzzle_rl.visualization.blockpuzzle_play import draw_board, draw_pieces


def build_env(render_mode: Optional[str] = None, use_resample: bool = True) -> gym.Env:
    env = gym.make("BlockPuzzle-10x10-v0", render_mode=render_mode)
    env = FlattenDiscreteActionWrapper(env)
    if use_resample:
        env = ResampleInvalidActionWrapper(env)
    return env


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "maskable"], default="ppo")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--fps", type=int, default=10)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.algo == "maskable":
        from sb3_contrib import MaskablePPO as Algo
    else:
        from stable_baselines3 import PPO as Algo

    env = build_env(render_mode=None, use_resample=(args.algo == "ppo"))
    model = Algo.load(args.model, device="auto")

    # Setup pygame viewer based on grid size
    game = env.unwrapped.game
    cell_size = 30
    margin = 20
    board_px_w = game.grid.size * cell_size
    board_px_h = game.grid.size * cell_size
    side_panel_w = 8 * cell_size
    width = margin * 3 + board_px_w + side_panel_w
    height = margin * 2 + board_px_h

    pygame.init()
    try:
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Block Puzzle - Agent Eval")
        font = pygame.font.SysFont(None, 24)
        clock = pygame.time.Clock()

        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while steps < args.steps:
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            # Predict action
            if args.algo == "maskable":
                mask = env.get_action_mask()
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
            if done:
                obs, info = env.reset()

            # Draw current game state
            game = env.unwrapped.game
            draw_board(screen, game.grid.grid, cell_size, margin)
            draw_pieces(screen, game, cell_size, margin, selected_piece=-1)
            txt = font.render(f"step {steps}/{args.steps}  reward {total_reward:.1f}", True, (230, 230, 230))
            screen.blit(txt, (margin, 2))
            pygame.display.flip()
            clock.tick(args.fps)
    finally:
        pygame.quit()


if __name__ == "__main__":  # pragma: no cover
    main()


