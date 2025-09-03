"""Gymnasium environments for Block Puzzle RL."""

from __future__ import annotations

from gymnasium.envs.registration import register

# Register default Block Puzzle environment
register(
    id="BlockPuzzle-10x10-v0",
    entry_point="block_puzzle_rl.env.block_puzzle_env:BlockPuzzleEnv",
)

__all__ = ["BlockPuzzle-10x10-v0"]


