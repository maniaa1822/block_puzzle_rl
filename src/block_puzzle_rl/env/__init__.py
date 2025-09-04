"""Gymnasium environments for Block Puzzle RL."""

from __future__ import annotations

from gymnasium.envs.registration import register

# Register default Block Puzzle environment
register(
    id="BlockPuzzle-10x10-v0",
    entry_point="block_puzzle_rl.env.block_puzzle_env:BlockPuzzleEnv",
)

# Register sequential placement environment (11 discrete actions)
register(
    id="BlockPuzzleSequential-10x10-v0",
    entry_point="block_puzzle_rl.env.sequential_env:SequentialPlacementEnv",
)

__all__ = ["BlockPuzzle-10x10-v0", "BlockPuzzleSequential-10x10-v0"]


