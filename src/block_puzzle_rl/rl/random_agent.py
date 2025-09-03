from __future__ import annotations

import random
from typing import Tuple

import gymnasium as gym


def run_random(steps: int = 200) -> None:
    env = gym.make("BlockPuzzle-10x10-v0")
    obs, info = env.reset()
    total_reward = 0.0
    for _ in range(steps):
        # Prefer valid actions if available
        valid = info.get("valid_actions", [])
        if valid:
            action = random.choice(valid)
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    print(f"Random agent total reward: {total_reward:.2f}")


if __name__ == "__main__":  # pragma: no cover
    run_random()


