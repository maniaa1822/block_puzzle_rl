from __future__ import annotations

import argparse
import os
from typing import Any

import gymnasium as gym

# Ensure envs are registered
import block_puzzle_rl.env  # noqa: F401
from block_puzzle_rl.env.wrappers import FlattenDiscreteActionWrapper, ResampleInvalidActionWrapper


def make_env(env_id: str, seed: int | None = None) -> gym.Env:
    env = gym.make(env_id)
    # Only flatten when using the MultiDiscrete placement env
    if env_id == "BlockPuzzle-10x10-v0":
        env = FlattenDiscreteActionWrapper(env)
    # Resample invalid actions for vanilla PPO; also forwards get_action_mask
    env = ResampleInvalidActionWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "maskable"], default="ppo")
    p.add_argument("--env", choices=["default", "sequential"], default="sequential",
                   help="Which environment to train: default (MultiDiscrete) or sequential (11-action)")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--logdir", type=str, default="./logs/ppo")
    p.add_argument("--save_path", type=str, default="./models/ppo_blockpuzzle.zip")
    p.add_argument("--n_envs", type=int, default=4)
    return p


def main() -> None:
    args = build_parser().parse_args()

    env_id = "BlockPuzzleSequential-10x10-v0" if args.env == "sequential" else "BlockPuzzle-10x10-v0"

    if args.algo == "maskable":
        # sb3-contrib MaskablePPO
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker

        def mask_fn(env):
            # env is the wrapped env; ResampleInvalidActionWrapper now forwards get_action_mask
            return env.get_action_mask()

        def make_env_idx(i: int):
            def thunk():
                e = make_env(env_id)
                e = ActionMasker(e, mask_fn)
                return e
            return thunk

        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

        vec_env = SubprocVecEnv([make_env_idx(i) for i in range(args.n_envs)])
        vec_env = VecMonitor(vec_env)
        model = MaskablePPO(
            policy="MultiInputPolicy",
            env=vec_env,
            verbose=1,
            tensorboard_log=args.logdir,
        )
    else:
        # Vanilla PPO with resampling wrapper
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

        def make_env_idx(i: int):
            def thunk():
                return make_env(env_id)
            return thunk

        vec_env = SubprocVecEnv([make_env_idx(i) for i in range(args.n_envs)])
        vec_env = VecMonitor(vec_env)
        model = PPO(
            policy="MultiInputPolicy",
            env=vec_env,
            verbose=1,
            tensorboard_log=args.logdir,
        )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)


if __name__ == "__main__":  # pragma: no cover
    main()


