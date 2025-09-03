# Block Puzzle RL

Project scaffold for a Gymnasium environment and RL agents for Block Puzzle.

## Quickstart

- Create venv and install deps:

	cd block_puzzle_rl
	uv sync

- Run a quick import check:

	uv run python -c "import gymnasium, numpy, pygame; print('OK')"

## Layout

- src/block_puzzle_rl: package code
- pyproject.toml: project metadata

## Run commands

- Play Tetris variant (keyboard):

\tuv run block-puzzle-play

- Play Block Puzzle (9x9) human UI:

\tuv run block-puzzle-sudoku-play

- Random agent (env sanity):

\tuv run block-puzzle-random

- PPO training (vanilla):

\tuv run block-puzzle-train-ppo --algo ppo --timesteps 200000 --n_envs 4

- Maskable PPO (requires sb3-contrib):

\tuv run block-puzzle-train-ppo --algo maskable --timesteps 200000 --n_envs 4

- Linear Q-learning baseline (CPU):

\tuv run block-puzzle-train-linear --episodes 500

- Evaluate saved PPO model visually:

\tuv run block-puzzle-eval --algo maskable --model models/ppo_blockpuzzle.zip --steps 500 --fps 10

- TensorBoard logs:

\tuv run tensorboard --logdir logs/ppo

## Experiments

- Log your runs in `docs/EXPERIMENTS.md`:
  - Date, commit, grid size, algo, steps, reward weights, n_envs
  - Results: `rollout/ep_rew_mean`, `rollout/ep_len_mean`
  - Artifacts: model path and logs path
