from __future__ import annotations

import argparse
import random
from typing import List, Tuple
import sys

import numpy as np

from block_puzzle_rl.blockpuzzle import BlockPuzzleGame, GameAnalytics


def extract_features(game: BlockPuzzleGame, piece_idx: int, x: int, y: int, r: int) -> np.ndarray:
    piece = game.current_pieces[piece_idx]
    shape = piece.get_shape(r)
    if not game.grid.is_valid_placement(shape, x, y):
        # Impossible action -> sentinel features
        return np.array([-1.0] * 8, dtype=np.float32)

    before = GameAnalytics.get_board_features(game.grid.grid)
    temp = game.grid.copy()
    cells_placed = temp.place_piece(shape, x, y)
    lines, _ = temp.clear_complete_lines()
    after = GameAnalytics.get_board_features(temp.grid)

    # Features: [bias, cells, lines, lines^2, d_holes, d_bump, d_height, fill_ratio]
    d_holes = after["holes"] - before["holes"]
    d_bump = after["bumpiness"] - before["bumpiness"]
    d_height = after["max_height"] - before["max_height"]
    fill_ratio = after["fill_ratio"]
    return np.array([
        1.0,
        float(cells_placed),
        float(lines),
        float(lines * lines),
        float(d_holes),
        float(d_bump),
        float(d_height),
        float(fill_ratio),
    ], dtype=np.float32)


def enumerate_actions(game: BlockPuzzleGame) -> List[Tuple[int, int, int, int]]:
    return game.get_valid_actions()


def _print_progress(ep_idx: int, total: int, last_return: float, last_steps: int) -> None:
    width = 30
    filled = int(width * (ep_idx + 1) / max(1, total))
    bar = "=" * filled + "." * (width - filled)
    msg = f"\r[{bar}] {ep_idx + 1}/{total}  return={last_return:.1f}  steps={last_steps}"
    print(msg, end="", file=sys.stdout, flush=True)


def train_linear_q(episodes: int = 2000, epsilon: float = 0.1, alpha: float = 1e-3, gamma: float = 0.99,
                   seed: int = 0, progress: bool = True) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)

    game = BlockPuzzleGame()
    w = np.zeros((8,), dtype=np.float32)  # weights for features

    for ep in range(episodes):
        game.reset(seed + ep)
        done = False
        ep_return = 0.0
        steps = 0

        while not done and steps < game.config.max_episode_steps:
            actions = enumerate_actions(game)
            if not actions:
                break
            # epsilon-greedy over valid actions
            if random.random() < epsilon:
                a = random.choice(actions)
            else:
                # choose argmax w·phi
                best_q = None
                best_a = None
                for a_cand in actions:
                    phi = extract_features(game, *a_cand)
                    q = float(np.dot(w, phi))
                    if (best_q is None) or (q > best_q):
                        best_q, best_a = q, a_cand
                a = best_a if best_a is not None else random.choice(actions)

            # Compute features BEFORE mutating state (piece list changes after placement)
            phi_sa = extract_features(game, *a)

            # Take action
            success, gained, _ = game.place_piece(*a)
            reward = float(gained)
            ep_return += reward
            steps += 1

            # TD target using next state's greedy evaluation
            if game.game_over:
                target = reward
                done = True
            else:
                next_actions = enumerate_actions(game)
                if not next_actions:
                    target = reward
                    done = True
                else:
                    q_next_max = max(float(np.dot(w, extract_features(game, *a2))) for a2 in next_actions)
                    target = reward + gamma * q_next_max

            # Update
            q_sa = float(np.dot(w, phi_sa))
            td_error = target - q_sa
            w += alpha * td_error * phi_sa

        if progress:
            _print_progress(ep, episodes, ep_return, steps)
        elif (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes} return={ep_return:.1f} steps={steps}")

    if progress:
        print()
    return w


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()

    w = train_linear_q(args.episodes, args.epsilon, args.alpha, args.gamma, args.seed, progress=not args.no_progress)
    np.save("models/linear_q_weights.npy", w)
    print("Saved weights to models/linear_q_weights.npy")


if __name__ == "__main__":  # pragma: no cover
    main()


