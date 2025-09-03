from __future__ import annotations

"""
Complete Block Puzzle Game Logic
Modern block puzzle game where players place 3 pieces at a time on a 9x9 grid
to clear complete rows and columns for points.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import IntEnum


@dataclass
class GameConfig:
    """Configuration for block puzzle game"""
    grid_size: int = 9
    max_episode_steps: int = 10000
    pieces_per_set: int = 3
    base_placement_points: int = 1
    line_clear_points: int = 10
    combo_multiplier: float = 2.0


class PieceType(IntEnum):
    """Enumeration of piece types"""
    I = 0  # Line piece
    O = 1  # Square piece
    T = 2  # T-piece
    L = 3  # L-piece
    J = 4  # J-piece (reverse L)
    S = 5  # S-piece
    Z = 6  # Z-piece


class PieceShapes:
    """Static piece shape definitions"""

    # Base shapes (rotation 0)
    SHAPES = {
        PieceType.I: np.array([[1, 1, 1, 1]], dtype=int),
        PieceType.O: np.array([[1, 1], [1, 1]], dtype=int),
        PieceType.T: np.array([[0, 1, 0], [1, 1, 1]], dtype=int),
        PieceType.L: np.array([[1, 0, 0], [1, 1, 1]], dtype=int),
        PieceType.J: np.array([[0, 0, 1], [1, 1, 1]], dtype=int),
        PieceType.S: np.array([[0, 1, 1], [1, 1, 0]], dtype=int),
        PieceType.Z: np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    }

    @classmethod
    def get_shape(cls, piece_type: PieceType, rotation: int = 0) -> np.ndarray:
        """Get piece shape with specified rotation"""
        shape = cls.SHAPES[piece_type].copy()
        for _ in range(rotation % 4):
            shape = np.rot90(shape)
        return shape

    @classmethod
    def get_all_rotations(cls, piece_type: PieceType) -> List[np.ndarray]:
        """Get all unique rotations for a piece type"""
        rotations: List[np.ndarray] = []
        for r in range(4):
            shape = cls.get_shape(piece_type, r)
            # Check if this rotation is unique
            if not any(np.array_equal(shape, existing) for existing in rotations):
                rotations.append(shape)
        return rotations


class Piece:
    """Individual piece instance"""

    def __init__(self, piece_type: PieceType):
        self.piece_type = piece_type
        self.rotation = 0

    def get_shape(self, rotation: int | None = None) -> np.ndarray:
        """Get current piece shape at specified rotation"""
        if rotation is None:
            rotation = self.rotation
        return PieceShapes.get_shape(self.piece_type, rotation)

    def get_all_rotations(self) -> List[np.ndarray]:
        """Get all unique rotations of this piece"""
        return PieceShapes.get_all_rotations(self.piece_type)

    def get_bounding_box(self, rotation: int = 0) -> Tuple[int, int]:
        """Get (height, width) of piece at rotation"""
        shape = self.get_shape(rotation)
        return shape.shape


class GameGrid:
    """10x10 game grid with placement and line clearing logic"""

    def __init__(self, size: int = 10):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)

    def is_valid_placement(self, piece_shape: np.ndarray, x: int, y: int) -> bool:
        """Check if piece can be placed at position (x,y)"""
        piece_h, piece_w = piece_shape.shape

        # Check bounds
        if x < 0 or y < 0:
            return False
        if x + piece_w > self.size or y + piece_h > self.size:
            return False

        # Check for collision with existing blocks
        for py in range(piece_h):
            for px in range(piece_w):
                if piece_shape[py, px] and self.grid[y + py, x + px]:
                    return False

        return True

    def place_piece(self, piece_shape: np.ndarray, x: int, y: int) -> int:
        """
        Place piece on grid and return number of cells placed
        Assumes position is already validated
        """
        cells_placed = 0
        piece_h, piece_w = piece_shape.shape

        for py in range(piece_h):
            for px in range(piece_w):
                if piece_shape[py, px]:
                    self.grid[y + py, x + px] = 1
                    cells_placed += 1

        return cells_placed

    def clear_complete_lines(self) -> Tuple[int, int]:
        """
        Clear complete rows and columns
        Returns: (lines_cleared, cells_cleared)
        """
        lines_cleared = 0
        cells_cleared = 0

        # Clear complete rows
        rows_to_clear = [row for row in range(self.size) if np.all(self.grid[row, :])]
        for row in rows_to_clear:
            self.grid[row, :] = 0
            lines_cleared += 1
            cells_cleared += self.size

        # Clear complete columns
        cols_to_clear = [col for col in range(self.size) if np.all(self.grid[:, col])]
        for col in cols_to_clear:
            self.grid[:, col] = 0
            lines_cleared += 1
            cells_cleared += self.size

        return lines_cleared, cells_cleared

    def get_valid_placements(self, piece_shape: np.ndarray) -> List[Tuple[int, int]]:
        """Get all valid (x,y) positions for a piece shape"""
        valid_positions: List[Tuple[int, int]] = []
        for x in range(self.size):
            for y in range(self.size):
                if self.is_valid_placement(piece_shape, x, y):
                    valid_positions.append((x, y))
        return valid_positions

    def get_filled_ratio(self) -> float:
        """Get percentage of grid that is filled"""
        return float(np.sum(self.grid)) / float(self.size * self.size)

    def copy(self) -> "GameGrid":
        """Create a copy of the current grid"""
        new_grid = GameGrid(self.size)
        new_grid.grid = self.grid.copy()
        return new_grid


class ScoreCalculator:
    """Handles scoring logic and reward calculation"""

    def __init__(self, config: GameConfig):
        self.base_placement_points = getattr(config, "base_placement_points", 1)
        self.line_clear_points = getattr(config, "line_clear_points", 10)
        self.combo_multiplier = getattr(config, "combo_multiplier", 2.0)

    def calculate_score(self, cells_placed: int, lines_cleared: int) -> int:
        """Calculate score for a single placement"""
        placement_score = int(cells_placed * self.base_placement_points)
        if lines_cleared == 0:
            return placement_score
        line_score = int(lines_cleared * self.line_clear_points)
        if lines_cleared > 1:
            combo_bonus = int(line_score * (self.combo_multiplier ** (lines_cleared - 1)))
            return placement_score + line_score + combo_bonus
        return placement_score + line_score


class BlockPuzzleGame:
    """Main game engine for block puzzle"""

    def __init__(self, config: GameConfig | None = None):
        self.config = config or GameConfig()
        self.grid = GameGrid(self.config.grid_size)
        self.score_calculator = ScoreCalculator(self.config)

        # Game state
        self.current_pieces: List[Piece] = []
        self.score = 0
        self.total_lines_cleared = 0
        self.total_pieces_placed = 0
        self.step_count = 0
        self.game_over = False

        self.generate_new_piece_set()

    def generate_new_piece_set(self) -> None:
        """Generate 3 new random pieces"""
        self.current_pieces = []
        for _ in range(self.config.pieces_per_set):
            piece_type = PieceType(np.random.randint(0, len(PieceType)))
            self.current_pieces.append(Piece(piece_type))

    def get_current_piece_types(self) -> List[int]:
        return [piece.piece_type.value for piece in self.current_pieces]

    def can_place_any_piece(self) -> bool:
        for piece in self.current_pieces:
            for rotation in range(4):
                shape = piece.get_shape(rotation)
                if len(self.grid.get_valid_placements(shape)) > 0:
                    return True
        return False

    def get_valid_actions(self) -> List[Tuple[int, int, int, int]]:
        """List of (piece_idx, x, y, rotation) valid actions"""
        actions: List[Tuple[int, int, int, int]] = []
        for piece_idx, piece in enumerate(self.current_pieces):
            for rotation in range(4):
                shape = piece.get_shape(rotation)
                for x, y in self.grid.get_valid_placements(shape):
                    actions.append((piece_idx, x, y, rotation))
        return actions

    def place_piece(self, piece_idx: int, x: int, y: int, rotation: int = 0) -> Tuple[bool, int, int]:
        if piece_idx < 0 or piece_idx >= len(self.current_pieces):
            return False, 0, 0
        piece = self.current_pieces[piece_idx]
        shape = piece.get_shape(rotation)
        if not self.grid.is_valid_placement(shape, x, y):
            return False, 0, 0
        cells_placed = self.grid.place_piece(shape, x, y)
        lines_cleared, _ = self.grid.clear_complete_lines()
        gained = self.score_calculator.calculate_score(cells_placed, lines_cleared)
        self.score += gained
        self.total_pieces_placed += 1
        self.total_lines_cleared += lines_cleared
        self.step_count += 1
        self.current_pieces.pop(piece_idx)
        if len(self.current_pieces) == 0:
            self.generate_new_piece_set()
        if not self.can_place_any_piece():
            self.game_over = True
        return True, gained, lines_cleared

    def simulate_placement(self, piece_idx: int, x: int, y: int, rotation: int = 0) -> Tuple[bool, int, int]:
        if piece_idx < 0 or piece_idx >= len(self.current_pieces):
            return False, 0, 0
        piece = self.current_pieces[piece_idx]
        shape = piece.get_shape(rotation)
        if not self.grid.is_valid_placement(shape, x, y):
            return False, 0, 0
        temp_grid = self.grid.copy()
        cells_placed = temp_grid.place_piece(shape, x, y)
        lines_cleared, _ = temp_grid.clear_complete_lines()
        potential = self.score_calculator.calculate_score(cells_placed, lines_cleared)
        return True, potential, lines_cleared

    def get_state(self) -> dict:
        return {
            "grid": self.grid.grid.copy(),
            "current_pieces": self.get_current_piece_types(),
            "pieces_remaining": len(self.current_pieces),
            "score": self.score,
            "total_lines_cleared": self.total_lines_cleared,
            "total_pieces_placed": self.total_pieces_placed,
            "step_count": self.step_count,
            "game_over": self.game_over,
            "filled_ratio": self.grid.get_filled_ratio(),
        }

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed)
        self.grid = GameGrid(self.config.grid_size)
        self.current_pieces = []
        self.score = 0
        self.total_lines_cleared = 0
        self.total_pieces_placed = 0
        self.step_count = 0
        self.game_over = False
        self.generate_new_piece_set()

    def get_game_stats(self) -> dict:
        return {
            "final_score": self.score,
            "pieces_placed": self.total_pieces_placed,
            "lines_cleared": self.total_lines_cleared,
            "steps_taken": self.step_count,
            "final_fill_ratio": self.grid.get_filled_ratio(),
            "avg_score_per_piece": self.score / max(1, self.total_pieces_placed),
            "avg_lines_per_piece": self.total_lines_cleared / max(1, self.total_pieces_placed),
        }


class GameAnalytics:
    """Helper class for analyzing game states and positions"""

    @staticmethod
    def get_board_features(grid: np.ndarray) -> dict:
        size = grid.shape[0]
        height_map: List[int] = []
        for col in range(size):
            height = 0
            for row in range(size):
                if grid[row, col] == 1:
                    height = size - row
                    break
            height_map.append(height)
        holes = 0
        for col in range(size):
            found_block = False
            for row in range(size):
                if grid[row, col] == 1:
                    found_block = True
                elif found_block and grid[row, col] == 0:
                    holes += 1
        bumpiness = sum(abs(height_map[i] - height_map[i + 1]) for i in range(size - 1))
        almost_complete_rows = sum(1 for row in range(size) if np.sum(grid[row, :]) >= size - 1)
        almost_complete_cols = sum(1 for col in range(size) if np.sum(grid[:, col]) >= size - 1)
        return {
            "max_height": max(height_map) if height_map else 0,
            "avg_height": float(np.mean(height_map)) if height_map else 0.0,
            "holes": holes,
            "bumpiness": bumpiness,
            "filled_cells": int(np.sum(grid)),
            "fill_ratio": float(np.sum(grid)) / float(size * size),
            "almost_complete_lines": almost_complete_rows + almost_complete_cols,
            "empty_rows": sum(1 for row in range(size) if np.sum(grid[row, :]) == 0),
            "empty_cols": sum(1 for col in range(size) if np.sum(grid[:, col]) == 0),
        }

    @staticmethod
    def evaluate_placement_quality(grid: GameGrid, piece_shape: np.ndarray, x: int, y: int) -> dict:
        if not grid.is_valid_placement(piece_shape, x, y):
            return {"valid": False}
        temp_grid = grid.copy()
        cells_placed = temp_grid.place_piece(piece_shape, x, y)
        lines_cleared, _ = temp_grid.clear_complete_lines()
        features_before = GameAnalytics.get_board_features(grid.grid)
        features_after = GameAnalytics.get_board_features(temp_grid.grid)
        return {
            "valid": True,
            "cells_placed": cells_placed,
            "lines_cleared": lines_cleared,
            "height_increase": features_after["max_height"] - features_before["max_height"],
            "holes_created": features_after["holes"] - features_before["holes"],
            "bumpiness_change": features_after["bumpiness"] - features_before["bumpiness"],
            "almost_complete_lines_after": features_after["almost_complete_lines"],
        }


def print_grid(grid: np.ndarray) -> None:
    for row in grid:
        print("".join(["█" if cell else "·" for cell in row]))


def print_piece(piece_shape: np.ndarray) -> None:
    for row in piece_shape:
        print("".join(["█" if cell else "·" for cell in row]))


def run_game_demo() -> None:  # pragma: no cover
    game = BlockPuzzleGame()
    print("=== Block Puzzle Game Demo ===")
    print(f"Initial pieces: {game.get_current_piece_types()}")
    print("\nInitial grid:")
    print_grid(game.grid.grid)
    if len(game.current_pieces) > 0:
        piece = game.current_pieces[0]
        shape = piece.get_shape(0)
        print(f"\nPiece 0 shape:")
        print_piece(shape)
        success, score_gained, lines_cleared = game.place_piece(0, 0, 0, 0)
        if success:
            print(f"\nPlaced piece! Score gained: {score_gained}, Lines cleared: {lines_cleared}")
            print("Grid after placement:")
            print_grid(game.grid.grid)
            print(f"Remaining pieces: {game.get_current_piece_types()}")
            print(f"Total score: {game.score}")
        else:
            print("Could not place piece at (0,0)")
    valid_actions = game.get_valid_actions()
    print(f"\nTotal valid actions available: {len(valid_actions)}")
    if valid_actions:
        print(f"Example valid action: {valid_actions[0]} (piece_idx, x, y, rotation)")


if __name__ == "__main__":  # pragma: no cover
    run_game_demo()
    print("\n=== Piece Rotation Test ===")
    for piece_type in PieceType:
        print(f"\n{piece_type.name} piece rotations:")
        rotations = PieceShapes.get_all_rotations(piece_type)
        for i, rotation in enumerate(rotations):
            print(f"Rotation {i}:")
            print_piece(rotation)



