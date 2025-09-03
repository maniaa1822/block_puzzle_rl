"""Block Puzzle (10x10) game logic.

This package provides a sudoku-style block puzzle game where the player places
3 pieces per set onto a 10x10 board and clears full rows and columns.
"""

from .logic import (
    GameConfig,
    PieceType,
    PieceShapes,
    Piece,
    GameGrid,
    ScoreCalculator,
    BlockPuzzleGame,
    GameAnalytics,
    print_grid,
    print_piece,
)

__all__ = [
    "GameConfig",
    "PieceType",
    "PieceShapes",
    "Piece",
    "GameGrid",
    "ScoreCalculator",
    "BlockPuzzleGame",
    "GameAnalytics",
    "print_grid",
    "print_piece",
]



