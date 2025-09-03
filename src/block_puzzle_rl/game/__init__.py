"""Game module for Block Puzzle RL.

Exports the core game engine and supporting classes:
- GameGrid: Grid representation and line clearing
- Piece: Tetromino piece with rotation mechanics
- TetrominoType: Enum of available piece types
- ScoringRules: Simple scoring configuration and helpers
- BlockPuzzleGame: Main game loop and state management
"""

from .grid import GameGrid
from .pieces import Piece, TetrominoType
from .rules import ScoringRules
from .core import BlockPuzzleGame, Action

__all__ = [
    "GameGrid",
    "Piece",
    "TetrominoType",
    "ScoringRules",
    "BlockPuzzleGame",
    "Action",
]



