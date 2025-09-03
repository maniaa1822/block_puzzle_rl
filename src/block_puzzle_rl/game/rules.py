from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoringRules:
    line_clear_scores: tuple[int, int, int, int] = (100, 300, 500, 800)
    placement_score: int = 0
    game_over_penalty: int = 0

    def score_for_lines(self, lines: int) -> int:
        if lines <= 0:
            return 0
        if 1 <= lines <= 4:
            return self.line_clear_scores[lines - 1]
        # Exaggerate beyond 4 just in case of variants
        return self.line_clear_scores[-1] + (lines - 4) * 400



