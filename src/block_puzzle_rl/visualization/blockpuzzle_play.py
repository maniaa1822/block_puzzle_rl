from __future__ import annotations

import pygame

from block_puzzle_rl.blockpuzzle import BlockPuzzleGame, PieceShapes, PieceType


def _color_for_value(v: int) -> tuple[int, int, int]:
    return (40, 40, 48) if v == 0 else (70, 200, 120)


def draw_board(screen: pygame.Surface, grid, cell_size: int, margin: int) -> None:
    h, w = grid.shape
    screen.fill((15, 15, 20))
    for y in range(h):
        for x in range(w):
            rect = pygame.Rect(margin + x * cell_size, margin + y * cell_size, cell_size - 1, cell_size - 1)
            pygame.draw.rect(screen, _color_for_value(int(grid[y, x])), rect)


def draw_pieces(screen: pygame.Surface, game: BlockPuzzleGame, cell_size: int, margin: int, selected_piece: int) -> None:
    # Draw the current 3 pieces at the right side
    x0 = margin * 2 + game.grid.size * cell_size
    y0 = margin
    for idx, piece in enumerate(game.current_pieces):
        shape = piece.get_shape(0)
        off_y = y0 + idx * (cell_size * 5)
        for py in range(shape.shape[0]):
            for px in range(shape.shape[1]):
                if shape[py, px]:
                    rect = pygame.Rect(x0 + px * cell_size, off_y + py * cell_size, cell_size - 1, cell_size - 1)
                    pygame.draw.rect(screen, (200, 180, 60), rect)
        # Highlight selected
        if idx == selected_piece:
            w = shape.shape[1] * cell_size
            h = shape.shape[0] * cell_size
            outline = pygame.Rect(x0, off_y, w, h)
            pygame.draw.rect(screen, (255, 255, 255), outline, 2)


def draw_ghost(screen: pygame.Surface, game: BlockPuzzleGame, grid_x: int, grid_y: int, rotation: int, cell_size: int, margin: int, selected_piece: int) -> None:
    if not (0 <= selected_piece < len(game.current_pieces)):
        return
    piece = game.current_pieces[selected_piece]
    shape = piece.get_shape(rotation)
    color_valid = (120, 220, 140)
    color_invalid = (220, 120, 120)
    is_valid = game.grid.is_valid_placement(shape, grid_x, grid_y)
    color = color_valid if is_valid else color_invalid
    for py in range(shape.shape[0]):
        for px in range(shape.shape[1]):
            if shape[py, px]:
                x = margin + (grid_x + px) * cell_size
                y = margin + (grid_y + py) * cell_size
                rect = pygame.Rect(x, y, cell_size - 1, cell_size - 1)
                pygame.draw.rect(screen, color, rect, 2)


def run() -> None:
    pygame.init()
    try:
        game = BlockPuzzleGame()
        cell_size = 30
        margin = 20
        board_px_w = game.grid.size * cell_size
        board_px_h = game.grid.size * cell_size
        side_panel_w = 8 * cell_size
        width = margin * 3 + board_px_w + side_panel_w
        height = margin * 2 + board_px_h
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Block Puzzle (10x10) - Human Play")
        font = pygame.font.SysFont(None, 24)

        selected_piece = 0
        rotation = 0

        key_to_index = {
            pygame.K_1: 0,
            pygame.K_2: 1,
            pygame.K_3: 2,
            pygame.K_KP1: 0,
            pygame.K_KP2: 1,
            pygame.K_KP3: 2,
        }

        running = True
        clock = pygame.time.Clock()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in key_to_index:
                        idx = key_to_index[event.key]
                        if idx < len(game.current_pieces):
                            selected_piece = idx
                            rotation = 0
                    elif event.key in (pygame.K_r, pygame.K_q, pygame.K_e):
                        delta = 1 if event.key in (pygame.K_r, pygame.K_e) else -1
                        rotation = (rotation + delta) % 4
                    elif event.key == pygame.K_n:
                        game.reset()

            # Mouse placement
            if not game.game_over and len(game.current_pieces) > 0:
                mouse_pressed = pygame.mouse.get_pressed()[0]
                mx, my = pygame.mouse.get_pos()
                grid_x = (mx - margin) // cell_size
                grid_y = (my - margin) // cell_size
                if 0 <= selected_piece < len(game.current_pieces):
                    piece = game.current_pieces[selected_piece]
                    shape = piece.get_shape(rotation)
                    if mouse_pressed:
                        if game.grid.is_valid_placement(shape, grid_x, grid_y):
                            game.place_piece(selected_piece, grid_x, grid_y, rotation)
                            selected_piece = 0
                            rotation = 0

            # Draw
            draw_board(screen, game.grid.grid, cell_size, margin)
            # Ghost preview under mouse
            mx, my = pygame.mouse.get_pos()
            grid_x = (mx - margin) // cell_size
            grid_y = (my - margin) // cell_size
            draw_ghost(screen, game, grid_x, grid_y, rotation, cell_size, margin, selected_piece)
            draw_pieces(screen, game, cell_size, margin, selected_piece)
            # UI text
            info_lines = [
                f"Score: {game.score}",
                f"Pieces left: {len(game.current_pieces)}",
                "Select: 1/2/3 or Numpad 1/2/3",
                "Rotate: R or Q/E",
                "Reset: N",
                "Place: Left click",
            ]
            x_text = margin * 2 + game.grid.size * cell_size
            y_text = margin + 5 * cell_size * 3 + 10
            for i, txt in enumerate(info_lines):
                img = font.render(txt, True, (230, 230, 230))
                screen.blit(img, (x_text, y_text + i * 20))
            if game.game_over:
                over = font.render("Game Over - Press N to reset", True, (255, 100, 100))
                screen.blit(over, (margin, 2))

            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()


if __name__ == "__main__":  # pragma: no cover
    run()
