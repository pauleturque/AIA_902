import cv2
import numpy as np

def preprocess_state(state, grid_size=(20, 20)):
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

    # Binarisation
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Détection des contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_position = (0, 0)
    paddle_positions = []

    height, width = state.shape[:2]
    top_border = int(height * 0.16)
    bottom_border = int(height * 0.93)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filtrage zone de jeu
        if y < top_border or y + h > bottom_border:
            continue

        # Filtrage objets
        if 1 <= w <= 4 and 1 <= h <= 4:  # balle
            ball_position = (x, y)
        elif 2 <= w <= 6 and 10 <= h <= 30:  # raquettes
            paddle_positions.append((x, y))

    # Normalisation dans la grille
    grid_w, grid_h = grid_size
    scaled_ball_x = int((ball_position[0] / width) * (grid_w - 1))
    scaled_ball_y = int((ball_position[1] / height) * (grid_h - 1))
    discrete_ball = (scaled_ball_x, scaled_ball_y)

    paddle_positions_scaled = []
    for paddle in paddle_positions:
        px = int((paddle[0] / width) * (grid_w - 1))
        py = int((paddle[1] / height) * (grid_h - 1))
        paddle_positions_scaled.append((px, py))

    return discrete_ball, paddle_positions_scaled
