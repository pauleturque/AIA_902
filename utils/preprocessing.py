import cv2
import numpy as np


def preprocess_state(state):
    # conversion en gris
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    print(f"Forme de l'image en niveaux de gris: {gray.shape}")

    #redimension img
    resized = cv2.resize(gray, (80, 80), interpolation=cv2.INTER_AREA)
    print(f"Forme après redimensionnement: {resized.shape}")

    #img en binaire
    _, binary = cv2.threshold(resized, 100, 255, cv2.THRESH_BINARY)
    print(f"Forme de l'image binaire: {binary.shape}")

    #détection balle
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_position = (0, 0)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 3 < w < 10 and 3 < h < 10:
            ball_position = (x, y)
            break

    print(f"Position de la balle détectée: {ball_position}")

    #normalisation -> évite les depassements
    max_x, max_y = 80, 80
    scaled_x = int((ball_position[0] / max_x) * 19)
    scaled_y = int((ball_position[1] / max_y) * 19)

    print(f"Position normalisée de la balle: ({scaled_x}, {scaled_y})")

    return np.array([scaled_x, scaled_y])

