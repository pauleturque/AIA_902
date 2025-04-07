import cv2
import numpy as np

def preprocess_state(state):
    # niveaux de gris
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

    # Redimensionne l'image
    #resized = cv2.resize(gray, (80, 80), interpolation=cv2.INTER_AREA)

    # Binarisation
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Détection des contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_position = (0, 0)
    paddle_positions = []

    height, width = state.shape[:2]

    top_border = int(height * 0.16)  # supprime 20% de la hauteur totale
    bottom_border = int(height * 0.93) # idem en bas

    print(f"Zone de jeu : top_border = {top_border}, bottom_border = {bottom_border}")

    # Limites visuelles de la zone de jeu
    cv2.rectangle(state, (0, top_border), (width, bottom_border), (0, 0, 255), 2)  # zone de jeu
    cv2.rectangle(state, (0, 0), (width, top_border), (0, 255, 0), 2)  #  hors jeu en haut
    cv2.rectangle(state, (0, bottom_border), (width, height), (0, 255, 0), 2) #hors jeu en bas

    # Test affichage zone de jeu
    # cv2.imshow('Zone de jeu', state)
    # cv2.waitKey(0)

    #print("Contours détectés :")
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Affiche les contours détectés en temps réel
        #print(f"Contour trouvé - X: {x}, Y: {y}, Width: {w}, Height: {h}")

        # contour dans la zone de jeu valide
        if y < top_border or y + h > bottom_border:
            #print(f"Contour hors zone de jeu exclu - X: {x}, Y: {y}, Width: {w}, Height: {h}")
            continue

        # Filtrage  taille des objets
        if w < 5 and h < 5:  # balle carré
            ball_position = (x, y)
            print(f"Balle détectée - X: {x}, Y: {y}")
        elif 10 < w < 30 and 60 < h < 80:
            paddle_positions.append((x, y))
            print(f"Raquette détectée - X: {x}, Y: {y}")

    # Normalisation pour éviter dépassements
    max_x, max_y = 210, 160
    scaled_ball_x = int((ball_position[0] / max_x) * 19)
    scaled_ball_y = int((ball_position[1] / max_y) * 19)

    paddle_positions_scaled = []
    for paddle in paddle_positions:
        scaled_paddle_x = int((paddle[0] / max_x) * 19)
        scaled_paddle_y = int((paddle[1] / max_y) * 19)
        paddle_positions_scaled.append((scaled_paddle_x, scaled_paddle_y))

    # positions finales
    #print(f"Balle (normalisée) - X: {scaled_ball_x}, Y: {scaled_ball_y}")
    #print(f"Raquettes (normalisées) - {paddle_positions_scaled}")

    # Dessiner les rectangles autour de la balle et des raquettes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if y >= top_border and y + h <= bottom_border:
            if w < 5 and h < 5:  # Balle
                cv2.rectangle(state, (x, y), (x + w, y + h), (0, 0, 255), 2)  # rouge
            elif  w < 5 and 10 < h < 80:  # Raquettes
                if x < 40:  # Raquette gauche
                    cv2.rectangle(state, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Jaune
                else:  # Raquette droite
                    cv2.rectangle(state, (x, y), (x + w, y + h), (255, 0, 255), 2)  # Violet

    # Afficher l'image finale avec les contours
    #cv2.imshow('Contours détectés', state)

    # Attendre une touche pour fermer les fenêtres
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return np.array([scaled_ball_x, scaled_ball_y]), paddle_positions_scaled

