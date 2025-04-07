import numpy as np

def discretize_state(state, state_size):
    # state contient les coordonnées de la balle (x, y)
    # state_size est la taille de l'espace discret

    # coordonnées de la balle
    discrete_x, discrete_y = state[0]

    # x et y sont des scalaires ?
    discrete_x = int(discrete_x[0]) if isinstance(discrete_x, np.ndarray) else int(discrete_x)
    discrete_y = int(discrete_y[0]) if isinstance(discrete_y, np.ndarray) else int(discrete_y)

    #  coordonnées avant  discrétisation
    #print(f"Avant discrétisation - Balle X: {discrete_x}, Y: {discrete_y}")

    # Discrétise en fonction de la taille de l'espace d'état
    discrete_x = np.clip(discrete_x, 0, state_size[0] - 1)
    discrete_y = np.clip(discrete_y, 0, state_size[1] - 1)

    # Affichage après la discrétisation
    #print(f"Après discrétisation - Balle X: {discrete_x}, Y: {discrete_y}")

    return discrete_x, discrete_y  # Retourner un tuple d'indices discrets






