import numpy as np

def discretize_state(state, state_size):

    #conversion un état continu (image traitée) en indices discrets pour la Q-table
    #state contient les coordonnées de la balle

    discrete_x, discrete_y = state  # `state` est déjà un tuple (x, y)

    # vérif plage valide
    discrete_x = np.clip(discrete_x, 0, state_size[0] - 1)
    discrete_y = np.clip(discrete_y, 0, state_size[1] - 1)

    return discrete_x, discrete_y  # tuple
