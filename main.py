import gymnasium as gym
import ale_py
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

from ale_py import ALEInterface
ale = ALEInterface()

gym.register_envs(ale_py)

env = gym.make('ALE/Pong-v5', render_mode="human")

# Paramètres agent
alpha = 0.1  # Taux d'apprentissage
gamma = 0.9  # Facteur de discount
epsilon = 0.1  # Exploration exploitation
num_episodes = 1000

num_actions = env.action_space.n

# Initialisation de la Q-table
state_space_size = 6  # Choisis une taille appropriée pour l'espace d'état discretisé
Q = np.zeros((state_space_size, num_actions))

# Fonction de prétraitement de l'image
def preprocess_state(state):
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (40, 40), interpolation=cv2.INTER_AREA)
    return resized.flatten()  # Aplatir l'image traitée


# Discrétisation de l'état : mapper l'état continu vers un espace discret
def discretize_state(state):
    # Normalise l'état entre 0 et 1
    state = state / 255.0
    # Aplatit image en 1D
    flattened_state = state.flatten()

    # Calcul indice discret pour cet état
    state_idx = int(np.sum(flattened_state) % state_space_size)  # Juste un exemple simple
    return state_idx


# Fonction epsilon-greedy pour choisir une action
def choose_action(state, epsilon):
    state_idx = discretize_state(state)  # Discrétiser état
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Exploration
    else:
        return np.argmax(Q[state_idx])  # Exploitation

# Stockage score de chaque épisode
episode_rewards = []

# Eentraînement agent
for episode in range(num_episodes):
    state, info = env.reset()
    state = preprocess_state(state)  # Prétraitement état
    total_reward = 0
    done = False
    while not done:
        # Choix l'action
        action = choose_action(state, epsilon)

        # Aaction est valide
        if action < 0 or action >= env.action_space.n:
            action = random.randint(0, env.action_space.n - 1)

        # Appliquer action dans l'environnement
        next_state, reward, terminated, truncated, info = env.step(action)

        # Prétraitement du nouvel état
        next_state = preprocess_state(next_state)

        # Discrétiser l'état suivant
        state_idx = discretize_state(state)
        next_state_idx = discretize_state(next_state)

        # Mise à jour Q-table
        best_next_action = np.argmax(Q[next_state_idx])
        Q[state_idx, action] = Q[state_idx, action] + alpha * (
                    reward + gamma * Q[next_state_idx, best_next_action] - Q[state_idx, action])

        # Passer à l'état suivant
        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)  # Ajoute la récompense de l'épisode à la liste
    print(f"Episode {episode + 1}/{num_episodes} finished with total reward: {total_reward}")

# Traçage graphique des scores
plt.plot(episode_rewards)
plt.title('Performance de l\'agent au fil des épisodes')
plt.xlabel('Épisodes')
plt.ylabel('Récompense totale')
plt.show()

env.close()
