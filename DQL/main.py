
import os
os.environ["SDL_AUDIODRIVER"] = "dummy"



import ale_py
import gymnasium as gym
import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
from collections import deque
from core.dqn_agent import DQNAgent
from ale_py import ALEInterface

# Configuration de l'environnement
ale = ALEInterface()
gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", render_mode="human")

# Prétraitement : grayscale + resize
def preprocess(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return obs / 255.0

# Empilement des frames
frame_stack = deque(maxlen=4)
obs, _ = env.reset()
frame = preprocess(obs)
for _ in range(4):
    frame_stack.append(frame)
state = np.stack(frame_stack, axis=0)

# Initialisation de l'agent
obs_shape = state.shape
num_actions = env.action_space.n
agent = DQNAgent(obs_shape, num_actions, device="cuda" if torch.cuda.is_available() else "cpu")

# Variables de l'entraînement
EPISODES = 1000
TARGET_UPDATE_FREQ = 1000
LEARNING_STARTS = 10000
TRAIN_FREQ = 4
step_count = 0

# Logs pour suivre les progrès
episode_rewards = []  # Récompenses par épisode
epsilon_values = []   # Valeur d'epsilon par épisode

# Boucle principale d'entraînement
for episode in range(EPISODES):
    obs, _ = env.reset()
    frame = preprocess(obs)
    frame_stack.clear()
    for _ in range(4):
        frame_stack.append(frame)
    state = np.stack(frame_stack, axis=0)
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        frame = preprocess(next_obs)
        frame_stack.append(frame)
        next_state = np.stack(frame_stack, axis=0)
        clipped_reward = np.sign(reward)
        done = terminated or truncated

        agent.buffer.push(state, action, clipped_reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(agent.buffer) > LEARNING_STARTS and step_count % TRAIN_FREQ == 0:
            agent.update()

        if step_count % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        step_count += 1

    agent.decay_epsilon()

    # Stockage des logs
    episode_rewards.append(total_reward)
    epsilon_values.append(agent.epsilon)

    print(f"Épisode {episode} — Total reward: {total_reward:.2f} — Epsilon: {agent.epsilon:.3f}")

# Visualisation des résultats
# Graphique des récompenses par épisode
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.xlabel("Épisode")
plt.ylabel("Récompense Totale")
plt.title("Récompense Totale par Épisode")

# Graphique de l'évolution de l'épsilon
plt.subplot(1, 2, 2)
plt.plot(epsilon_values, color='red')
plt.xlabel("Épisode")
plt.ylabel("Epsilon")
plt.title("Évolution de l'Epsilon")

# Affichage des graphiques
plt.tight_layout()
plt.show()




