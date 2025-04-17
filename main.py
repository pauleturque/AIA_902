import os
os.environ["SDL_AUDIODRIVER"] = "dummy"

from core.agent import QLearningAgent
from envs.environment import Environment
from utils.file_manager import save_q_table, load_q_table
from utils.preprocessing import preprocess_state
from utils.state_processing import discretize_state
from ui.display import plot_rewards
import numpy as np  # 👈 nécessaire pour la moyenne

# Init env et agent
env = Environment(render_mode="human")
agent = QLearningAgent(state_size=(80, 80), num_actions=env.num_actions)

# Chargement Q-table
saved_q_table = load_q_table()
if saved_q_table is not None:
    agent.q_table = saved_q_table

# apprentissage
EPISODES = 100
episode_rewards = []

for episode in range(EPISODES):
    state = env.reset()
    processed_state = preprocess_state(state)  # Prétraitement de l'image

    discrete_state = discretize_state(processed_state, agent.state_size)
    print(f"🎬 Épisode {episode + 1}/{EPISODES}")

    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(discrete_state)
        new_state, reward, done = env.step(action)
        processed_new_state = preprocess_state(new_state)
        new_discrete_state = discretize_state(processed_new_state, agent.state_size)

        agent.update_q_table(discrete_state, action, reward, new_discrete_state)
        discrete_state = new_discrete_state
        total_reward += reward

    agent.decay_epsilon()
    episode_rewards.append(total_reward)

    # 🔥 Affichage épisode + stats
    print(f"✅ Score: {total_reward:.2f} — Epsilon: {agent.epsilon:.4f}")

    # 📊 Moyenne tous les 100 épisodes
    if (episode + 1) % 100 == 0:
        moyenne = np.mean(episode_rewards[-100:])
        print(f"📈 Moyenne des 100 derniers épisodes : {moyenne:.2f}")

    # 💾 Sauvegarde
    save_q_table(agent.q_table)

env.close()

# 📊 Affichage global
plot_rewards(episode_rewards)
