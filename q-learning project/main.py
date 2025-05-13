import os
os.environ["SDL_AUDIODRIVER"] = "dummy"

from core.agent import QLearningAgent
from envs.environment import Environment
from utils.file_manager import save_q_table, load_q_table
from utils.preprocessing import preprocess_state
from ui.display import plot_rewards
import numpy as np
import psutil
import sys

# --- DÉBUT LOG RAM & ÉTAT PAR FRAME ---
process       = psutil.Process(os.getpid())
log_path      = "ram_etat_par_frame.txt"
frame_counter = 0

with open(log_path, "w") as f:
    f.write("frame\tram_MB\tball_x\tball_y\tball_size_B\tpaddles_size_B\tpaddle_positions\n")

def log_frame_info(ball_position, paddle_positions):
    global frame_counter
    ram_mb = process.memory_info().rss / (1024 * 1024)
    size_ball    = sys.getsizeof(ball_position)
    size_paddles = sys.getsizeof(paddle_positions) + sum(sys.getsizeof(p) for p in paddle_positions)

    with open(log_path, "a") as f:
        f.write(
            f"{frame_counter}\t"
            f"{ram_mb:.2f}\t"
            f"{ball_position[0]}\t"
            f"{ball_position[1]}\t"
            f"{size_ball}\t"
            f"{size_paddles}\t"
            f"{paddle_positions}\n"
        )
    frame_counter += 1
# --- FIN LOG ---

# Init env et agent
env = Environment(render_mode="human")
allowed_actions = [0, 2, 3]
EPISODES = 100000
agent = QLearningAgent(state_size=(20, 20), action_list=allowed_actions, num_actions=len(allowed_actions), total_episodes=EPISODES)

# Chargement Q-table
saved_q_table = load_q_table()
if saved_q_table is not None:
    agent.q_table = saved_q_table

episode_rewards = []
best_mean_reward = float('-inf')  # Pour sauvegarder la meilleure Q-table

for episode in range(EPISODES):
    state = env.reset()
    discrete_state, _ = preprocess_state(state, grid_size=agent.state_size)

    print(f"🎬 Épisode {episode + 1}/{EPISODES}")
    total_reward = 0
    done = False
    defaites = 0
    punitions_legères = 0

    while not done:
        action = agent.choose_action(discrete_state)
        new_state, reward, done, _, _ = env.step(action)

        new_discrete_state, paddle_positions = preprocess_state(new_state, grid_size=agent.state_size)

        # Log RAM + état par frame
        log_frame_info(new_discrete_state, paddle_positions)

        agent.update_q_table(discrete_state, action, reward, new_discrete_state)
        discrete_state = new_discrete_state
        total_reward += reward

        if reward == -0.1 or reward == -1:
            punitions_legères += 1
        elif reward == -10:
            defaites += 1

    agent.decay_epsilon(current_episode=episode)
    episode_rewards.append(total_reward)

    # ✅ Résumé de l'épisode
    print(f"✅ Score: {total_reward:.2f} — Epsilon: {agent.epsilon:.4f}")
    print(f"📊 Récapitulatif de l’épisode :")
    print(f"💢 Punitions lourdes : {defaites}")
    print(f"💥 Pénalités mineures : {punitions_legères}")
    print(f"🏁 Récompense totale : {total_reward:.2f}")

    # 📈 Moyenne tous les 100 épisodes
    if (episode + 1) % 100 == 0:
        moyenne = np.mean(episode_rewards[-100:])
        print(f"📈 Moyenne des 100 derniers épisodes : {moyenne:.2f}")

        if moyenne > best_mean_reward:
            print(f"🔍 Moyenne actuelle: {moyenne:.2f}, Meilleure moyenne précédente: {best_mean_reward:.2f}")
            best_mean_reward = moyenne
            save_q_table(agent.q_table, filename="best_q_table.pkl")
            print(f"🎯 Nouvelle meilleure moyenne atteinte ! Q-table sauvegardée ✅")

    # 💾 Sauvegarde courante
    save_q_table(agent.q_table)

env.close()

# 📊 Graphe final
plot_rewards(episode_rewards)
