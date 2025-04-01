import os
os.environ["SDL_AUDIODRIVER"] = "dummy"

from core.agent import QLearningAgent
from envs.environment import Environment
from utils.file_manager import save_q_table, load_q_table
from utils.preprocessing import preprocess_state
from utils.state_processing import discretize_state
from ui.display import plot_rewards

# Init env et agent
env = Environment(render_mode="human")
agent = QLearningAgent(state_size=(20, 20), num_actions=env.num_actions)

# Chargement Q-table
saved_q_table = load_q_table()
if saved_q_table is not None:
    agent.q_table = saved_q_table

#apprentissage
EPISODES = 1000
episode_rewards = []

for episode in range(EPISODES):
    state = env.reset()
    processed_state = preprocess_state(state)  # Prétraitement img
    discrete_state = discretize_state(processed_state, agent.state_size)  # Discrétisation état
    print(f"Episode  {episode} / {EPISODES} ")

    total_reward = 0
    done = False

    while not done:
        # choix action
        action = agent.choose_action(discrete_state)

        # exécution
        new_state, reward, done = env.step(action)
        processed_new_state = preprocess_state(new_state)
        new_discrete_state = discretize_state(processed_new_state, agent.state_size)

        # MàJ Q-table
        agent.update_q_table(discrete_state, action, reward, new_discrete_state)

        # etat suivant
        discrete_state = new_discrete_state
        total_reward += reward

    # diminution explo epsilon
    agent.decay_epsilon()

    # sauvegarde recwards
    episode_rewards.append(total_reward)
    print(f"Épisode {episode + 1}/{EPISODES}, Score: {total_reward}, EPSILON: {agent.epsilon:.4f}")

    # sauvegarde qtable après 100 épisodes
    if episode % 100 == 0:
        save_q_table(agent.q_table)

env.close()

plot_rewards(episode_rewards)
