import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, num_actions, learning_rate=0.1, discount=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # init Q-table
        self.q_table = np.random.uniform(low=-2, high=0, size=(20, 20, num_actions))

    #choix action en fonction d'epsilon
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.num_actions - 1)  # explo
        return np.argmax(self.q_table[state])  # exploit

    # appli formule q-learning / màj q table
    def update_q_table(self, state, action, reward, new_state):
        # Vérification des limites des indices
        if not np.all([0 <= new_state[0] < self.state_size[0], 0 <= new_state[1] < self.state_size[1]]):
            print(f"ERREUR: new_state {new_state} hors limites")
            return

        max_future_q = np.max(self.q_table[new_state])
        current_q = self.q_table[state][action]
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount * max_future_q)
        self.q_table[state][action] = new_q

    # réduc epsilon pour exploitation + que exploration
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
