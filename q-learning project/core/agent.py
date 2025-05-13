# core/agent.py

import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_list, num_actions, total_episodes):
        self.state_size      = state_size
        self.action_list     = action_list
        self.num_actions     = num_actions
        self.total_episodes  = total_episodes

        self.q_table         = np.zeros(state_size + (num_actions,))
        self.learning_rate   = 0.2
        self.discount_factor = 0.99
        self.epsilon         = 1.0  # exploration au départ
        self.epsilon_min     = 0.05
        self.epsilon_decay   = 0.990

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.action_list)
        else:
            action_index = np.argmax(self.q_table[state])
            return self.action_list[action_index]

    def update_q_table(self, state, action, reward, next_state):
        action_index = self.action_list.index(action)
        best_next_action = np.max(self.q_table[next_state])
        td_target = reward + self.discount_factor * best_next_action
        td_error  = td_target - self.q_table[state][action_index]
        self.q_table[state][action_index] += self.learning_rate * td_error

    def decay_epsilon(self, current_episode):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
