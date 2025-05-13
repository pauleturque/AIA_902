import torch
import torch.nn.functional as F
import numpy as np
import random
from models.q_network import QNetwork
from utils.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, obs_shape, num_actions, device="cpu"):
        self.device = device
        self.num_actions = num_actions

        # Paramètres d'exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999  # Décroissance exponentielle

        # Hyperparamètres DQN
        self.gamma = 0.99
        self.lr = 1e-4
        self.batch_size = 32

        # Buffer de replay
        self.buffer = ReplayBuffer()

        # Réseaux Q (policy + target)
        self.policy_net = QNetwork(obs_shape, num_actions).to(device)
        self.target_net = QNetwork(obs_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def act(self, state):
        """Choisit une action selon une stratégie epsilon-greedy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()

    def update(self):
        """Met à jour le réseau de Q à partir d’un batch échantillonné du buffer."""
        if len(self.buffer) < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = list(zip(*transitions))

        state = torch.tensor(np.stack(batch[0]), dtype=torch.float32).to(self.device)
        action = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1).to(self.device)
        reward = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state = torch.tensor(np.stack(batch[3]), dtype=torch.float32).to(self.device)
        done = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.policy_net(state).gather(1, action)
        next_q_values = self.target_net(next_state).max(1)[0].unsqueeze(1).detach()
        target = reward + self.gamma * next_q_values * (1 - done)

        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        """Copie les poids du réseau de policy vers le réseau target."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Réduit epsilon de manière exponentielle."""
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 100000
