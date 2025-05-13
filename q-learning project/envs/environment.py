import ale_py
import gymnasium as gym
import numpy as np

from ale_py import ALEInterface
from utils.preprocessing import preprocess_state

ale = ALEInterface()
gym.register_envs(ale_py)

class Environment(gym.Env):
    def __init__(self, render_mode="rgb_array"):
        super(Environment, self).__init__()
        self.render_mode = "human"
        self.env = gym.make('ALE/Pong-v5', render_mode="human")
        self.action_space = gym.spaces.Discrete(4)  # Actions possibles (0, 1, 2, 3)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)

    def reset(self):
        # Remise à zéro de l'état initial
        initial_state = np.zeros((210, 160, 3), dtype=np.uint8)
        return initial_state

    def step(self, action):
        """
        Étape dans l'environnement selon l'action prise.
        :param action: L'action choisie par l'agent.
        :return: (new_state, reward, done)
        """
        # Simule l'environnement ici, selon l'action prise
        # Exemple : la balle bouge, on vérifie si elle touche les bords ou les paddles
        new_state = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)  # Exemple de nouvel état
        reward = -0.1  # Par défaut, une petite pénalité
        done = False  # La partie continue par défaut

        # Logique de récompense (exemple simplifié)
        if action == 0:  # Exemple d'action où la récompense est -1
            reward = -1
        elif action == 1:  # Autre action avec une récompense différente
            reward = -0.5
        elif action == 2:  # Action qui mène à un échec
            reward = -10
            done = True
        elif action == 3:  # Une bonne action
            reward = 1

        return new_state, reward, done, {}, {}
