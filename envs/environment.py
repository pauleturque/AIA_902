import ale_py
import gymnasium as gym

from ale_py import ALEInterface
ale = ALEInterface()

gym.register_envs(ale_py)

import numpy as np


def check_near_miss(ball_x, paddle_x):
    return 2 < abs(ball_x - paddle_x) < 5


class Environment:
    def __init__(self, render_mode="rgb_array"):
        self.env = gym.make('ALE/Pong-v5', render_mode=render_mode)
        self.num_actions = self.env.action_space.n
        self.current_step = 0
        self.last_contact = False
        self.total_reward = 0.0

    def reset(self):
        self.current_step = 0
        self.last_contact = False
        self.total_reward = 0.0
        state, _ = self.env.reset()
        print("=== Nouvel épisode ===")
        return state

    def step(self, action):
        self.current_step += 1
        new_state, reward, done, _, info = self.env.step(action)

        # Analyse de l'état pour trouver les positions (si l'état est une image)
        ball_x, paddle_x = self.extract_positions(new_state)

        # Affichage des positions
        #print(f"[Step {self.current_step}] Ball X: {ball_x}, Paddle X: {paddle_x}")

        # Récompense si la balle touche la raquette
        if self.check_ball_contact(ball_x, paddle_x):
            if not self.last_contact:
                print("🎯 Contact balle-raquette ! +1 récompense")
                reward += 1.0
                self.last_contact = True

        elif check_near_miss(ball_x, paddle_x):
            print("⚠️ Near miss détecté (+0.2)")
            reward += 0.2

        if self.current_step % 100 == 0:
            print("💢 Punition (-0.1) pour stagnation")
            reward -= 0.1

        self.total_reward += reward
        print(f"🏅 Récompense (step) : {reward:.2f} | 🧮 Récompense cumulée : {self.total_reward:.2f}")

        return new_state, reward, done

    # def extract_positions(self, state):
    #     """Extrait les positions de la balle et de la raquette à partir de l'état (image)"""
    #     # Recherche d'un pixel spécifique représentant la balle
    #     ball_position = np.where(state == 255)  # Par exemple, si la balle est blanche
    #
    #     # Supposons que la raquette soit un bloc noir, on peut ajuster cette logique
    #     paddle_position = np.where(state == 0)  # Cette condition devra être plus spécifique
    #
    #     # Renvoie les positions trouvées
    #     ball_x = ball_position[1][0] if len(ball_position[1]) > 0 else 0
    #     paddle_x = paddle_position[1][0] if len(paddle_position[1]) > 0 else 0
    #
    #     return ball_x, paddle_x

    def extract_positions(self, state):
        """Extrait les positions de la balle et de la raquette à partir de l'état (image)"""
        height, width, _ = state.shape


        # Recherche pixels blancs pour la balle
        ball_position = np.where(np.all(state[:height // 3] == [255, 255, 255],
                                        axis=-1))

        # Recherche des pixels verts pour la raquette
        paddle_position = np.where(np.all(state[height // 3:height] == [0, 255, 0],
                                          axis=-1))

        # Si la balle est trouvée
        if len(ball_position[1]) > 0:
            ball_x = ball_position[1][0]
            ball_y = ball_position[0][0]
        else:
            ball_x = 0
            ball_y = 0

        # Si la raquette est trouvée
        if len(paddle_position[1]) > 0:
            paddle_x = paddle_position[1][0]
            paddle_y = paddle_position[0][0]
        else:
            paddle_x = 0
            paddle_y = 0

        # positions extraites pour débogage
        #print(
            #f"Positions extraites - Balle X: {ball_x}, Balle Y: {ball_y}, Raquette X: {paddle_x}, Raquette Y: {paddle_y}")

        return ball_x, paddle_x

    def check_ball_contact(self, ball_x, paddle_x):
        return abs(ball_x - paddle_x) <= 1

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

