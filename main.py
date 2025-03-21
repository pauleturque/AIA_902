import ale_py
import gymnasium as gym

from ale_py import ALEInterface
ale = ALEInterface()

gym.register_envs(ale_py)

env = gym.make('ALE/Pong-v5', render_mode="human")

obs, info = env.reset()

for _ in range(500):
    action = env.action_space.sample()  # Action aléatoire
    obs, reward, terminated, truncated, info = env.step(action)  # Jouer

    # Rendu du jeu
    env.render()

    if terminated or truncated:
        obs, _ = env.reset()

env.close()

