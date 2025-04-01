import ale_py
import gymnasium as gym

from ale_py import ALEInterface
ale = ALEInterface()

gym.register_envs(ale_py)

#états possibles
class Environment:
    def __init__(self, render_mode="rgb_array"):
        self.env = gym.make('ALE/Pong-v5', render_mode=render_mode)
        self.num_actions = self.env.action_space.n

    def reset(self):
        state, _ = self.env.reset()
        return state

    #action + renvoi infos
    def step(self, action):
        new_state, reward, done, _, _ = self.env.step(action)
        return new_state, reward, done

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
