import ale_py
import gymnasium as gym

from ale_py import ALEInterface
from utils.preprocessing import preprocess_state

ale = ALEInterface()
gym.register_envs(ale_py)



class Environment:
    def __init__(self, render_mode="human"):
        self.penalties = None
        self.env = gym.make('ALE/Pong-v5', render_mode=render_mode)
        self.num_actions = self.env.action_space.n
        self.current_step = 0
        self.last_contact = False
        self.total_reward = 0.0
        self.contact_hits = 0
        self.near_misses = 0

        # Compteurs pour analyse
        self.reset_reward_counters()

    def reset_reward_counters(self):
        self.contact_hits = 0
        self.near_misses = 0
        self.penalties = 0

    def reset(self):
        self.current_step = 0
        self.last_contact = False
        self.total_reward = 0.0
        self.contact_hits = 0
        self.near_misses = 0
        state, _ = self.env.reset()
        self.reset_reward_counters()
        return state

    def step(self, action):
        self.current_step += 1
        new_state, reward, done, _, info = self.env.step(action)

        # Prétraitement
        ball_position, paddle_positions = preprocess_state(new_state)
        ball_x, ball_y = ball_position[0], ball_position[1]
        paddle_x, paddle_y = paddle_positions[0] if paddle_positions else (0, 0)

        # Contact balle-raquette
        if self.check_ball_contact(ball_x, ball_y, paddle_x, paddle_y):
            if not self.last_contact:
                reward += 1.0
                self.contact_hits += 1
                self.last_contact = True
            else:
                self.last_contact = False


        if self.current_step % 100 == 0:

            reward -= 0.1

        # Presque touché
        if self.check_near_miss(ball_x, ball_y, paddle_x, paddle_y):
            reward += 0.2
            self.near_misses += 1

        # Punition périodique
        if self.current_step % 100 == 0:
            reward -= 0.1
            self.penalties += 1

        self.total_reward += reward

        if done:
            self.log_episode_summary()

        return new_state, reward, done

    def check_ball_contact(self, ball_x, ball_y, paddle_x, paddle_y, paddle_width=3, paddle_height=15):
        return (
            paddle_x <= ball_x <= paddle_x + paddle_width and
            paddle_y <= ball_y <= paddle_y + paddle_height
        )

    def check_near_miss(self, ball_x, ball_y, paddle_x, paddle_y, tolerance_x=5, tolerance_y=20, paddle_height=15):
        return (
            2 < abs(ball_x - paddle_x) < tolerance_x and
            (paddle_y - 5 <= ball_y <= paddle_y + paddle_height + 5)
        )

    def log_episode_summary(self):
        print("\n📊 Récapitulatif de l’épisode :")
        print(f"🎯 Contacts réussis : {self.contact_hits}")
        print(f"⚠️  Near misses       : {self.near_misses}")
        print(f"💢 Punitions         : {self.penalties}")
        print(f"🏁 Récompense totale : {self.total_reward:.2f}\n")

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

