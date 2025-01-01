import stable_baselines3
from raiSimulationEnv import RobotSimEnv
import robotic as ry
import numpy as np
import time

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3 import PPO


class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_file: str, verbose: int = 0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.log_file = log_file
        self.episode_rewards = []
        self.current_episode_reward = 0

        # Create the log file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("Episode,Total Reward\n")

    def _on_step(self) -> bool:
        # Check if the episode has ended by using `done`
        dones = self.locals["dones"]
        rewards = self.locals["rewards"]

        # Accumulate rewards for the current episode
        self.current_episode_reward += rewards[0]

        # If the episode is done, log the reward
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            with open(self.log_file, 'a') as f:
                f.write(f"{len(self.episode_rewards)},{self.current_episode_reward}\n")
            # Reset the reward counter for the next episode
            self.current_episode_reward = 0

        return True

    def _on_training_end(self) -> None:
        # Optionally summarize results at the end of training
        print("Training finished. Total episodes:", len(self.episode_rewards))
        print("Episode rewards:", self.episode_rewards)

class CustomCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super(CustomCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        # Save the model every `save_freq` steps
        if self.n_calls % self.save_freq == 0:
            model_path = f"{self.save_path}/model_checkpoint_{self.n_calls}_steps.zip"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls} to {model_path}")
        return True

# Usage
checkpoint_callback = CustomCheckpointCallback(
    save_freq=200000, save_path='./checkpoints/', verbose=1
)
reward_logger = RewardLoggerCallback(log_file="rewards_log_2M_trial1.csv")

callbacks = CallbackList([checkpoint_callback, reward_logger])


env = RobotSimEnv(render_mode='human')

policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256])]  # 'pi' is the actor network, 'vf' is the critic network
)

model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0, device='cpu')   
#model = SAC("MlpPolicy", env, verbose=0)   

model.learn(total_timesteps=2000000, callback=callbacks)

model.save("ppo_2M_trial1")