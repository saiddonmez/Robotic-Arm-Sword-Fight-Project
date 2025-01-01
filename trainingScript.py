import stable_baselines3
from raiSimulationEnvATKDEF import RobotSimEnv
import robotic as ry
import numpy as np
import time

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3 import PPO
import gymnasium as gym

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

class StaticOpponentWrapper(gym.Wrapper):
    """
    A wrapper for a 1v1 environment where one agent is controlled by a fixed, pre-trained policy.
    """
    def __init__(self, env, static_policy_attacker, static_policy_defender,staticAttacker=True,test=False):
        super(StaticOpponentWrapper, self).__init__(env)
        self.static_policy_attacker = static_policy_attacker
        self.static_policy_defender = static_policy_defender
        self.staticAttacker = staticAttacker
        self.test = test
        self.initiallyObserved = False

    def step(self, action):
        # Get the static opponent's action
        obs = self.env.state  # Modify this based on how your env works
        if not self.initialObserved:
            self.initialObs = obs
            self.initialObserved = True
        if self.staticAttacker:
            attacker_action, _ = self.static_policy_attacker.predict(self.initiallyObserved, deterministic=True)
        else:
            attacker_action, _ = self.static_policy_attacker.predict(obs, deterministic=True)
        
        defender_action, _ = self.static_policy_defender.predict(obs, deterministic=True)
        # Combine both actions into a joint action
        if self.staticAttacker:
            joint_action = (attacker_action, action)
        else:
            joint_action = (action,defender_action)
        
        if self.test:
            joint_action = (attacker_action, defender_action)

        # Step the environment with both actions
        obs, reward, done, truncated, info = self.env.step(joint_action)
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# Usage
checkpoint_callback = CustomCheckpointCallback(
    save_freq=100000, save_path='./checkpoints/', verbose=1
)
reward_logger = RewardLoggerCallback(log_file="rewards_log_2M_trial1.csv")

callbacks = CallbackList([checkpoint_callback, reward_logger])


env = RobotSimEnv(render_mode='human',staticAttacker=True)

attackModel = PPO.load("sword_model_best")
defenceModel = PPO.load("shield_model_best")

wrapped_env = StaticOpponentWrapper(env, attackModel,defenceModel,attacker=True)

policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256])]  # 'pi' is the actor network, 'vf' is the critic network
)

model = PPO("MlpPolicy", wrapped_env, policy_kwargs=policy_kwargs, verbose=0, device='cpu')   
#model = SAC("MlpPolicy", env, verbose=0)   

model.learn(total_timesteps=100000, callback=callbacks)

model.save("ppo_100k_defense_trial1")