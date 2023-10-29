import numpy as np
import gymnasium as gym
import stable_baselines3
import cv2
import os
import time

from stable_baselines3 import PPO
from custom_environment import SnekEnv12

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

# save on best reward, function provided in stable_baselines3 examples
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir

        self.save_path = os.path.join(log_dir, 'best_model')

        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                        self.model.save(self.save_path)

        return True

TIMESTEPS = 20_000
version = 12

env = SnekEnv12()

train_model = "PPO"

# Create log dir
log_dir = f"logs/snek-{version}/{train_model}-{int(time.time())}/"

os.makedirs(log_dir, exist_ok=True)

# wrap the environment with a monitor that records a CSV for the save best agent function
env = Monitor(env, log_dir)
env.reset()

# verbose and logging also in callbacks
model = PPO("MlpPolicy", env, tensorboard_log=log_dir, verbose=1)

callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, verbose=1)

while True:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=train_model, callback=callback)


 