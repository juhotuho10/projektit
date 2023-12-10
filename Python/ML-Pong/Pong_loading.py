import numpy as np
import gymnasium as gym
import stable_baselines3
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from Pong_train_env import Pong_train_env

RENDER = True

TIMESTEPS = 5_000
version = 1

env = Pong_train_env()
env.reset()

model_folder = "1702140034"

model_path = f"logs/Pong-{version}/{model_folder}/best_model.zip"

model = PPO.load(model_path, env=env)

# Enjoy trained agent
obs, _ = env.reset()

done_count = 0

# while loop has to be here since for loop would only render x amount of steps
# but a while loop checks how many times the environment is completed
while done_count <= 100:

    action, _states = model.predict(obs, deterministic=True)

    obs, reward, done, truncated, info = env.step(action)
 
    #print(obs)

    if RENDER:
        env.render()

    if done:
        obs, _ = env.reset()
        done_count += 1
