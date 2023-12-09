import numpy as np
import gymnasium as gym
import stable_baselines3
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# HUOM. train mallin ja load mallin täytyy olla samat
from Pong_env import Pong_env

# If you want to render the game, goes a lot faster without rendering, but you can only see the snake length
RENDER = True

TIMESTEPS = 5_000
version = 1

env = Pong_env()
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

    # if the snake is stuck and spinning in circles, this resets it

    if done:
        obs, _ = env.reset()
        done_count += 1
