import numpy as np
import gym
import stable_baselines3
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


# HUOM. train mallin ja load mallin täytyy olla samat
from custom_environment import SnekEnv12

# If you want to render the game, goes a lot faster without rendering, but you can only see the snake length
RENDER = True

TIMESTEPS = 5_000
version = 12

env = SnekEnv12()
env.reset()

model_folder = "PPO-1674319025"

model_path = f"logs/snek-{version}/{model_folder}/best_model.zip"

model = PPO.load(model_path, env=env)

# Enjoy trained agent
obs = env.reset()
snake_lengths = []

done_count = 0

# while loop has to be here since for loop would only render x amount of steps
# but a while loop checks how many times the environment is completed
while done_count <= 100:
    # fully deterministic models sometimes get stuck
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)

    if RENDER:
        # "robot" or "human"
        # "robot" is 5x faster than human render
        env.render(mode="human")

    # if the snake is stuck and spinning in circles, this resets it

    if done:
        print(info["length"])
        snake_lengths.append(info["length"])
        obs = env.reset()
        done_count += 1

print(f"avg: {sum(snake_lengths) / len(snake_lengths)}")
print(f"all lengths:{sorted(snake_lengths)}")
print(f"longest snake: {sorted(snake_lengths)[-1]}")
