import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2
import math
import random
from Pong_play_env import Pong_play_env

black = (0,0,0)
white = (255,255,255)
gray = (200,200,200)

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)

class Pong_train_env(Pong_play_env):

    def __init__(self):
        super().__init__()

    # the only difference is that the paddle opposing the ml model is big so it passively blocks almost all shots
    def reset_game(self):

        self.reward = 0

        self.paddle1_height = self.height - 60
        self.paddle2_height = 80

        self.img = np.zeros((self.width, self.height, 3), dtype='uint8') + 20

        self.ball_speed_x = random.choice([-5,5]) 
        self.ball_speed_y = random.uniform(-10, 10)

        self.paddle1_y = 30
        self.paddle2_y = self.height / 2

        self.paddle1_speed = 0
        self.paddle2_speed = 0

        self.ball_x = self.width / 2
        self.ball_y = self.height / 2

        self.ball_spin = random.uniform(-10, 10)
        self.ball_spin_angle = 0

