from PIL import Image

import dxcam

import win32gui
import time
import numpy as np
from pynput.keyboard import Key, Controller
import gymnasium as gym
from gymnasium import spaces

camera = dxcam.create(output_color="GRAY")
keyboard = Controller()

class brawlhalla(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self) -> None:
        super(brawlhalla, self).__init__()

        self.brawl_window, self.hwnd = self.set_window()

        self.action_space = spaces.Discrete(12)
        self.observation_space = spaces.Box(low=0, high=255, shape=(540, 960, 1), dtype=np.uint8)

        self.my_health = 1
        self.enemy_health = 1
        self.my_previous_helth = 1
        self.enemy_previous_health = 1

        self.current_screen = None

        self.keyboard_dict = {0:[Key.left],
                              1:[Key.right],
                              2:[Key.space],
                              3:[Key.down],
                              4:["e"],

                              5:[Key.left, "q"],
                              6:[Key.right, "q"],
                              7:[Key.up, "q"],
                              8:[Key.down, "q"],

                              9:[Key.left, "w"],
                              10:[Key.right, "w"],
                              11:[Key.up, "w"],
                              12:[Key.down, "w"]}
        
        self.previous_keys = []

    def set_window(self):

        toplist, winlist = [], []
        def enum_cb(hwnd, results):
            winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
        win32gui.EnumWindows(enum_cb, toplist)

        brawl_window = [(hwnd, title) for hwnd, title in winlist if 'Brawlhalla' in title][0]

        return brawl_window[0], brawl_window[0]


    def step(self, action):

        for key in self.previous_keys:
            keyboard.release(key)

        self.previous_keys = []

        print(f"predictied action: {self.keyboard_dict[action]}")

        for key in self.keyboard_dict[action]:
            keyboard.press(key)
            self.previous_keys.append(key)

        
        self.current_screen = self.get_observation()

        observation = self.current_screen
        
        reward = self.get_reward()
        
        terminated = False

        truncated = False
        info = {}

        print(f"reward: {reward}\n")

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None):
        self.my_health = 1
        self.enemy_health = 1
        self.my_previous_helth = 1
        self.enemy_previous_health = 1
        self.previous_keys = []

        observation = self.get_observation()

        print("reset")

        return observation, False  # reward, done, info can't be included
    
    def render(self, mode='human'):
        pass

    def close (self):
        for key in [Key.alt, Key.f4]:
            keyboard.press(key)

        for key in [Key.alt, Key.f4]:
            keyboard.release(key)

    def get_screenshot(self):

        frame = 0
        while not isinstance(frame, np.ndarray):
            win32gui.SetForegroundWindow(self.hwnd)
            bbox = win32gui.GetWindowRect(self.hwnd)
            frame = camera.grab(region=bbox)
        
        return frame
    
    def get_observation(self):
        return self.get_screenshot()
    
    def get_health(self):

        self.my_previous_helth = self.my_health
        self.enemy_previous_health = self.enemy_health
    
        self.my_health = self.current_screen[80, 922, 0] / 255

        self.enemy_health = self.current_screen[80, 880, 0] / 255

    
    def get_reward(self):

        self.get_health()

        passive_punishment = 0.01

        my_health_change = self.my_previous_helth - self.my_health
        enemy_health_change = self.enemy_previous_health - self.enemy_health

        if my_health_change < 0:
            return - 10
        elif enemy_health_change < 0:
            return 10
        else:
            return  (enemy_health_change - my_health_change) * 10 - passive_punishment

    
