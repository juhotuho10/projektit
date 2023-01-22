
import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

N_DISCRETE_ACTIONS = 4

SNAKE_LEN_GOAL = 35
# observations length
N_CHANNELS = 12 + SNAKE_LEN_GOAL


def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    score += 1
    return apple_position, score


def collision_with_boundaries(snake_head):
    if snake_head[0] >= 500 or snake_head[0] < 0 or snake_head[1] >= 500 or snake_head[1] < 0:
        return 1
    else:
        return 0


def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0


class SnekEnv2(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(SnekEnv2, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        # self.reset()
        # print(len(self.observation))

        self.SNAKE_LEN_GOAL = 35

        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-500, high=500,  # maximum possible values
                                            # shape = observations length
                                            shape=(12 + self.SNAKE_LEN_GOAL,),
                                            dtype=np.float32)  # HEIGHT, WIDTH for N_CHANNELS

        self.previous_distance = 0
        self.cycle = 0

    def all_observations(self, reset):

        if self.apple_hit:
            apple_hit = 1
        else:
            apple_hit = 0

        self.apple_hit = False

        head_x = self.snake_head[0] / 20
        head_y = self.snake_head[1] / 20

        neck_x = self.snake_position[1][0] / 20
        neck_y = self.snake_position[1][1] / 20

        tail_x = self.snake_position[-1][0] / 20
        tail_y = self.snake_position[-1][1] / 20

        apple_x = self.apple_position[0] / 20
        apple_y = self.apple_position[1] / 20

        self.apple_delta_x = head_x - apple_x
        self.apple_delta_y = head_y - apple_y

        neck_delta_x = head_x - neck_x
        neck_delta_y = head_y - neck_y

        self.snake_len = len(self.snake_position)

        tail_delta_x = head_x - tail_x
        tail_delta_y = head_y - tail_y

        if reset:
            apple_hit = 0

            self.prev_actions = deque(maxlen=self.SNAKE_LEN_GOAL)

            for _ in range(self.SNAKE_LEN_GOAL):
                self.prev_actions.append(-1)  # create null history

        self.observation = [head_x, head_y, apple_x, apple_y, neck_delta_x,
                            neck_delta_y, self.apple_delta_x, self.apple_delta_y, apple_hit,
                            self.snake_len, tail_delta_x, tail_delta_y] + list(self.prev_actions)

        # print(self.observation)

        self.observation = np.array(self.observation)

    def all_rewards(self):

        self.cycle += 1
        if self.cycle > 10000:
            self.done = True

        euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

        self.distance = (euclidean_dist_to_apple / 100)

        # getting closer to the apple is rewarded, not just being close to the apple
        if self.previous_distance > 0.1:
            self.reward = self.previous_distance - self.distance
        else:
            self.reward = 0

        # print(f"{self.previous_distance} - {self.distance} = {self.reward}    D: {(euclidean_dist_to_apple/100)}")

        self.previous_distance = self.distance

        if self.reward < 0:
            self.reward *= 1.1

        if self.apple_hit:
            self.reward = 30 + self.score * 4 + 1.2 ** self.score
            self.previous_distance = 0
            # print(f"we hit the apple, {self.reward}")

        if self.done:
            self.reward -= (30 + self.score * 4 + 1.2 ** self.score)
            self.previous_distance = 0

    def get_info(self):
        return {"length": self.snake_len, "cycle": self.cycle,
                "snake_x": self.snake_head[0], "snake_y": self.snake_head[1]}

    def step(self, action):

        self.prev_actions.append(action)
        self.done = False

        #self.render()

        # cv2.imshow('a', self.img)
        # cv2.waitKey(1)
        # self.img = np.zeros((500, 500, 3), dtype='uint8')
        # # Display Apple
        # cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]), (self.apple_position[0] + 10, self.apple_position[1] + 10),
        #               (0, 0, 255), 3)
        # # Display Snake
        # for position in self.snake_position:
        #     cv2.rectangle(self.img, (position[0], position[1]), (position[0] + 10, position[1] + 10), (0, 255, 0), 3)

        # # Takes step after fixed time
        # t_end = time.time() + 0.03
        # k = -1
        # while time.time() < t_end:
        #     if k == -1:
        #         k = cv2.waitKey(1)
        #     else:
        #         continue

        # 0-Left, 1-Right, 3-Up, 2-Down, q-Break
        # a-Left, d-Right, w-Up, s-Down

        # Change the head position based on the button direction
        if action == 1:
            self.snake_head[0] += 10
        elif action == 0:
            self.snake_head[0] -= 10
        elif action == 2:
            self.snake_head[1] += 10
        elif action == 3:
            self.snake_head[1] -= 10

        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_hit = True
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0, list(self.snake_head))

        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # On collision kill the snake and print the score
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1:
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # self.img = np.zeros((500, 500, 3), dtype='uint8')
            # cv2.putText(self.img, 'Your Score is {}'.format(self.score), (140, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.imshow('a', self.img)
            self.done = True

        '''-----------------------------REWARD----------------------'''
        self.all_rewards()

        '''# print(euclidean_dist_to_apple)
        self.reward = 10+self.score

        self.reward -= (euclidean_dist_to_apple*0.01)**2

        MAX_REWARD = 500

        if self.apple_hit:
            self.reward += (MAX_REWARD+self.score*5)
            print(f"we hit the apple, {self.reward}")

        if self.done:
            self.reward -= 700 + (self.score*10)

        #if self.reward > 3:
        #    print(self.score, self.reward)'''

        '''-----------------------------OBSERVATION----------------------'''

        self.all_observations(reset=False)

        '''---------------------------INFO-------------------------------'''

        info = self.get_info()

        '''---------------------------RETURN-------------------------------'''
        return self.observation, self.reward, self.done, info

    def reset(self):
        self.done = False

        self.img = np.zeros((500, 500, 3), dtype='uint8')
        # Initial Snake and Apple position
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
        self.score = 0
        self.reward = 0
        self.cycle = 0
        self.apple_hit = False
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [250, 250]

        self.cycle = 0

        # obs examples: head_x, head_y, apple_x, apple_y, self.apple_delta_x, self.apple_delta_y,
        # snake_lengths, previous_moves, tail_x , tail_y

        self.all_observations(reset=True)

        return self.observation  # reward, done, info can't be included

    def render(self, mode='robot'):

        cv2.imshow('a', self.img)
        cv2.waitKey(1)
        self.img = np.zeros((500, 500, 3), dtype='uint8')
        # Display Apple
        cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]),
                      (self.apple_position[0] + 10, self.apple_position[1] + 10),
                      (0, 0, 255), 3)
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(self.img, (position[0], position[1]), (position[0] + 10, position[1] + 10), (0, 255, 0), 3)

        if mode == "human":
            speed = 0.005
        else:
            speed = 0.001

        # Takes step after fixed time
        t_end = time.time() + speed
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue
        # On collision kill the snake and print the score
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1:
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((500, 500, 3), dtype='uint8')
            cv2.putText(self.img, 'Your Score is {}'.format(self.score), (140, 250), font, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow('a', self.img)

    # def close(self):
    #    pass


'''-------------------------------------------------------------------------------'''

class SnekEnv3(SnekEnv2):

    def __init__(self):
        super().__init__()
        self.SNAKE_LEN_GOAL = 45


        self.observation_space = spaces.Box(low=-500, high=500,  # maximum possible values
                                            # shape = observations length
                                            shape=(16 + self.SNAKE_LEN_GOAL,),
                                            dtype=np.float32)  # HEIGHT, WIDTH for N_CHANNELS

    def all_observations(self, reset):

        if self.apple_hit:
            apple_hit = 1
        else:
            apple_hit = 0

        self.apple_hit = False

        if reset:
            self.left = 1
        else:
            self.left = 0
        self.right = 0
        self.up = 0
        self.down = 0

        for pos in self.snake_position[1:]:
            x_pos = pos[0]
            y_pos = pos[1]

            # 500 >= or 0 <
            if self.snake_head[0] - x_pos == 10 and self.snake_head[1] - y_pos == 0:
                self.left = 1

            if self.snake_head[0] - x_pos == -10 and self.snake_head[1] - y_pos == 0:
                self.right = 1

            if self.snake_head[1] - y_pos == 10 and self.snake_head[0] - x_pos == 0:
                self.up = 1

            if self.snake_head[1] - y_pos == -10 and self.snake_head[0] - x_pos == 0:
                self.down = 1

        if self.snake_head[0] == 490:
            self.right = 1

        elif self.snake_head[0] == 0:
            self.left = 1

        if self.snake_head[1] == 490:
            self.down = 1

        elif self.snake_head[1] == 0:
            self.up = 1

        head_x = self.snake_head[0] / 20
        head_y = self.snake_head[1] / 20

        neck_x = self.snake_position[1][0] / 20
        neck_y = self.snake_position[1][1] / 20

        tail_x = self.snake_position[-1][0] / 20
        tail_y = self.snake_position[-1][1] / 20

        apple_x = self.apple_position[0] / 20
        apple_y = self.apple_position[1] / 20

        self.apple_delta_x = head_x - apple_x
        self.apple_delta_y = head_y - apple_y

        neck_delta_x = head_x - neck_x
        neck_delta_y = head_y - neck_y

        self.snake_len = len(self.snake_position)

        tail_delta_x = head_x - tail_x
        tail_delta_y = head_y - tail_y

        if reset:
            apple_hit = 0

            self.prev_actions = deque(maxlen=self.SNAKE_LEN_GOAL)

            for _ in range(self.SNAKE_LEN_GOAL):
                self.prev_actions.append(-1)  # create null history

        self.observation = [head_x, head_y, apple_x, apple_y, neck_delta_x,
                            neck_delta_y, self.apple_delta_x, self.apple_delta_y, apple_hit,
                            self.snake_len, tail_delta_x, tail_delta_y,
                            self.left, self.right, self.up, self.down] + list(self.prev_actions)

        # print(self.observation)

        self.observation = np.array(self.observation)

    def get_info(self):
        return {"length": self.snake_len, "cycle": self.cycle,
                "snake_x": self.snake_head[0], "snake_y": self.snake_head[1],
                "left": self.left, "right": self.right, "up": self.up, "down": self.down}


'''-------------------------------------------------------------------------------'''


class SnekEnv4(SnekEnv3):

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(low=-500, high=500,  # maximum possible values
                                            # shape = observations length
                                            shape=(20 + self.SNAKE_LEN_GOAL,),
                                            dtype=np.float32)  # HEIGHT, WIDTH for N_CHANNELS

    def all_observations(self, reset):

        if self.apple_hit:
            apple_hit = 1
        else:
            apple_hit = 0

        self.apple_hit = False

        self.left = 0
        self.right = 0
        self.up = 0
        self.down = 0

        self.apple_left = 0
        self.apple_right = 0
        self.apple_up = 0
        self.apple_down = 0

        for pos in self.snake_position[1:]:
            x_pos = pos[0]
            y_pos = pos[1]

            # death: 500 >= or 0 <
            if self.snake_head[0] - x_pos == 10 and self.snake_head[1] - y_pos == 0:
                self.left = 1

            if self.snake_head[0] - x_pos == -10 and self.snake_head[1] - y_pos == 0:
                self.right = 1

            if self.snake_head[1] - y_pos == 10 and self.snake_head[0] - x_pos == 0:
                self.up = 1

            if self.snake_head[1] - y_pos == -10 and self.snake_head[0] - x_pos == 0:
                self.down = 1

        if self.snake_head[0] == 490:
            self.right = 1
        elif self.snake_head[0] == 0:
            self.left = 1

        if self.snake_head[1] == 490:
            self.down = 1
        elif self.snake_head[1] == 0:
            self.up = 1

        if self.snake_head[0] - self.apple_position[0] == 10:
            self.apple_left = 1
        elif self.snake_head[0] - self.apple_position[0] == -10:
            self.apple_right = 1
        elif self.snake_head[1] - self.apple_position[1] == 10:
            self.apple_up = 1
        elif self.snake_head[1] - self.apple_position[1] == -10:
            self.apple_down = 1

        head_x = self.snake_head[0] / 20
        head_y = self.snake_head[1] / 20

        neck_x = self.snake_position[1][0] / 20
        neck_y = self.snake_position[1][1] / 20

        tail_x = self.snake_position[-1][0] / 20
        tail_y = self.snake_position[-1][1] / 20

        apple_x = self.apple_position[0] / 20
        apple_y = self.apple_position[1] / 20

        self.apple_delta_x = head_x - apple_x
        self.apple_delta_y = head_y - apple_y

        neck_delta_x = head_x - neck_x
        neck_delta_y = head_y - neck_y

        self.snake_len = len(self.snake_position)

        tail_delta_x = head_x - tail_x
        tail_delta_y = head_y - tail_y

        if reset:
            apple_hit = 0

            self.prev_actions = deque(maxlen=self.SNAKE_LEN_GOAL)

            for _ in range(self.SNAKE_LEN_GOAL):
                self.prev_actions.append(-1)  # create null history

        self.observation = [head_x, head_y, apple_x, apple_y, neck_delta_x,
                            neck_delta_y, self.apple_delta_x, self.apple_delta_y, apple_hit,
                            self.snake_len, tail_delta_x, tail_delta_y,
                            self.left, self.right, self.up, self.down,
                            self.apple_left, self.apple_right, self.apple_up, self.apple_down] + list(self.prev_actions)

        self.observation = np.array(self.observation)

    def all_rewards(self):

        self.cycle += 1
        if self.cycle > 10000:
            self.done = True

        euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

        self.distance = (euclidean_dist_to_apple / 100)

        # getting closer to the apple is rewarded, not just being close to the apple
        if self.previous_distance > 0.1:
            self.reward = self.previous_distance - self.distance
        else:
            self.reward = 0

        # print(f"{self.previous_distance} - {self.distance} = {self.reward}    D: {(euclidean_dist_to_apple/100)}")

        self.previous_distance = self.distance

        if self.reward < 0:
            self.reward *= 1.1

        if self.apple_hit:
            self.reward = 30 + self.score * 4 + 1.15 ** self.score
            self.previous_distance = 0
            # print(f"we hit the apple, {self.reward}")

        if self.done:
            self.reward -= (30 + self.score * 4 + 1.15 ** self.score)

            self.previous_distance = 0

    def get_info(self):
        return {"length": self.snake_len, "cycle": self.cycle,
                "snake_x": self.snake_head[0], "snake_y": self.snake_head[1],
                "left": self.left, "right": self.right, "up": self.up, "down": self.down}


'''-------------------------------------------------------------------------------'''


class SnekEnv5(SnekEnv4):

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(low=-25, high=200,  # maximum possible values
                                            # shape = observations length
                                            shape=(20 + self.SNAKE_LEN_GOAL,),
                                            dtype=np.float32)  # HEIGHT, WIDTH for N_CHANNELS

    def all_observations(self, reset):

        if self.apple_hit:
            apple_hit = 1
        else:
            apple_hit = 0

        self.apple_hit = False

        self.left = -1
        self.right = -1
        self.up = -1
        self.down = -1

        self.apple_left = 0
        self.apple_right = 0
        self.apple_up = 0
        self.apple_down = 0

        for pos in self.snake_position[1:]:
            x_pos = pos[0]
            y_pos = pos[1]

            # death: 500 >= or 0

            if self.snake_head[1] - y_pos == 0:
                if not self.left == 1:
                    if self.snake_head[0] - x_pos == 20:
                        self.left = 0
                    elif self.snake_head[0] - x_pos == 10:
                        self.left = 1

                if not self.right == 1:
                    if self.snake_head[0] - x_pos == -20:
                        self.right = 0
                    elif self.snake_head[0] - x_pos == -10:
                        self.right = 1

            if self.snake_head[0] - x_pos == 0:
                if not self.up == 1:
                    if self.snake_head[1] - y_pos == 20:
                        self.up = 0
                    elif self.snake_head[1] - y_pos == 10:
                        self.up = 1

                if not self.down == 1:
                    if self.snake_head[1] - y_pos == -20:
                        self.down = 0
                    elif self.snake_head[1] - y_pos == -10:
                        self.down = 1

        if not self.right == 1:
            if self.snake_head[0] == 480:
                self.right = 0
            elif self.snake_head[0] == 490:
                self.right = 1

        if not self.left == 1:
            if self.snake_head[0] == 10:
                self.left = 0
            elif self.snake_head[0] == 0:
                self.left = 1

        if not self.down == 1:
            if self.snake_head[1] == 480:
                self.down = 0
            elif self.snake_head[1] == 490:
                self.down = 1

        if not self.up == 1:
            if self.snake_head[1] == 10:
                self.up = 0
            elif self.snake_head[1] == 0:
                self.up = 1

        if self.snake_head[0] - self.apple_position[0] == 10:
            self.apple_left = 1

        elif self.snake_head[0] - self.apple_position[0] == -10:
            self.apple_right = 1

        elif self.snake_head[1] - self.apple_position[1] == 10:
            self.apple_up = 1

        elif self.snake_head[1] - self.apple_position[1] == -10:
            self.apple_down = 1

        head_x = self.snake_head[0] / 20
        head_y = self.snake_head[1] / 20

        neck_x = self.snake_position[1][0] / 20
        neck_y = self.snake_position[1][1] / 20

        tail_x = self.snake_position[-1][0] / 20
        tail_y = self.snake_position[-1][1] / 20

        apple_x = self.apple_position[0] / 20
        apple_y = self.apple_position[1] / 20

        self.apple_delta_x = (apple_x - head_x)
        self.apple_delta_y = (apple_y - head_y)

        if self.apple_delta_x < 0:
            self.apple_delta_x = -1
        elif self.apple_delta_x > 0:
            self.apple_delta_x = 1
        else:
            self.apple_delta_x = 0

        if self.apple_delta_y < 0:
            self.apple_delta_y = -1
        elif self.apple_delta_y > 0:
            self.apple_delta_y = 1
        else:
            self.apple_delta_y = 0

        neck_delta_x = head_x - neck_x
        neck_delta_y = head_y - neck_y

        self.snake_len = len(self.snake_position)

        tail_delta_x = head_x - tail_x
        tail_delta_y = head_y - tail_y

        if reset:
            apple_hit = 0

            self.prev_actions = deque(maxlen=self.SNAKE_LEN_GOAL)

            for _ in range(self.SNAKE_LEN_GOAL):
                self.prev_actions.append(-1)  # create null history

        self.observation = [head_x, head_y, apple_x, apple_y, neck_delta_x,
                            neck_delta_y, self.apple_delta_x, self.apple_delta_y, apple_hit,
                            self.snake_len, tail_delta_x, tail_delta_y,
                            self.left, self.right, self.up, self.down,
                            self.apple_left, self.apple_right, self.apple_up, self.apple_down] + list(self.prev_actions)

        self.observation = np.array(self.observation)

    def get_info(self):
        return {"length": self.snake_len, "cycle": self.cycle,
                "apple_delta_x": self.apple_delta_x, "apple_delta_y": self.apple_delta_y,
                "left": self.left, "right": self.right, "up": self.up, "down": self.down}


'''-------------------------------------------------------------------------------'''


class SnekEnv6(SnekEnv5):


    def __init__(self):
        super().__init__()
        self.SNAKE_LEN_GOAL = 50

        self.observation_space = spaces.Box(low=-25, high=200,  # maximum possible values
                                            # shape = observations length
                                            shape=(10 + self.SNAKE_LEN_GOAL,),
                                            dtype=np.float32)  # HEIGHT, WIDTH for N_CHANNELS

    def all_observations(self, reset):

        if self.apple_hit:
            apple_hit = 1
        else:
            apple_hit = 0

        self.apple_hit = False

        self.left = -1
        self.right = -1
        self.up = -1
        self.down = -1

        self.apple_left = 0
        self.apple_right = 0
        self.apple_up = 0
        self.apple_down = 0

        for pos in self.snake_position[1:]:
            x_pos = pos[0]
            y_pos = pos[1]

            # death: 500 >= or 0

            if self.snake_head[1] - y_pos == 0:
                if not self.left == 1:
                    if self.snake_head[0] - x_pos == 20:
                        self.left = 0
                    elif self.snake_head[0] - x_pos == 10:
                        self.left = 1

                if not self.right == 1:
                    if self.snake_head[0] - x_pos == -20:
                        self.right = 0
                    elif self.snake_head[0] - x_pos == -10:
                        self.right = 1

            if self.snake_head[0] - x_pos == 0:
                if not self.up == 1:
                    if self.snake_head[1] - y_pos == 20:
                        self.up = 0
                    elif self.snake_head[1] - y_pos == 10:
                        self.up = 1

                if not self.down == 1:
                    if self.snake_head[1] - y_pos == -20:
                        self.down = 0
                    elif self.snake_head[1] - y_pos == -10:
                        self.down = 1

        if not self.right == 1:
            if self.snake_head[0] == 480:
                self.right = 0
            elif self.snake_head[0] == 490:
                self.right = 1

        if not self.left == 1:
            if self.snake_head[0] == 10:
                self.left = 0
            elif self.snake_head[0] == 0:
                self.left = 1

        if not self.down == 1:
            if self.snake_head[1] == 480:
                self.down = 0
            elif self.snake_head[1] == 490:
                self.down = 1

        if not self.up == 1:
            if self.snake_head[1] == 10:
                self.up = 0
            elif self.snake_head[1] == 0:
                self.up = 1

        if self.snake_head[0] - self.apple_position[0] == 10:
            self.apple_left = 1
        elif self.snake_head[0] - self.apple_position[0] == -10:
            self.apple_right = 1
        elif self.snake_head[1] - self.apple_position[1] == 10:
            self.apple_up = 1
        elif self.snake_head[1] - self.apple_position[1] == -10:
            self.apple_down = 1

        head_x = self.snake_head[0] / 20
        head_y = self.snake_head[1] / 20

        neck_x = self.snake_position[1][0] / 20
        neck_y = self.snake_position[1][1] / 20

        tail_x = self.snake_position[-1][0] / 20
        tail_y = self.snake_position[-1][1] / 20

        apple_x = self.apple_position[0] / 20
        apple_y = self.apple_position[1] / 20

        self.apple_delta_x = (apple_x - head_x)
        self.apple_delta_y = (apple_y - head_y)

        if self.apple_delta_x < 0:
            self.apple_delta_x = -1
        elif self.apple_delta_x > 0:
            self.apple_delta_x = 1
        else:
            self.apple_delta_x = 0

        if self.apple_delta_y < 0:
            self.apple_delta_y = -1
        elif self.apple_delta_y > 0:
            self.apple_delta_y = 1
        else:
            self.apple_delta_y = 0

        # neck_delta_x = head_x - neck_x
        # neck_delta_y = head_y - neck_y

        # if neck_delta_x < 0:
        #    neck_delta_x = -1
        # elif neck_delta_x > 0:
        #    neck_delta_x = 1
        # else:
        #    neck_delta_x = 0

        # if neck_delta_y < 0:
        #    neck_delta_y = -1
        # elif neck_delta_y > 0:
        #    neck_delta_y = 1
        # else:
        #    neck_delta_y = 0

        self.snake_len = len(self.snake_position) / 10

        tail_delta_x = head_x - tail_x
        tail_delta_y = head_y - tail_y

        if reset:
            apple_hit = 0

            self.prev_actions = deque(maxlen=self.SNAKE_LEN_GOAL)

            for _ in range(self.SNAKE_LEN_GOAL):
                self.prev_actions.append(-1)  # create null history

        self.observation = [self.apple_delta_x, self.apple_delta_y, apple_hit,
                            self.snake_len, tail_delta_x, tail_delta_y,
                            self.left, self.right, self.up, self.down] + list(self.prev_actions)

        self.observation = np.array(self.observation)


'''-------------------------------------------------------------------------------'''

class SnekEnv7(SnekEnv6):

    def __init__(self):
        super().__init__()
        self.SNAKE_LEN_GOAL = 60
        self.observation_space = spaces.Box(low=-25, high=200,  # maximum possible values
                                            # shape = observations length
                                            shape=(10 + self.SNAKE_LEN_GOAL,),
                                            dtype=np.float32)  # HEIGHT, WIDTH for N_CHANNELS

    def all_rewards(self):

        self.cycle += 1
        if self.cycle > 10000:
            self.done = True

        euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

        self.distance = (euclidean_dist_to_apple / 50)

        # getting closer to the apple is rewarded, not just being close to the apple
        if self.previous_distance > 0.1:
            self.reward = self.previous_distance - self.distance
        else:
            self.reward = 0

        # print(f"{self.previous_distance} - {self.distance} = {self.reward}    D: {(euclidean_dist_to_apple/100)}")

        self.previous_distance = self.distance

        if self.reward < 0:
            self.reward *= 1.1

        if self.apple_hit:
            self.reward = 30 + self.score * 4 + 1.1 ** self.score
            self.previous_distance = 0
            # print(f"we hit the apple, {self.reward}")

        if self.done:
            self.reward -= (30 + self.score * 4 + 1.1 ** self.score)

            self.previous_distance = 0


'''-------------------------------------------------------------------------------'''


class SnekEnv8(SnekEnv7):

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(low=-25, high=200,  # maximum possible values
                                            # shape = observations length
                                            shape=(8,), dtype=np.float32)  # HEIGHT, WIDTH for N_CHANNELS

    def all_observations(self, reset):

        if self.apple_hit:
            apple_hit = 1
        else:
            apple_hit = 0

        self.apple_hit = False

        self.left = -1
        self.right = -1
        self.up = -1
        self.down = -1

        self.apple_left = 0
        self.apple_right = 0
        self.apple_up = 0
        self.apple_down = 0

        for pos in self.snake_position[1:]:
            x_pos = pos[0]
            y_pos = pos[1]

            # death: 500 >= or 0

            if self.snake_head[1] - y_pos == 0:
                if not self.left == 1:
                    if self.snake_head[0] - x_pos == 20:
                        self.left = 0
                    elif self.snake_head[0] - x_pos == 10:
                        self.left = 1

                if not self.right == 1:
                    if self.snake_head[0] - x_pos == -20:
                        self.right = 0
                    elif self.snake_head[0] - x_pos == -10:
                        self.right = 1

            if self.snake_head[0] - x_pos == 0:
                if not self.up == 1:
                    if self.snake_head[1] - y_pos == 20:
                        self.up = 0
                    elif self.snake_head[1] - y_pos == 10:
                        self.up = 1

                if not self.down == 1:
                    if self.snake_head[1] - y_pos == -20:
                        self.down = 0
                    elif self.snake_head[1] - y_pos == -10:
                        self.down = 1

        if not self.right == 1:
            if self.snake_head[0] == 480:
                self.right = 0
            elif self.snake_head[0] == 490:
                self.right = 1

        if not self.left == 1:
            if self.snake_head[0] == 10:
                self.left = 0
            elif self.snake_head[0] == 0:
                self.left = 1

        if not self.down == 1:
            if self.snake_head[1] == 480:
                self.down = 0
            elif self.snake_head[1] == 490:
                self.down = 1

        if not self.up == 1:
            if self.snake_head[1] == 10:
                self.up = 0
            elif self.snake_head[1] == 0:
                self.up = 1

        if self.snake_head[0] - self.apple_position[0] == 10:
            self.apple_left = 1
        elif self.snake_head[0] - self.apple_position[0] == -10:
            self.apple_right = 1
        elif self.snake_head[1] - self.apple_position[1] == 10:
            self.apple_up = 1
        elif self.snake_head[1] - self.apple_position[1] == -10:
            self.apple_down = 1

        head_x = self.snake_head[0] / 20
        head_y = self.snake_head[1] / 20

        neck_x = self.snake_position[1][0] / 20
        neck_y = self.snake_position[1][1] / 20

        tail_x = self.snake_position[-1][0] / 20
        tail_y = self.snake_position[-1][1] / 20

        apple_x = self.apple_position[0] / 20
        apple_y = self.apple_position[1] / 20

        self.apple_delta_x = (apple_x - head_x)
        self.apple_delta_y = (apple_y - head_y)

        if self.apple_delta_x < 0:
            self.apple_delta_x = -1
        elif self.apple_delta_x > 0:
            self.apple_delta_x = 1
        else:
            self.apple_delta_x = 0

        if self.apple_delta_y < 0:
            self.apple_delta_y = -1
        elif self.apple_delta_y > 0:
            self.apple_delta_y = 1
        else:
            self.apple_delta_y = 0

        # neck_delta_x = head_x - neck_x
        # neck_delta_y = head_y - neck_y

        # if neck_delta_x < 0:
        #    neck_delta_x = -1
        # elif neck_delta_x > 0:
        #    neck_delta_x = 1
        # else:
        #    neck_delta_x = 0

        # if neck_delta_y < 0:
        #    neck_delta_y = -1
        # elif neck_delta_y > 0:
        #    neck_delta_y = 1
        # else:
        #    neck_delta_y = 0

        self.snake_len = len(self.snake_position)

        tail_delta_x = head_x - tail_x
        tail_delta_y = head_y - tail_y

        if reset:
            apple_hit = 0

            self.prev_actions = deque(maxlen=self.SNAKE_LEN_GOAL)

            for _ in range(self.SNAKE_LEN_GOAL):
                self.prev_actions.append(-1)  # create null history

        self.observation = [self.apple_delta_x, self.apple_delta_y,
                            tail_delta_x, tail_delta_y,
                            self.left, self.right, self.up, self.down]

        self.observation = np.array(self.observation)


'''-------------------------------------------------------------------------------'''


class SnekEnv9(SnekEnv8):

    def __init__(self):
        super().__init__()

    def all_rewards(self):

        self.cycle += 1
        if self.cycle > 10000:
            self.done = True

        euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

        self.distance = (euclidean_dist_to_apple / 25)

        # getting closer to the apple is rewarded, not just being close to the apple
        if self.previous_distance > 0.1:
            self.reward = self.previous_distance - self.distance
        else:
            self.reward = 0

        # print(f"{self.previous_distance} - {self.distance} = {self.reward}    D: {(euclidean_dist_to_apple/100)}")

        self.previous_distance = self.distance

        if self.reward < 0:
            self.reward *= 1.1

        if self.apple_hit:
            self.reward = 30
            self.previous_distance = 0
            # print(f"we hit the apple, {self.reward}")

        if self.done:
            self.reward -= 30

            self.previous_distance = 0


class SnekEnv10(SnekEnv9):

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(low=-25, high=200,  # maximum possible values
                                            # shape = observations length
                                            shape=(10,), dtype=np.float32)  # HEIGHT, WIDTH for N_CHANNELS

    def lock_to_value(self, value):
        if value < 0:
            return_int = -1
        elif value > 0:
            return_int = 1
        else:
            return_int = 0

        return return_int

    def all_observations(self, reset):
        self.apple_hit = False

        self.left = -1
        self.right = -1
        self.up = -1
        self.down = -1

        self.apple_left = 0
        self.apple_right = 0
        self.apple_up = 0
        self.apple_down = 0

        for pos in self.snake_position[1:]:
            x_pos = pos[0]
            y_pos = pos[1]

            # death: 500 >= or 0

            if self.snake_head[1] - y_pos == 0:
                if not self.left == 1:
                    if self.snake_head[0] - x_pos == 20:
                        self.left = 0
                    elif self.snake_head[0] - x_pos == 10:
                        self.left = 1

                if not self.right == 1:
                    if self.snake_head[0] - x_pos == -20:
                        self.right = 0
                    elif self.snake_head[0] - x_pos == -10:
                        self.right = 1

            if self.snake_head[0] - x_pos == 0:
                if not self.up == 1:
                    if self.snake_head[1] - y_pos == 20:
                        self.up = 0
                    elif self.snake_head[1] - y_pos == 10:
                        self.up = 1

                if not self.down == 1:
                    if self.snake_head[1] - y_pos == -20:
                        self.down = 0
                    elif self.snake_head[1] - y_pos == -10:
                        self.down = 1

        if not self.right == 1:
            if self.snake_head[0] == 480:
                self.right = 0
            elif self.snake_head[0] == 490:
                self.right = 1

        if not self.left == 1:
            if self.snake_head[0] == 10:
                self.left = 0
            elif self.snake_head[0] == 0:
                self.left = 1

        if not self.down == 1:
            if self.snake_head[1] == 480:
                self.down = 0
            elif self.snake_head[1] == 490:
                self.down = 1

        if not self.up == 1:
            if self.snake_head[1] == 10:
                self.up = 0
            elif self.snake_head[1] == 0:
                self.up = 1

        if self.snake_head[0] - self.apple_position[0] == 10:
            self.apple_left = 1
        elif self.snake_head[0] - self.apple_position[0] == -10:
            self.apple_right = 1
        elif self.snake_head[1] - self.apple_position[1] == 10:
            self.apple_up = 1
        elif self.snake_head[1] - self.apple_position[1] == -10:
            self.apple_down = 1

        head_x = self.snake_head[0] / 20
        head_y = self.snake_head[1] / 20

        tail_x = self.snake_position[-1][0] / 20
        tail_y = self.snake_position[-1][1] / 20

        apple_x = self.apple_position[0] / 20
        apple_y = self.apple_position[1] / 20

        self.apple_delta_x = self.lock_to_value(apple_x - head_x)
        self.apple_delta_y = self.lock_to_value(apple_y - head_y)

        self.snake_len = len(self.snake_position)

        middle_index = round((len(self.snake_position) - 1) / 2)

        middle_x = self.snake_position[middle_index][0] / 20
        middle_y = self.snake_position[middle_index][1] / 20

        self.delta_middle_x = self.lock_to_value(head_x - middle_x)
        self.delta_middle_y = self.lock_to_value(head_y - middle_y)

        tail_delta_x = self.lock_to_value(head_x - tail_x)
        tail_delta_y = self.lock_to_value(head_y - tail_y)

        if reset:
            apple_hit = 0

            self.prev_actions = deque(maxlen=self.SNAKE_LEN_GOAL)

            for _ in range(self.SNAKE_LEN_GOAL):
                self.prev_actions.append(-1)  # create null history

        self.observation = [self.apple_delta_x, self.apple_delta_y,
                            tail_delta_x, tail_delta_y, self.delta_middle_x, self.delta_middle_y,
                            self.left, self.right, self.up, self.down]

        self.observation = np.array(self.observation)

    def get_info(self):
        return {"cycle": self.cycle, "length": self.snake_len, "delta_middle_x": self.delta_middle_x,
                "delta_middle_y": self.delta_middle_y,
                "apple_delta_x": self.apple_delta_x, "apple_delta_y": self.apple_delta_y,
                "left": self.left, "right": self.right, "up": self.up, "down": self.down,
                "action": list(self.prev_actions)[-1]}


class SnekEnv11(SnekEnv10):

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(low=-25, high=200,  # maximum possible values
                                            # shape = observations length
                                            shape=(10,), dtype=np.int8)  # HEIGHT, WIDTH for N_CHANNELS

    def all_observations(self, reset):
        self.apple_hit = False

        self.left = 4
        self.right = 4
        self.up = 4
        self.down = 4

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        try:
            self.left = -2 + (min([head_x - pos[0] for pos in self.snake_position[1:] if
                                   head_x > pos[0] and head_y - pos[1] == 0]) / 10)
        except:
            pass
        try:
            self.right = -2 + (min([pos[0] - head_x for pos in self.snake_position[1:] if
                                    pos[0] > head_x and head_y - pos[1] == 0]) / 10)
        except:
            pass
        try:
            self.up = -2 + (min([head_y - pos[1] for pos in self.snake_position[1:] if
                                 head_y > pos[1] and head_x - pos[0] == 0]) / 10)
        except:
            pass
        try:
            self.down = -2 + (min([pos[1] - head_y for pos in self.snake_position[1:] if
                                   pos[1] > head_y and head_x - pos[0] == 0]) / 10)
        except:
            pass

        wall_left = -2 + ((head_x + 10) / 10)
        wall_right = -2 + ((500 - head_x) / 10)
        wall_up = -2 + ((head_y + 10) / 10)
        wall_down = -2 + ((500 - head_y) / 10)

        if wall_left < self.left:
            self.left = wall_left

        if wall_right < self.right:
            self.right = wall_right

        if wall_up < self.up:
            self.up = wall_up

        if wall_down < self.down:
            self.down = wall_down

        if self.left > 4:
            self.left = 4
        if self.right > 4:
            self.right = 4
        if self.up > 4:
            self.up = 4
        if self.down > 4:
            self.down = 4

        tail_x = self.snake_position[-1][0]
        tail_y = self.snake_position[-1][1]

        apple_x = self.apple_position[0]
        apple_y = self.apple_position[1]
        self.apple_delta_x = self.lock_to_value(apple_x - head_x)
        self.apple_delta_y = self.lock_to_value(apple_y - head_y)

        self.snake_len = len(self.snake_position)

        middle_index = round((len(self.snake_position) - 1) / 2)
        middle_x = self.snake_position[middle_index][0]
        middle_y = self.snake_position[middle_index][1]
        self.delta_middle_x = self.lock_to_value(head_x - middle_x)
        self.delta_middle_y = self.lock_to_value(head_y - middle_y)

        tail_delta_x = self.lock_to_value(head_x - tail_x)
        tail_delta_y = self.lock_to_value(head_y - tail_y)

        if reset:
            apple_hit = 0

            self.prev_actions = deque(maxlen=self.SNAKE_LEN_GOAL)

            for _ in range(self.SNAKE_LEN_GOAL):
                self.prev_actions.append(-1)  # create null history

        self.observation = [self.apple_delta_x, self.apple_delta_y,
                            tail_delta_x, tail_delta_y, self.delta_middle_x, self.delta_middle_y,
                            self.left, self.right, self.up, self.down]

        self.observation = [int(x) for x in self.observation]

        self.observation = np.array(self.observation)


# uusin env
class SnekEnv12(SnekEnv11):

    def __init__(self):
        super().__init__()

    def all_rewards(self):

        self.cycle += 1
        if self.cycle > 10000:
            self.done = True

        euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

        self.distance = (euclidean_dist_to_apple / 25)

        # getting closer to the apple is rewarded, not just being close to the apple
        if self.previous_distance > 0.1:
            self.reward = self.previous_distance - self.distance
        else:
            self.reward = 0

        # print(f"{self.previous_distance} - {self.distance} = {self.reward}    D: {(euclidean_dist_to_apple/100)}")

        self.previous_distance = self.distance

        self.reward += (self.left + self.right + self.up + self.down - 9) / 2

        if self.reward < 0:
            self.reward *= 1.1

        if self.apple_hit:
            self.reward = 40
            self.previous_distance = 0
            # print(f"we hit the apple, {self.reward}")

        if self.done:
            self.reward -= 40

            self.previous_distance = 0


# starting point:
'''
import gym
from gym import spaces


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, arg1, arg2, ...):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        observation_space = spaces.Box(low=0, high=255,
                                            shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

    def step(self, action):
        ...
        return observation, reward, done, info

    def reset(self):
        ...
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        ...

    def close(self):
        ...


# check env
from stable_baselines3.common.env_checker import check_env
from customenv import CustomEnv


env = CustomEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)
'''
