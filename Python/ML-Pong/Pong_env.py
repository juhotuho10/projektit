import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2
import math
import random

black = (0,0,0)
white = (255,255,255)
gray = (200,200,200)

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)


class Pong_env(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255, shape=(6,), dtype=np.float16)

        self.width = 600
        self.height = 400

        self.paddle_width = 10

        self.paddle1_height = 300
        self.paddle2_height = 60

        self.paddle_x = 30

        self.paddle1_suface = self.paddle_x + self.paddle_width
        self.paddle2_suface = self.width - self.paddle_x 

        self.ball_size = 10
        self.ball_radius = self.ball_size / 2

        self.paddle_acceleration = 4

        self.reset_game()


    def reset_game(self):

        self.reward = 0

        self.img = np.zeros((self.width, self.height, 3), dtype='uint8') + 20

        self.ball_speed_x = random.choice([-5,5]) 
        self.ball_speed_y = random.uniform(-5, 5)

        self.paddle1_y = self.height / 5
        self.paddle2_y = self.height / 2

        self.paddle1_speed = 0
        self.paddle2_speed = 0

        self.ball_x = self.width / 2
        self.ball_y = self.height / 2
        
        # ball outer surface spin speed
        # positive means clockwise spin
        #self.ball_spin = random.uniform(-5, 5)

        self.ball_spin = random.choice([-10,10])
        self.ball_spin_angle = 0
        
    def take_action(self, action):
        
        if action == 0: 
            # move up
            self.paddle2_speed += self.paddle_acceleration
        elif action == 1:
            # possibility to not move at all
            pass
        elif action == 2:  
            # move down
            self.paddle2_speed -= self.paddle_acceleration

    def get_observation(self):

        abs_y = self.paddle2_y / self.height

        relative_y = (self.paddle2_y - self.ball_y) / self.height

        relative_ball_x = self.ball_x / (self.width - self.paddle_x)

        ball_direction = self.ball_speed_x > 0

        observation = [abs_y, relative_y, self.paddle2_speed, relative_ball_x, ball_direction, self.ball_speed_y]

        return np.array(observation)

    def get_done(self):

        done = False

        reward = 0
        if self.ball_x <= 0: 
            reward = 1
            done = True
        elif self.ball_x >= self.width - self.ball_size:
            reward = -1
            done = True
        
        return reward, done
    
    def get_paddle_collision(self):

        if self.paddle1_y <= 0:
            self.paddle1_y = 0
            self.paddle1_speed = -self.paddle1_speed * 0.5

        elif self.paddle1_y + self.paddle1_height >= self.height:
            self.paddle1_y = self.height - self.paddle1_height
            self.paddle1_speed = -self.paddle1_speed * 0.5


        if self.paddle2_y <= 0:
            self.paddle2_y = 0
            self.paddle2_speed = -self.paddle2_speed * 0.5

        elif self.paddle2_y + self.paddle2_height >= self.height:
            self.paddle2_y = self.height - self.paddle2_height
            self.paddle2_speed = -self.paddle2_speed * 0.5

    def random_momentum_transfer(self):

        if abs(self.ball_speed_x) > 0.5:
            return
    
        # Transfer some of the movement to spin
        spin_transfer_ratio = random.uniform(0.2, 0.6)
        y_transfer_ratio = random.uniform(0.2, 0.6)

        # np sign turns into poositive or negative 1
        self.ball_speed_x + np.sign(self.ball_speed_x) * (abs(self.ball_spin * spin_transfer_ratio) + abs(self.ball_speed_y * y_transfer_ratio))


        self.ball_spin *= (1 - spin_transfer_ratio)
        self.ball_speed_y *= (1 - y_transfer_ratio)


    def get_ball_collisions(self):
            
        reward = 0

        momentum_multiplier = 0.3

        ball_up_surface = self.ball_y - self.ball_radius
        ball_left_surface = self.ball_x - self.ball_radius

        ball_down_surface = self.ball_y + self.ball_radius
        ball_right_surface = self.ball_x + self.ball_radius


        # ball collision with top and bottom walls
        if ball_up_surface <= 0:
            self.ball_speed_y = -self.ball_speed_y
            # adjust ball position to be within bounds
            self.ball_y = self.ball_radius

            # ball spin is transferred to x momentum 
            # and ball momentum is transferred to speed
            speed_induced_spin = self.ball_speed_x * momentum_multiplier * 1.05
            spin_induced_speed = self.ball_spin * momentum_multiplier * 1.05

            self.ball_speed_x *= 1 - momentum_multiplier
            self.ball_spin  *= 1 - momentum_multiplier

            self.ball_spin -= speed_induced_spin

            self.ball_speed_x -= spin_induced_speed
            self.ball_speed_y += spin_induced_speed

            self.random_momentum_transfer()    

        elif ball_down_surface >= self.height:
            self.ball_speed_y = -self.ball_speed_y
            # adjust ball position to be within bounds
            self.ball_y = self.height - self.ball_radius
    
            # ball spin is transferred to x momentum 
            # and ball momentum is transferred to speed
            speed_induced_spin = -self.ball_speed_x * momentum_multiplier * 1.05
            spin_induced_speed = -self.ball_spin * momentum_multiplier * 1.05

            self.ball_speed_x *= 1 - momentum_multiplier
            self.ball_spin *= 1 - momentum_multiplier

            self.ball_spin -= speed_induced_spin

            self.ball_speed_x -= spin_induced_speed
            self.ball_speed_y += spin_induced_speed

            self.random_momentum_transfer()   


        # Ball collision with left paddle
        if ball_left_surface <= self.paddle1_suface and \
            ball_down_surface >= self.paddle1_y and \
            ball_up_surface <= self.paddle1_y + self.paddle1_height:
            
            self.ball_speed_x = -self.ball_speed_x

            # adjust ball position to prevent sticking
            self.ball_x = self.paddle1_suface + self.ball_radius

            # getting the spin speed difference and applying that the the Y momentum
            speed_difference = self.paddle1_speed - self.ball_speed_y

            # ball spin is transferred to x momentum 
            # and ball momentum is transferred to speed
            speed_induced_spin = -speed_difference * momentum_multiplier * 1.05
            spin_induced_speed = -self.ball_spin * momentum_multiplier * 1.05

            self.ball_speed_y *= 1 - momentum_multiplier
            self.ball_spin *= 1 - momentum_multiplier

            self.ball_spin -= speed_induced_spin

            self.ball_speed_y -= spin_induced_speed
            self.ball_speed_x += spin_induced_speed


        # Ball collision with right paddle
        if ball_right_surface >= self.paddle2_suface and \
            ball_down_surface >= self.paddle2_y and \
            ball_up_surface <= self.paddle2_y + self.paddle2_height:

            self.ball_speed_x = -self.ball_speed_x

            # adjust ball position to prevent sticking
            self.ball_x = self.paddle2_suface - self.ball_radius  

            # getting the spin speed difference and applying that the the Y momentum
            speed_difference = self.paddle2_speed - self.ball_speed_y

            # ball spin is transferred to x momentum 
            # and ball momentum is transferred to speed
            speed_induced_spin = speed_difference * momentum_multiplier * 1.05
            spin_induced_speed = self.ball_spin * momentum_multiplier * 1.05

            self.ball_speed_y *= 1 - momentum_multiplier
            self.ball_spin *= 1 - momentum_multiplier

            self.ball_spin -= speed_induced_spin

            self.ball_speed_y -= spin_induced_speed
            self.ball_speed_x += spin_induced_speed

            reward = 0.05

        return reward
  
                
    def step(self, action):

        # movement = lower reward
        # this is to descourage rapid movement
        if action != 1:
            self.reward -= 0.001

        self.paddle1_speed *= 0.99
        self.paddle2_speed *= 0.99

        self.take_action(action)

        self.paddle1_y += self.paddle1_speed
        self.paddle2_y += self.paddle2_speed

        self.get_paddle_collision()

        self.ball_x += self.ball_speed_x
        self.ball_y += self.ball_speed_y

        hit_reward = self.get_ball_collisions()

        self.reward += hit_reward

        win_reward, done = self.get_done()

        self.reward += win_reward

        observation = self.get_observation()

        if done:
            self.reset_game()

        truncated = False

        #info = {"action": action, "reward": self.reward}

        info = {}

        return observation, self.reward, done, truncated, info

    def reset(self, seed=None, options=None):

        self.reset_game()

        observation = self.get_observation()

        info = {}
        return observation, info

    def render(self):
        # initialize the image array with black background
        self.img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # left paddle
        cv2.rectangle(self.img, (self.paddle_x, int(self.paddle1_y)), (self.paddle_x + self.paddle_width, int(self.paddle1_y + self.paddle1_height)), gray, -1)

        # right paddle
        cv2.rectangle(self.img, (self.width - self.paddle_x - self.paddle_width, int(self.paddle2_y)), (self.width - self.paddle_x, int(self.paddle2_y + self.paddle2_height)), gray, -1)

        # ball
        cv2.circle(self.img, (int(self.ball_x), int(self.ball_y)), self.ball_size // 2, gray, -1)

        self.ball_spin_angle += self.ball_spin / 10

        # red line across the ball to indicate spin
        line_length = self.ball_radius - 1
        
        start_x = int(self.ball_x + line_length * math.cos(self.ball_spin_angle))
        start_y = int(self.ball_y + line_length * math.sin(self.ball_spin_angle))
        end_x = int(self.ball_x - line_length * math.cos(self.ball_spin_angle))
        end_y = int(self.ball_y - line_length * math.sin(self.ball_spin_angle))


        cv2.line(self.img, (start_x, start_y), (end_x, end_y), red, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'X: {self.ball_speed_x:.2f}\nY: {self.ball_speed_y:.2f}\nSpin: {self.ball_spin:.2f}'
        y0, dy = 30, 15
        for i, line in enumerate(text.split('\n')):
            y = y0 + i*dy
            cv2.putText(self.img, line, (10, y), font, 0.5, gray, 1, cv2.LINE_AA)

        cv2.imshow("Pong", self.img)
        cv2.waitKey(1)