import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2
import math
import random
import keyboard
import time

black = (0,0,0)
white = (255,255,255)
gray = (200,200,200)

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)


class Pong_play_env(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(low=0, high=255, shape=(6,), dtype=np.float16)

        # window sizes
        self.width = 600
        self.height = 400

        self.paddle_width = 4

        # paddle X from the wall
        self.paddle_x = 30

        # precalculated surface X for both paddles
        self.paddle1_suface = self.paddle_x + self.paddle_width
        self.paddle2_suface = self.width - self.paddle_x 

        self.ball_size = 10
        self.ball_radius = self.ball_size / 2

        self.paddle_acceleration1 = 1
        self.paddle_acceleration2 = 1

        self.reset_game()


    def reset_game(self):

        self.reward = 0

        self.paddle1_height = 80
        self.paddle2_height = 80

        self.img = np.zeros((self.width, self.height, 3), dtype='uint8') + 20

        self.ball_speed_x = random.choice([-5,5]) 
        self.ball_speed_y = random.uniform(-10, 10)

        self.paddle1_y = self.height / 2
        self.paddle2_y = self.height / 2

        self.paddle1_speed = 0
        self.paddle2_speed = 0

        self.ball_x = self.width / 2
        self.ball_y = self.height / 2
        
        # ball outer surface spin speed
        # positive means clockwise spin
        self.ball_spin = random.uniform(-10, 10)
        self.ball_spin_angle = 0


    def key_listener(self):
        while True:
            key = cv2.waitKey(1)
            if key == ord('w'):
                self.key_pressed = 'w'
            elif key == ord('s'):
                self.key_pressed = 's'
            else:
                self.key_pressed = None
        
    def take_action(self, action):
        
        if action == 0: 
            # move up
            self.paddle2_speed += self.paddle_acceleration2
        elif action == 1:
            # possibility to not move at all
            pass
        elif action == 2:  
            # move down
            self.paddle2_speed -= self.paddle_acceleration2

    def get_observation(self):

        paddle2_center = self.paddle2_y + self.paddle2_height / 2

        observation = [
            (paddle2_center - self.ball_y) / self.height,
            paddle2_center / self.height,  
            self.ball_x / self.width,      
            self.ball_y / self.height,     
            self.ball_speed_x / 15,        
            self.ball_speed_y / 15         
        ]

        return np.array(observation)

    def get_done(self):

        done = False
        Won = False
        Lost = False

        if self.ball_x <= 0: 
            Won = True
            done = True

        elif self.ball_x >= self.width:
            Lost = True
            done = True
        
        return Won, Lost, done
    
    def get_paddle_collision(self):

        # left paddle collides with upper wall
        if self.paddle1_y <= 0:
            self.paddle1_y = 0
            self.paddle1_speed = -self.paddle1_speed * 0.5

        # left paddle collides with lower wall
        elif self.paddle1_y + self.paddle1_height >= self.height:
            self.paddle1_y = self.height - self.paddle1_height
            self.paddle1_speed = -self.paddle1_speed * 0.5

        # right paddle collides with upper wall
        if self.paddle2_y <= 0:
            self.paddle2_y = 0
            self.paddle2_speed = -self.paddle2_speed * 0.5

        # right paddle collides with lower wall
        elif self.paddle2_y + self.paddle2_height >= self.height:
            self.paddle2_y = self.height - self.paddle2_height
            self.paddle2_speed = -self.paddle2_speed * 0.5

    def random_momentum_transfer(self):

        # this function is there to prevent the ball from getting stuck bouncing up and down

        if abs(self.ball_speed_x) > 1:
            return
    
        # Transfer some of the movement to spin
        spin_transfer_ratio = random.uniform(0.1, 0.3)
        y_transfer_ratio = random.uniform(0.1, 0.3)

        # np sign turns into poositive or negative 1
        self.ball_speed_x += np.sign(self.ball_speed_x) * (abs(self.ball_spin * spin_transfer_ratio) + abs(self.ball_speed_y * y_transfer_ratio))


    def add_momentum(self, amount = 1.15):
        # adds momentum so the ball consistenly gets faster
        #self.ball_spin *= amount
        self.ball_speed_x *= amount
        #self.ball_speed_y *= amount

        self.ball_spin = np.clip(self.ball_spin, -15, 15)
        self.ball_speed_x = np.clip(self.ball_speed_x, -15, 15)
        self.ball_speed_y = np.clip(self.ball_speed_y, -15, 15)


    def get_ball_collisions(self):
            
        ball_hit = False

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

            self.add_momentum()

            # ball spin is transferred to x momentum 
            # and ball momentum is transferred to speed
            speed_induced_spin = self.ball_speed_x * momentum_multiplier
            spin_induced_speed = self.ball_spin * momentum_multiplier

            self.ball_speed_x *= 1 - momentum_multiplier
            self.ball_spin *= 1 - momentum_multiplier

            self.ball_spin -= speed_induced_spin

            self.ball_speed_x -= spin_induced_speed
            self.ball_speed_y += spin_induced_speed


            self.random_momentum_transfer()    

        elif ball_down_surface >= self.height:
            self.ball_speed_y = -self.ball_speed_y
            # adjust ball position to be within bounds
            self.ball_y = self.height - self.ball_radius

            self.add_momentum()
    
            # ball spin is transferred to x momentum 
            # and ball momentum is transferred to speed
            speed_induced_spin = -self.ball_speed_x * momentum_multiplier
            spin_induced_speed = -self.ball_spin * momentum_multiplier

            self.ball_speed_x *= 1 - momentum_multiplier
            self.ball_spin *= 1 - momentum_multiplier

            self.ball_spin -= speed_induced_spin

            self.ball_speed_x -= spin_induced_speed
            self.ball_speed_y += spin_induced_speed

            self.random_momentum_transfer()   


        # Ball collision with left paddle
        if ball_left_surface <= self.paddle1_suface and \
            ball_right_surface >= self.paddle1_suface and \
            self.ball_y >= self.paddle1_y and \
            self.ball_y <= self.paddle1_y + self.paddle1_height:
            
            self.ball_speed_x = -self.ball_speed_x

            # adjust ball position to prevent sticking
            self.ball_x = self.paddle1_suface + self.ball_radius

            self.add_momentum()

            # getting the spin speed difference and applying that the the Y momentum
            speed_difference = self.paddle1_speed - self.ball_speed_y

            # ball spin is transferred to x momentum 
            # and ball momentum is transferred to speed
            speed_induced_spin = -speed_difference * momentum_multiplier
            spin_induced_speed = -self.ball_spin * momentum_multiplier

            self.ball_speed_y *= 1 - momentum_multiplier
            self.ball_spin *= 1 - momentum_multiplier

            self.ball_spin = speed_induced_spin

            self.ball_speed_y -= spin_induced_speed
            self.ball_speed_x += spin_induced_speed


        # Ball collision with right paddle
        if ball_right_surface >= self.paddle2_suface and \
            ball_left_surface <= self.paddle2_suface and \
            self.ball_y >= self.paddle2_y and \
            self.ball_y <= self.paddle2_y + self.paddle2_height:

            self.ball_speed_x = -self.ball_speed_x

            # adjust ball position to prevent sticking
            self.ball_x = self.paddle2_suface - self.ball_radius  

            self.add_momentum()

            # getting the spin speed difference and applying that the the Y momentum
            speed_difference = self.paddle2_speed - self.ball_speed_y

            # ball spin is transferred to x momentum 
            # and ball momentum is transferred to speed
            speed_induced_spin = speed_difference * momentum_multiplier
            spin_induced_speed = self.ball_spin * momentum_multiplier

            self.ball_speed_y *= 1 - momentum_multiplier
            self.ball_spin *= 1 - momentum_multiplier

            self.ball_spin = speed_induced_spin

            self.ball_speed_y -= spin_induced_speed
            self.ball_speed_x += spin_induced_speed

            ball_hit = True

        return ball_hit
    
    def ball_spin_effect(self):
        # ball curves to the direction of the spin
        current_angle = np.arctan2(self.ball_speed_y, self.ball_speed_x)

        spin_direction = np.pi / 2 if self.ball_spin > 0 else -np.pi / 2

        new_angle = current_angle + spin_direction
        additional_velocity = abs(self.ball_spin) * 0.005

        self.ball_speed_x += additional_velocity * np.cos(new_angle)
        self.ball_speed_y += additional_velocity * np.sin(new_angle)

    def get_reward(self, action, ball_hit, Won, Lost):
        
        self.reward = 0
        if ball_hit:
            self.reward += 1
        if Won:
            self.reward += 10 
        elif Lost:
            self.reward -= 10
    
    def movement_handler(self):

        self.paddle1_speed *= 0.99
        self.paddle2_speed *= 0.99

        self.paddle1_y += self.paddle1_speed
        self.paddle2_y += self.paddle2_speed

        self.ball_x += self.ball_speed_x
        self.ball_y += self.ball_speed_y
                
    def step(self, action):
        # main stepping function
        self.take_action(action)

        self.movement_handler()

        self.get_paddle_collision()

        self.ball_spin_effect()

        ball_hit = self.get_ball_collisions()

        Won, Lost, done = self.get_done()

        self.get_reward(action, ball_hit, Won, Lost)

        observation = self.get_observation()

        truncated = False

        info = {"action": action, "self.reward": self.reward, "p1_won": Lost, "p2_won": Won}

        return observation, self.reward, done, truncated, info

    def reset(self, seed=None, options=None):

        self.reset_game()

        observation = self.get_observation()

        info = {}
        return observation, info

    def render(self, p1_score = 0, p2_score = 0):
        
        # Check for 'w' and 's' key presses
        if keyboard.is_pressed('w'):  # 'w' key for moving up
            self.paddle1_speed -= self.paddle_acceleration1
        elif keyboard.is_pressed('s'):  # 's' key for moving down
            self.paddle1_speed += self.paddle_acceleration1

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

        # Render the scores at the top middle of the screen

        font_scale = 0.5
        thickness = 1
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        score_text = f'Score: {p1_score} - {p2_score}'
        text_size = cv2.getTextSize(score_text, font, font_scale, thickness)[0]
        text_x = (self.width - text_size[0]) // 2
        text_y = text_size[1] + 20
        cv2.putText(self.img, score_text, (text_x, text_y), font, font_scale, white, thickness)

        cv2.imshow("Pong", self.img)
        cv2.waitKey(1)