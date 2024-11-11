# Projects
A collection of some of the small projects I have with languages like Python, Rust and Java

# Rust

### Sorting algorithms:

Made in August 2023, mostly because I just wanted to try out Rust and programming with it.
I implemented the algorithms based only on descriptions of how they work, so their performance is not always optimal, but I learned a lot.

You can run the code from the sorting_algo.exe file, and the source code is in src/main.rs

### rust_raytracer:

Be sure to see the [Fully featured GPU raytracer](https://github.com/juhotuho10/rust_GPU_raytracing)

Made in June 2024, A primitive CPU raytracer made in rust before I started to move everything to the much larger GPU taytracer version of this project

# Python

###  ML and data analytics projects:

(More of my project work can be found at the link below) Link to my Kaggle profile for data analysis / ML prediction projects:
https://www.kaggle.com/juhotuho10/code

### Lotto Simulator:

- Created in January 2019
- One of the first projects I made
- Completed a high school computer science course with this project
- The project is a lottery simulator with a user interface
- You can try to see how many years it would take to win the lottery (spoiler: quite a few)

###  Alcoholic Spider:

- Created in September 2021
- A fun web crawler suggested by a friend that fetches the HTML code from the Alko website
- It extracts product data for all alcoholic beverages and calculates the alcohol percentage per liter per euro value
- The program determines the best cost-to-alcohol ratio drink from Alko

Note: The program has to parse a lot of HTML code, so loading it into RAM and running it can be a bit slow, taking about 10-20 minutes. 
the project is currently a litte broken fron changes Alko has made to their website

###  Minesweeper:

- Created in November 2021
- Final project for a university Programming 1 course, done individually
- A simple Minesweeper game
- At the start, you set the grid size and number of mines
- In the game, left-click opens a tile, and right-click places a flag


###  ML-Snake:

- Created in August 2022
- A modified Snake game driven by an ML algorithm with a custom reward function and vision signals
- The environment evolved through iterative changes, with each new class inheriting from the previous one, resulting in the latest iteration (SnekEnv12) being the most refined

Snake vision signals (simplified):

- Position of the apple relative to the head: X and Y [-1, 0, 1] (doesn't give distance, only direction)
- Position of the snake’s middle body part relative to the head: X and Y [-1, 0, 1]
- Position of the snake’s tail relative to the head: X and Y [-1, 0, 1]
- Proximity of "danger tiles" (body parts or edges) in each direction

Rewards (simplified):

- Moving closer to the apple: (+)
- Moving farther from the apple: (-) (heavier penalty to prevent the snake from circling aimlessly)
- Being far from danger tiles: (+) or close to danger tiles: (-) (encourages safety over taking the shortest route to the apple)
- Reaching the apple: (+)
- Dying: (-)

The vision signals and rewards are relatively simple but required a lot of thought, theory, and trial-and-error to simplify effectively.
Note: The original Snake game was not made by me, but most of it has been replaced, except for rendering. (You can find the original game in original_snake_game.py if you're interested in comparing or trying to outperform the ML algorithm.)

You can run the pre-trained model using snek_game_loading.py to see it in action.
Everything else is already set up; no changes needed.
You can set RENDER = False to quickly run 100 games and print the final length to the console after each death.
The best length achieved was 178.


###  Brawlhalla ML:

- Created in August 2023
- Wanted to experiment with creating an ML bot for the game Brawlhalla: https://store.steampowered.com/app/291550/Brawlhalla/

Note: I haven’t trained an agent yet because it requires real-time training, and I don’t have a custom environment. Training would take several weeks, and I don’t have the time right now.
In the demo video (Demo_recording), I'm playing an offline match against a harmless training bot to showcase functionality.

The program takes black-and-white screenshots of the Brawlhalla window, converts them to a 540x960 numpy array, and feeds this into the ML algorithm.
It captures both health bars and uses changes in the bars to generate rewards:

- The more damage dealt, the higher the reward
- Dying: -10 reward
- Making the opponent die: +10 reward

###  ML-Pong:

- Created in December 2023
- It's Pong, but with a twist: the ball behaves like a hyperactive bouncy ball, affected by exaggerated physics
- The ball is unpredictable, and as its speed increases over time, the game becomes quite intense

You can play against the ML model, and the game keeps score of points won by each side.

Instructions:
Run Pong_loading.py to play.

Controls:

- w = up
- s = down

ML model vision signals:

- Difference between the ball’s y-coordinate and the center of the paddle, normalized
- Paddle y-coordinate, normalized
- Ball x and y coordinates, normalized
- Ball x and y velocities, normalized

Rewards:

- Hitting the ball with the paddle: (+)
- Scoring a point: (+)
- Opponent scoring a point: (-)

# Java

### WeatherApp:

- Created in October 2023
- Final project for Programming 3 course
- The project was supposed to be a group assignment, but in reality, I completed it on my own
- It's a convenient weather application with an impressive interactive UI that uses the OpenWeather API

All project documentation can be found in the Documentation folder.

To run the application, open the command prompt and execute:

```mvn package```

and then:

```java -jar target\WeatherApp-1.0.one-jar.jar```

Note: You need to have Java and Maven installed on your system.

