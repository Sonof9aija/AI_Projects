import os
import pygame
import sys
import random
import math
import numpy as np
from tensorflow.keras import models, layers

pygame.init()

WIDTH, HEIGHT = 1000, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bouncing Ball Game")

# Colors
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

# Fonts
FONT = pygame.font.Font(None, 36)

# Ball properties
BALL_RADIUS = 10

# Game loop control
clock = pygame.time.Clock()


class Ball:

    def __init__(self, px, py, vx, vy):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        


    def move(self):
        self.px += self.vx
        self.py += self.vy

        if self.py > HEIGHT - BALL_RADIUS:
            self.py = HEIGHT - BALL_RADIUS
            self.vy *= -0.75  # reverse the direction of the velocity and reduce its magnitude to simulate a bounce

        elif self.py < BALL_RADIUS:
            self.py = BALL_RADIUS
            self.vy *= -0.75
           
    def check_collision(self, paddle):
        if self.px + BALL_RADIUS >= paddle.rect.x and self.vy > 0:
            if paddle.rect.y <= self.py + BALL_RADIUS and self.py + BALL_RADIUS <= paddle.rect.y + paddle.rect.height:
                self.vx *= -1
                self.px = paddle.rect.x - BALL_RADIUS



    def hit(self):
        self.vx *= -1
        self.vy = 10

    def reset(self):
        self.px = random.randint(BALL_RADIUS, WIDTH - BALL_RADIUS)
        self.py = random.randint(30 + BALL_RADIUS, HEIGHT - BALL_RADIUS)
        self.vx = random.randint(6, 12)
        self.vy = random.randint(4, 10)

class Scoreboard:

    def __init__(self, x, y):
        self.score = 0
        self.x = x
        self.y = y

    def increase_score(self):
        self.score += 1

    def draw(self):
        score_text = FONT.render(str(self.score), True, WHITE)
        screen.blit(score_text, (self.x, self.y))

class Paddle:

    def __init__(self, x, y, width=10, height=50, speed=10):
        self.rect = pygame.Rect(x, y, width, height)
        self.speed = speed

    def move_up(self):
        self.rect.y -= self.speed

        # Don't let the paddle move off the screen
        if self.rect.y < 0:
            self.rect.y = 0

    def move_down(self):
        self.rect.y += self.speed

        # Don't let the paddle move off the screen
        if self.rect.y > HEIGHT - self.rect.height:
            self.rect.y = HEIGHT - self.rect.height

    def move_left(self):
        if self.rect.x > WIDTH // 2 + self.rect.width:  # Check if the paddle is to the right of the middle
            self.rect.x -= self.speed
        elif self.rect.x < WIDTH // 2 + self.rect.width:  # Check if the paddle is on the left side
            self.rect.x -= self.speed


    def move_right(self):

        if self.rect.x < WIDTH // 2 - self.rect.width:  # Check if the paddle is to the right of the middle
            self.rect.x += self.speed
        elif self.rect.x > WIDTH // 2 - self.rect.width:  # Check if the paddle is on the left side
            self.rect.x += self.speed

    
    def draw(self):
        pygame.draw.rect(screen, WHITE, self.rect)

class AI:

    def __init__(self, paddle, ball):
        self.paddle = paddle
        self.ball = ball

    def update(self, ai_move):
        if ai_move < -0.5:
            self.paddle.move_up()
        elif ai_move > 0:
            self.paddle.move_down()
    def get_ai_move(self, state):
        paddle_move = model.predict(state.reshape(1, input_size))[0][0]
        return paddle_move
def draw_court():
    pygame.draw.line(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, 500), 15)

# Define the input and output sizes
input_size = 6 # ball x, ball y, ball vx, ball vy, paddle x, paddle y
output_size = 1 # paddle movement

# Define the model architecture
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(input_size,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(output_size)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')




# Define a function to get the movement of the AI paddle


# Define a function to train the model
def train_model(num_epochs=250, batch_size=32, learning_rate=0.001):
    # Create a new game and AI opponent
    ball = Ball(random.randint(550, 950), random.randint(30, 470), random.randint(6, 12), random.randint(4, 10))
    left_paddle = Paddle(50, 200)
    right_paddle = Paddle(940, 200)
    ai_opponent = AI(left_paddle, ball)

    # Create arrays to store training data
    states = []
    moves = []

    # Simulate games and record training data
    for i in range(num_epochs):
        # Reset the game
        ball.reset()
        left_paddle.rect.y = 200
        right_paddle.rect.y = 200

        # Play the game
        while True:
            # Get the current state of the game
            state = get_state(ball, left_paddle, right_paddle)

            # Get the movement of the AI paddle
            ai_move = ai_opponent.get_ai_move(state)
            # Move the AI paddle
            ai_opponent.update(ai_move)

            # Move the ball and check for collisions
            ball.move()
            if ball.px - BALL_RADIUS <= left_paddle.rect.x + left_paddle.rect.width and \
            ball.py + BALL_RADIUS >= left_paddle.rect.y and \
            ball.py - BALL_RADIUS <= left_paddle.rect.y + left_paddle.rect.height:
                ball.vx *= -1
                ball.px = left_paddle.rect.x + left_paddle.rect.width + BALL_RADIUS

            
            if ball.px + BALL_RADIUS >= right_paddle.rect.x and \
            ball.py + BALL_RADIUS >= right_paddle.rect.y and \
            ball.py - BALL_RADIUS <= right_paddle.rect.y + right_paddle.rect.height:
                ball.vx *= -1
                ball.px = right_paddle.rect.x - BALL_RADIUS

            # If the game is over, record the data and start a new game
            if ball.px > 1000 or ball.px < 0:
                states.append(state)
                moves.append(ai_move)
                break
            

    # Convert training data to arrays
    states = np.array(states)
    moves = np.array(moves)

    # Train the model
    model.fit(states, moves, batch_size=batch_size, epochs=num_epochs, verbose=0)

def get_state(ball, left_paddle, right_paddle):
        ball_x = ball.px
        ball_y = ball.py
        ball_vx = ball.vx
        ball_vy = ball.vy
        paddle_x = left_paddle.rect.x if left_paddle else right_paddle.rect.x
        paddle_y = left_paddle.rect.y if left_paddle else right_paddle.rect.y
        state = np.array([ball_x, ball_y, ball_vx, ball_vy, paddle_x, paddle_y])
        return state



class Game:

    def __init__(self):
        self.left_scoreboard = Scoreboard(200, 10)
        self.right_scoreboard = Scoreboard(750, 10)
        self.ball = Ball(random.randint(550, 950), random.randint(30, 470), random.randint(6, 12), random.randint(4, 10))
        self.left_paddle = Paddle(50, 200)
        self.right_paddle = Paddle(940, 200)

        self.ai_opponent = AI(self.left_paddle, self.ball)
        self.left_paddle_moving_up = False
        self.left_paddle_moving_down = False
        self.right_paddle_moving_up = False
        self.right_paddle_moving_down = False

        self.left_paddle_moving_left = False
        self.left_paddle_moving_right = False
        self.right_paddle_moving_left = False
        self.right_paddle_moving_right = False

    # Define a function to get the state of the game
    def get_state(self):
        ball_x = self.ball.px
        ball_y = self.ball.py
        ball_vx = self.ball.vx
        ball_vy = self.ball.vy
        paddle_x = self.left_paddle.rect.x if self.left_paddle else self.right_paddle.rect.x
        paddle_y = self.left_paddle.rect.y if self.left_paddle else self.right_paddle.rect.y
        state = np.array([ball_x, ball_y, ball_vx, ball_vy, paddle_x, paddle_y])
        return state

    def game_loop(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        self.left_paddle_moving_up = True
                    elif event.key == pygame.K_s:
                        self.left_paddle_moving_down = True
                    elif event.key == pygame.K_UP:
                        self.right_paddle_moving_up = True
                    elif event.key == pygame.K_DOWN:
                        self.right_paddle_moving_down = True
                    elif event.key == pygame.K_a:
                        self.left_paddle_moving_left = True
                    elif event.key == pygame.K_d:
                        self.left_paddle_moving_right = True
                    elif event.key == pygame.K_LEFT:
                        self.right_paddle_moving_left = True
                    elif event.key == pygame.K_RIGHT:
                        self.right_paddle_moving_right = True
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_w:
                        self.left_paddle_moving_up = False
                    elif event.key == pygame.K_s:
                        self.left_paddle_moving_down = False
                    elif event.key == pygame.K_UP:
                        self.right_paddle_moving_up = False
                    elif event.key == pygame.K_DOWN:
                        self.right_paddle_moving_down = False
                    elif event.key == pygame.K_a:
                        self.left_paddle_moving_left = False
                    elif event.key == pygame.K_d:
                        self.left_paddle_moving_right = False
                    elif event.key == pygame.K_LEFT:
                        self.right_paddle_moving_left = False
                    elif event.key == pygame.K_RIGHT:
                        self.right_paddle_moving_right = False

            if self.left_paddle_moving_up:
                self.left_paddle.move_up()
            elif self.left_paddle_moving_down:
                self.left_paddle.move_down()
            if self.left_paddle_moving_left:
                self.left_paddle.move_left()
            elif self.left_paddle_moving_right:
                self.left_paddle.move_right()
            if self.right_paddle_moving_up:
                self.right_paddle.move_up()
            elif self.right_paddle_moving_down:
                self.right_paddle.move_down()
            if self.right_paddle_moving_left:
                self.right_paddle.move_left()
            elif self.right_paddle_moving_right:
                self.right_paddle.move_right()

            screen.fill(BLUE)
            draw_court()
            self.left_scoreboard.draw()
            self.right_scoreboard.draw()
            self.left_paddle.draw()
            self.right_paddle.draw()

            pygame.draw.circle(screen, WHITE, (int(self.ball.px), int(self.ball.py)),
                            BALL_RADIUS)

            # Get the current state of the game
            state = self.get_state()

            # Get the movement of the AI paddle
            ai_move = self.ai_opponent.get_ai_move(state)

            # Move the AI paddle
            self.ai_opponent.update(ai_move)

            self.ball.move()
            curr_x = self.ball.px + self.ball.vx

            if self.ball.px - BALL_RADIUS <= self.left_paddle.rect.x + self.left_paddle.rect.width and \
            self.ball.py + BALL_RADIUS >= self.left_paddle.rect.y and \
            self.ball.py - BALL_RADIUS <= self.left_paddle.rect.y + self.left_paddle.rect.height:
                self.ball.vx *= -1
                self.ball.px = self.left_paddle.rect.x + self.left_paddle.rect.width + BALL_RADIUS

            
            if self.ball.px + BALL_RADIUS >= self.right_paddle.rect.x and \
            self.ball.py + BALL_RADIUS >= self.right_paddle.rect.y and \
            self.ball.py - BALL_RADIUS <= self.right_paddle.rect.y + self.right_paddle.rect.height:
                self.ball.vx *= -1
                self.ball.px = self.right_paddle.rect.x - BALL_RADIUS


            if curr_x > 1000:
                self.left_scoreboard.increase_score()
                self.ball.reset()
            elif curr_x < 0:
                self.right_scoreboard.increase_score()
                self.ball.reset()

            pygame.display.flip()
            clock.tick(30)

train_model()
fun = Game()
fun.game_loop()




