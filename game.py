import random
import sys
from dataclasses import dataclass
from enum import Enum

import pygame

class Colors:
    WHITE = (255, 255, 255)
    RED = (200, 0, 0)
    BLUE1 = (0, 0, 255)
    BLUE2 = (0, 100, 255)
    BLACK = (0, 0, 0)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

@dataclass(frozen=True)
class Point:
    x: int
    y: int

class SnakeGame:
    def __init__(self, w=640, h=480, block_size=20, speed=10, show_game=True):
        self.w = w
        self.h = h
        self.block_size = block_size
        self.speed = speed
        # show game can be false for faster training
        self.show_game = show_game

        self.font = None
        self.display = None
        self.clock = None

        if self.show_game:
            pygame.init()
            self.font = pygame.font.SysFont("arial", 25)
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake")
            self.clock = pygame.time.Clock()
        
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(
            (self.w // 2 // self.block_size) * self.block_size,
            (self.h // 2 // self.block_size) * self.block_size
        )
        self.snake = [
            self.head,
            Point(self.head.x - self.block_size, self.head.y),
            Point(self.head.x - (2 * self.block_size), self.head.y)
        ]

        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self.place_food()

    def place_food(self):
        while True:
            x = random.randint(0, (self.w - self.block_size) // self.block_size) * self.block_size
            y = random.randint(0, (self.h - self.block_size) // self.block_size) * self.block_size
            food = Point(x, y)
            if food not in self.snake:
                self.food = food
                return

    def read_manual_input(self, action):
        if not self.show_game:
            return action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                if event.key == pygame.K_LEFT:
                    if self.direction == Direction.UP:
                        action = [0, 0, 1]
                    elif self.direction == Direction.DOWN:
                        action = [0, 1, 0]

                elif event.key == pygame.K_RIGHT:
                    if self.direction == Direction.UP:
                        action = [0, 1, 0]
                    elif self.direction == Direction.DOWN:
                        action = [0, 0, 1]

                elif event.key == pygame.K_UP:
                    if self.direction == Direction.LEFT:
                        action = [0, 1, 0]
                    elif self.direction == Direction.RIGHT:
                        action = [0, 0, 1]

                elif event.key == pygame.K_DOWN:
                    if self.direction == Direction.LEFT:
                        action = [0, 0, 1]
                    elif self.direction == Direction.RIGHT:
                        action = [0, 1, 0]

        return action

    def play_step(self, action):
        self.frame_iteration += 1

        # override with manual control if the user wants to play
        action = self.read_manual_input(action)

        self.move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False

        if self.is_collision() or self.frame_iteration > 1_000 * len(self.snake):
             # stop if the snake survives too long without making progress
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()
        
        if self.show_game:
            self.update_ui()
            self.clock.tick(self.speed)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # boundary hit check
        if pt.x < 0 or pt.x >= self.w or pt.y < 0 or pt.y >= self.h:
            return True
        
        # self hit check
        if pt in self.snake[1:]:
            return True

        return False

    def update_ui(self):
        self.display.fill(Colors.BLACK)

        for pt in self.snake:
            pygame.draw.rect(
                self.display, 
                Colors.BLUE1, 
                pygame.Rect(pt.x, pt.y, self.block_size, self.block_size)
            )
            pygame.draw.rect(
                self.display,
                Colors.BLUE2,
                pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)
            )

        pygame.draw.rect(
            self.display,
            Colors.RED,
            pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size)
        )

        text = self.font.render("Score: " + str(self.score), True, Colors.WHITE)
        self.display.blit(text, (0, 0))
        pygame.display.flip()

    def move(self, action):
        # action = [straight, right, left]

        # directions are ordered so turning right/left is just an index shift
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == [1, 0, 0]: # straight, no change
            new_dir = clock_wise[idx] 
        elif action == [0, 1, 0]: # right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] 
        elif action == [0, 0, 1]: # left turn
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] 
        else:
            raise ValueError("Invalid action")

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size

        self.head = Point(x, y)

if __name__ == "__main__":
    game = SnakeGame()

    while True:
        reward, game_over, score = game.play_step([1, 0, 0])

        if game_over:
            print(f"Game Over - Final score: {score}")
            break

    pygame.quit()