import pygame
import random
from enum import Enum
from collections import namedtuple
import sys

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

SPEED = 10

class SnakeGame:

    def __init__(self, w=640, h=480, block_size=20, speed=10):
        self.w = w
        self.h = h
    
        self.block_size = block_size
        self.speed = speed

        
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        self.reset()


    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point((self.w // 2 // self.block_size) * self.block_size,
                          (self.h // 2 // self.block_size) * self.block_size)
        self.snake = [self.head,
                      Point(self.head.x-self.block_size, self.head.y),
                      Point(self.head.x-(2*self.block_size), self.head.y)]

        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0


    def place_food(self):
        while True:
            x = random.randint(0, (self.w - self.block_size) // self.block_size) * self.block_size
            y = random.randint(0, (self.h - self.block_size) // self.block_size) * self.block_size
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def read_manual_input(self, action):
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
                        action = [0,0,1]

                    elif self.direction == Direction.DOWN:
                        action = [0,1,0]

                elif event.key == pygame.K_RIGHT:

                    if self.direction == Direction.UP:
                        action = [0,1,0]

                    elif self.direction == Direction.DOWN:
                        action = [0,0,1]

                elif event.key == pygame.K_UP:

                    if self.direction == Direction.LEFT:
                        action = [0,1,0]

                    elif self.direction == Direction.RIGHT:
                        action = [0,0,1]

                elif event.key == pygame.K_DOWN:

                    if self.direction == Direction.LEFT:
                        action = [0,0,1]

                    elif self.direction == Direction.RIGHT:
                        action = [0,1,0]

        return action

    def play_step(self, action):
        self.frame_iteration += 1

        action = self.read_manual_input(action)

        self.move(action) # update the head
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            # game over
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()
        
        # update ui
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
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == [1, 0, 0]:
            new_dir = clock_wise[idx] # no change
        elif action == [0, 1, 0]:
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        elif action == [0, 0, 1]:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
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
            print(f"Game Over (Final score: {score})")
            break

    pygame.quit()