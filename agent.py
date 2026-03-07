import numpy as np
import random
import time
from game import SnakeGame, Direction, Point
from qlearn import Qlearn
from deepqlearn import DeepQlearn

class Agent:
    
    def __init__(self, mode, model_type):
        
        self.nb_games = 0
        self.epsilon = 0
        self.discount_factor = 0.9
        self.learning_rate = 0.001
        self.ite = 0
        self.record = 0
        self.mode = mode
        
        self.model_type = model_type

        if self.model_type == "deepqlearn":
            self.model = DeepQlearn(self.learning_rate)

        if self.model_type == "qlearn":
            self.model = Qlearn()
        
        if self.mode == "play":
            self.model.load()
            self.epsilon = 0
        elif self.mode == "train":
            self.epsilon = 0.8
        else:
            raise ValueError("Unknown mode")
            
        self.state_memory = []
        self.action_memory = []
        self.next_state_memory = []
        self.reward_memory = []
        self.done_memory = []
        
        self.memory_size = 0
        
        self.max_memory = 10000
        
    def get_state(self,game):

        head = game.snake[0]

        point_l = Point(head.x - game.block_size, head.y)
        point_r = Point(head.x + game.block_size, head.y)
        point_u = Point(head.x, head.y - game.block_size)
        point_d = Point(head.x, head.y + game.block_size)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # danger straight
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_r)),

            # danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger left
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food location
            game.food.x < game.head.x, # food is on the left
            game.food.x > game.head.x, # food is on the right
            game.food.y < game.head.y, # food is above
            game.food.y > game.head.y  # food is down
        ]
        return np.array(state,dtype=int)
        
        
    def get_move(self, state):
        U = random.uniform(0, 1)
        action = [0, 0, 0]
        if U < self.epsilon:
            action[random.randrange(0, 2)] = 1
            return action
        else:
            if self.model_type == "deepqlearn":
                action[np.argmax(self.model.predict(state))] = 1
                return action
            elif self.model_type == "qlearn":
                return self.model.predict(state)
               
    def replay(self, batch_size=32): # replay experience
        
        batch = np.random.choice(len(self.state_memory), batch_size, replace=False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.done_memory[batch]
        
        targets = self.model.predict(states)
        targets_f = self.model.predict(next_states)
        
        for idx in range(batch_size):
            y = rewards[idx][0]
            if not dones[idx]:
                y += self.discount_factor * max(targets_f[idx])       
            targets[idx][np.argmax(actions[idx])] = y
        
        self.model.model.fit(states, targets, verbose=0)
        
    def train(self, state, action, reward, next_state, done):
        if self.mode == "train":
            if self.model_type == "qlearn":
                self.model.train(state, action, next_state, reward, self.learning_rate, self.discount_factor)
            else:
                self.remember(state, action, reward, next_state, done)
                if (self.memory_size > 32):
                    self.replay()
        
    def remember(self,state,action,reward,next_state,done):
        
        self.memory_size +=1
        
        action = np.reshape(action,((1, 3)))
        reward = np.reshape(reward,(1, 1))
        done = np.reshape(done,(1, 1))
        
        if self.memory_size == 1:
            self.state_memory = state
            self.action_memory = action
            self.reward_memory = reward
            self.next_state_memory = next_state
            self.done_memory = done       
            return
            
        if self.memory_size <= self.max_memory:
            self.state_memory = np.concatenate((self.state_memory, state), axis=0)
            self.action_memory = np.concatenate((self.action_memory, action), axis=0)   
            self.reward_memory = np.concatenate((self.reward_memory, reward), axis=0) 
            self.next_state_memory = np.concatenate((self.next_state_memory, next_state), axis=0) 
            self.done_memory = np.concatenate((self.done_memory,done), axis=0)
            
        else:

            index = self.memory_size % self.max_memory
            
            self.state_memory[index] =state
            self.action_memory[index] = action
            self.reward_memory[index] = reward
            self.next_state_memory[index] = next_state
            self.done_memory[index] = done
        
    def save(self):
        self.model.save()
        
    def load(self):
        self.model.load()
    
def play(mode, model):
    
    # exponential moving average to track the evolution over the last games
    mean_score = 0
    alpha = 0.95
    
    # create the game and the agent
    speed = 10 if mode == "play" else 10_000 # fast training
    agent = Agent(mode, model)
    game = SnakeGame(h=800, w=800, speed=speed)
    current_time = time.time()

    while True:
        state = agent.get_state(game)
        if agent.model_type == "deepqlearn":
            # reshape to be in the expected format for the neural network
            state = np.reshape(state, (1, 11))
        
        action = agent.get_move(state)
        
        reward, done, score = game.play_step(action)
        
        next_state = agent.get_state(game)
        if agent.model_type == "deepqlearn":
            next_state = np.reshape(next_state, (1, 11))
            
        agent.train(state, action, reward, next_state, done)
        
        state = next_state
        
        if done:
            agent.ite +=1
            agent.epsilon = agent.epsilon * 0.9 # lower the epsilon exploration parameter
            
            if (game.score > agent.record): # best result so far
                agent.record = game.score
                agent.save() # save the new best config
            
            mean_score = round(alpha * mean_score + (1 - alpha) * score, 2)
            timer = round(time.time() - current_time, 2)
            
            print(f"Game {agent.ite} Score {game.score} Record {agent.record} \
                    EMA {mean_score} Time {timer} seconds")
                
            game.reset() # launch a new game

         
if __name__ == '__main__':
    
    mode = "train" # or train
    model = "qlearn" # or deepqlearn

    play(mode, model)
        
        
