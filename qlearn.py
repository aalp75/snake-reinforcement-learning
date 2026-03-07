import os
import numpy as np

class Qlearn:
    """
    Q-learning algorithm

    The state is represented by 11 binary features (2^11 = 2048 possible states)

    The Q-table has shape (2048, 3), where rows correspond to states and
    columns correspond to the three possible actions:
        [1, 0, 0]: straight
        [0, 1, 0]: turn right
        [0, 0, 1]: turn left
    """
    def __init__(self):
        self.q = np.zeros((1 << 11, 3))
    
    def convert_state(self,state):
        idx = 0
        for i in range(11):
            idx += state[i] * (1 << i)
        return idx
    
    def convert_action(self, action):
        return int(np.argmax(action))
    
    def predict(self,state):
        idx = self.convert_state(state)
        best_action_index = int(np.argmax(self.q[idx]))
        
        best_action = [0, 0, 0]
        best_action[best_action_index] = 1
        return best_action
    
    def train(self, state, action, next_state, reward, learning_rate, discount_factor):
        state_idx = self.convert_state(state)
        action_idx = self.convert_action(action)
        next_state_idx = self.convert_state(next_state)

        best_next_q = np.max(self.q[next_state_idx])
        
        self.q[state_idx, action_idx] += learning_rate * (
            reward
            + discount_factor * best_next_q
            - self.q[state_idx,action_idx]
        )
        
    def save(self):
        os.makedirs("parameters", exist_ok=True)
        with open("parameters/qlearn_cfg.npy", "wb") as file:
            print("-- Config saved --")
            np.save(file, self.q)

    def load(self):
        with open("parameters/qlearn_cfg.npy", "rb") as file:
            print("-- Config loaded --")
            self.q = np.load(file)