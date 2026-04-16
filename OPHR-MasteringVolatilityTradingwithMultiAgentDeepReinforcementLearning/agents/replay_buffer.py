import random
from collections import deque
from typing import List, Tuple, Any
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool
    ):
        self.buffer.append((state, action, reward, next_state, done))
    
    def push_n_step(
        self,
        state: Any,
        action: int,
        rewards: List[float],
        next_state: Any,
        done: bool
    ):
        self.buffer.append((state, action, rewards, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()


class NStepBuffer:
    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)
    
    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool
    ):
        self.buffer.append((state, action, reward, next_state, done))
    
    def get_n_step_transition(self) -> Tuple:
        if len(self.buffer) < self.n_step:
            return None
        
        state, action = self.buffer[0][0], self.buffer[0][1]
        
        n_step_reward = 0.0
        actual_steps = 0
        
        for i, (_, _, r, _, d) in enumerate(self.buffer):
            n_step_reward += (self.gamma ** i) * r
            actual_steps = i + 1
            if d:  
                break
        
        _, _, _, n_step_next_state, done = self.buffer[actual_steps - 1]
        
        return state, action, n_step_reward, n_step_next_state, done
    
    def clear(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)



