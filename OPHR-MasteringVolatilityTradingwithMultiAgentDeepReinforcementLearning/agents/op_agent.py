import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List
import random

from agents.replay_buffer import ReplayBuffer, NStepBuffer


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 3):
        super(QNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class OPAgent:
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        n_step: int = 24,
        replay_buffer_size: int = 100000,
        batch_size: int = 256,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        update_frequency: int = 4,
        target_update_frequency: int = 1000,
        device: str = None
    ):
        if device is None:
            try:
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    self.device = torch.device('cuda')
                else:
                    self.device = torch.device('cpu')
            except:
                self.device = torch.device('cpu')
        else:
            try:
                self.device = torch.device(device)
            except:
                print(f"Warning: Device {device} not available, using CPU")
                self.device = torch.device('cpu')
        
        self.q_network = QNetwork(state_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.n_step_buffer = NStepBuffer(n_step, gamma)
    
        self.steps = 0
        self.episodes = 0
    
    def extract_features(self, state: dict) -> np.ndarray:
        vola_features = state.get('volatility_tickers', np.array([]))
        perp_features = state.get('features', np.array([]))
        
        if len(vola_features) > 0 and len(perp_features) > 0:
            features = np.concatenate([vola_features, perp_features])
        elif len(vola_features) > 0:
            features = vola_features
        elif len(perp_features) > 0:
            features = perp_features
        else:
            features = np.zeros(48)  
        
        return features.astype(np.float32)
    
    def select_action(self, state: dict, epsilon: float = None) -> int:
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randint(0, 2)
        else:
            features = self.extract_features(state)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(features_tensor)
            
            return q_values.argmax().item()
    
    def action_to_direction(self, action: int) -> int:
        return [1, 0, -1][action]
    
    def direction_to_action(self, direction: int) -> int:
        return {1: 0, 0: 1, -1: 2}[direction]
    
    def store_transition(
        self,
        state: dict,
        action: int,
        reward: float,
        next_state: dict,
        done: bool
    ):
        self.n_step_buffer.push(state, action, reward, next_state, done)
        n_step_transition = self.n_step_buffer.get_n_step_transition()
        
        if n_step_transition is not None:
            s, a, n_step_r, n_step_s, d = n_step_transition
            self.replay_buffer.push_n_step(s, a, [n_step_r], n_step_s, d)
    
    def update(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        states, actions, rewards_list, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        state_features = np.array([self.extract_features(s) for s in states])
        next_state_features = np.array([self.extract_features(s) for s in next_states])
        
        state_batch = torch.FloatTensor(state_features).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_features).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        
        n_step_rewards = torch.FloatTensor([sum(r_list) for r_list in rewards_list]).to(self.device)
        
        current_q = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.q_network(next_state_batch).argmax(1, keepdim=True)
            next_q = self.target_network(next_state_batch).gather(1, next_actions)
            
            target_q = n_step_rewards.unsqueeze(1) + \
                       (self.gamma ** self.n_step) * next_q * (1 - done_batch.unsqueeze(1))
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes
        }, filepath)
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']


