import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config import HedgerPoolConfig
import random

from hedgers.delta_hedger import DeltaThresholdHedger
from hedgers.deep_hedger import DeepHedger
from hedgers.price_move_hedger import PriceMoveHedger
from agents.replay_buffer import ReplayBuffer


class HRQNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], num_hedgers: int):
        super(HRQNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_hedgers))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class HRAgent:
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [128, 128],
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        replay_buffer_size: int = 50000,
        batch_size: int = 128,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        n_hr: int = 24,  # Decision interval (hours)
        device: str = None,
        hedger_pool_config: Optional['HedgerPoolConfig'] = None,  # New: config-based hedger pool
        update_frequency: int = 1,
        target_update_frequency: int = 100
    ):
        self.hedger_pool_config = hedger_pool_config
        self.hedgers = self._create_hedger_pool(hedger_pool_config)
        self.num_hedgers = len(self.hedgers)

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
        
        self.q_network = HRQNetwork(state_dim, hidden_dims, self.num_hedgers).to(self.device)
        self.target_network = HRQNetwork(state_dim, hidden_dims, self.num_hedgers).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_hr = n_hr
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        self.current_hedger_idx = 0
        self.steps_since_decision = 0
        
        self.steps = 0
        self.episodes = 0
    
    def _create_hedger_pool(self, config: Optional['HedgerPoolConfig'] = None) -> List:
        hedgers = []
        
        if config is None:
            hedgers = [
                DeltaThresholdHedger(delta_threshold=0.05, hedge_ratio=1.0),
                DeltaThresholdHedger(delta_threshold=0.10, hedge_ratio=1.0),
                DeltaThresholdHedger(delta_threshold=0.20, hedge_ratio=1.0),
                DeltaThresholdHedger(delta_threshold=0.30, hedge_ratio=1.0),
                DeltaThresholdHedger(delta_threshold=3.0, hedge_ratio=0.0),
                PriceMoveHedger(price_move_threshold=0.01, hedge_ratio=1.0),
                PriceMoveHedger(price_move_threshold=0.02, hedge_ratio=1.0),
                PriceMoveHedger(price_move_threshold=0.03, hedge_ratio=1.0),
            ]
        else:
            for threshold in config.delta_hedger_thresholds:
                hedgers.append(
                    DeltaThresholdHedger(
                        delta_threshold=threshold,
                        hedge_ratio=config.delta_hedger_ratio
                    )
                )
            for threshold in config.price_hedger_thresholds:
                hedgers.append(
                    PriceMoveHedger(
                        price_move_threshold=threshold,
                        hedge_ratio=config.price_hedger_ratio
                    )
                )
            for model_config in config.deep_hedger_models:
                try:
                    import os
                    model_path = model_config.get('model_path', '')
                    if os.path.exists(model_path):
                        hedgers.append(
                            DeepHedger(
                                model_path=model_path,
                                feature_config=config.deep_hedger_feature_config,
                                device=model_config.get('device', 'cpu')
                            )
                        )
                    else:
                        print(f"Warning: Deep hedger model not found: {model_path}")
                        print(f"  Skipping this deep hedger")
                except Exception as e:
                    print(f"Warning: Failed to load deep hedger: {e}")
                    print(f"  Model path: {model_config.get('model_path', 'N/A')}")
        
        if len(hedgers) == 0:
            raise ValueError("No valid hedgers created! Check configuration or model paths.")
        
        print(f"Created hedger pool with {len(hedgers)} hedgers:")
        print(f"  - Delta-based: {sum(1 for h in hedgers if isinstance(h, DeltaThresholdHedger))}")
        print(f"  - Price-based: {sum(1 for h in hedgers if isinstance(h, PriceMoveHedger))}")
        print(f"  - Deep hedgers: {sum(1 for h in hedgers if isinstance(h, DeepHedger))}")
        
        return hedgers
    
    def extract_features(
        self,
        state: dict,
        position,
        greeks: tuple
    ) -> np.ndarray:
        vola_features = state.get('volatility_tickers', np.array([]))
        perp_features = state.get('features', np.array([]))

        if len(vola_features) > 0 and len(perp_features) > 0:
            market_features = np.concatenate([vola_features, perp_features])
        elif len(vola_features) > 0:
            market_features = vola_features
        elif len(perp_features) > 0:
            market_features = perp_features
        else:
            market_features = np.zeros(48)
        
        num_option_positions = len(position.option_positions)
        perp_position = float(position.perpetual_position.net_quantity)
        position_features = np.array([num_option_positions, perp_position])
        
        delta, gamma, theta, vega = greeks
        greeks_features = np.array([
            float(delta),
            float(gamma),
            float(theta),
            float(vega)
        ])
    
        features = np.concatenate([
            market_features,
            position_features,
            greeks_features
        ])
        
        return features.astype(np.float32)
    
    def select_hedger(
        self,
        state: dict,
        position,
        greeks: tuple,
        epsilon: float = None
    ) -> int:

        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            # Random hedger
            return random.randint(0, self.num_hedgers - 1)
        else:
            # Greedy hedger selection
            features = self.extract_features(state, position, greeks)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(features_tensor)
            
            return q_values.argmax().item()
    
    def get_current_hedger(self) -> DeltaThresholdHedger:
        return self.hedgers[self.current_hedger_idx]
    
    def should_make_decision(self) -> bool:
        return self.steps_since_decision >= self.n_hr
    
    def step(
        self,
        state: dict,
        position,
        greeks: tuple
    ) -> int:
        self.steps_since_decision += 1
        
        if self.should_make_decision():
            # Make new hedger selection
            hedger_idx = self.select_hedger(state, position, greeks)
            self.current_hedger_idx = hedger_idx
            self.steps_since_decision = 0
        
        return self.current_hedger_idx
    
    def compute_hedge(
        self,
        delta,
        gamma,
        theta,
        vega,
        position_info: dict = None,
        market_info: dict = None
    ):
        current_hedger = self.get_current_hedger()
        
        return current_hedger.compute_hedge(
            delta, gamma, theta, vega,
            position_info or {},
            market_info or {}
        )
    
    def store_transition(
        self,
        state: dict,
        position,
        greeks: tuple,
        hedger_idx: int,
        reward: float,
        next_state: dict,
        next_position,
        next_greeks: tuple,
        done: bool
    ):

        state_features = self.extract_features(state, position, greeks)
        next_state_features = self.extract_features(next_state, next_position, next_greeks)

        self.replay_buffer.push(
            state_features,
            hedger_idx,
            reward,
            next_state_features,
            done
        )
    
    def update(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        state_batch = torch.FloatTensor(np.array(states)).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_states)).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.q_network(next_state_batch).argmax(1, keepdim=True)
            next_q = self.target_network(next_state_batch).gather(1, next_actions)
            target_q = reward_batch.unsqueeze(1) + self.gamma * next_q * (1 - done_batch.unsqueeze(1))
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_decision_counter(self):
        self.steps_since_decision = 0
    
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
    
    def get_hedger_info(self) -> List[str]:
        return [str(hedger) for hedger in self.hedgers]


