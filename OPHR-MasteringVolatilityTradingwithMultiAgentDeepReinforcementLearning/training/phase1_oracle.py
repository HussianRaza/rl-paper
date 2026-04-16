import datetime
import numpy as np
from tqdm import tqdm
from typing import Tuple

from env.data.data_handler import DataHandler
from env.base_env import BaseEnv
from env.rl_env import RLEnv
from oracle.oracle_policy import OraclePolicy
from agents.op_agent import OPAgent
from config import ConfigManager
from training.checkpoint_utils import save_replay_buffer, save_stats, get_checkpoint_path

def collect_oracle_experience(
    crypto: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    num_episodes: int,
    config_manager: ConfigManager,
    data_paths: dict,
    op_agent: OPAgent = None,
    checkpoint_dir: str = "checkpoints/phase1.1_oracle",
    save_checkpoint: bool = True,
    verbose: bool = True
) -> Tuple[OPAgent, dict]:
    if verbose:
        print("=" * 60)
        print("Phase 1: Oracle Policy Initialization")
        print("=" * 60)
    
    # Get configurations
    env_config = config_manager.env_config
    training_config = config_manager.training_config
    
    # Initialize environment
    data_handler = DataHandler(
        start_date, end_date, crypto,
        data_paths['option_chain'],
        data_paths['perpetual'],
        volatility_ticker_path=data_paths.get('volatility_ticker')
    )
    
    time_config = {
        'episode_length': env_config.episode_length,
        'option_interval': env_config.option_interval,
        'include_volatility_tickers': True
    }
    
    base_env = BaseEnv(data_handler, time_config, crypto=crypto)
    env = RLEnv(base_env)
    
    # Initialize Oracle policy
    oracle = OraclePolicy(
        beta=training_config.oracle_beta,
        lookforward_window=training_config.oracle_lookforward_window
    )
    
    if verbose:
        print(f"\nOracle Settings:")
        print(f"  Beta: {oracle.beta}")
        print(f"  Lookforward window: {oracle.lookforward_window}h")
    
    # Initialize OP-Agent if not provided
    if op_agent is None:
        # Determine state dimension
        state, info = env.reset(start_date, end_date)
        sample_features = np.concatenate([
            state.get('volatility_tickers', np.zeros(48)),
            state.get('features', np.zeros(48))
        ])
        state_dim = len(sample_features)
        
        op_agent = OPAgent(
            state_dim=state_dim,
            hidden_dims=training_config.op_hidden_dims,
            learning_rate=training_config.op_learning_rate,
            gamma=training_config.op_gamma,
            n_step=training_config.op_n_step,
            replay_buffer_size=training_config.op_replay_buffer_size,
            batch_size=training_config.op_batch_size
        )
        
        if verbose:
            print(f"\nOP-Agent initialized:")
            print(f"  State dim: {state_dim}")
            print(f"  Hidden dims: {training_config.op_hidden_dims}")
            print(f"  n-step: {training_config.op_n_step}")
    
    # Statistics
    stats = {
        'episode_returns': [],
        'episode_lengths': [],
        'long_signals': 0,
        'short_signals': 0,
        'neutral_signals': 0
    }
    
    # Collect experience
    if verbose:
        print(f"\nCollecting Oracle experience ({num_episodes} episodes)...")
    
    for episode in tqdm(range(num_episodes), disable=not verbose):
        state, info = env.reset()
        done = False
        episode_return = 0
        episode_length = 0
        
        oracle.reset()
        
        while not done:
            # Get Oracle action
            option_action, hedge_action = oracle.step(
                state, info['account'], info['position'], info
            )
            
            # Convert Oracle signal to OP-Agent action
            signal = oracle.generate_signal(state, info)
            op_action = op_agent.direction_to_action(signal)
            
            # Count signals
            if signal == 1:
                stats['long_signals'] += 1
            elif signal == -1:
                stats['short_signals'] += 1
            else:
                stats['neutral_signals'] += 1
            
            # Step environment
            next_state, reward, done, next_info = env.step(option_action, hedge_action)
            
            # Store transition in OP-Agent's replay buffer
            op_agent.store_transition(state, op_action, reward, next_state, done)
            
            episode_return += reward
            episode_length += 1
            
            state = next_state
            info = next_info
        
        # Clear n-step buffer at episode boundary to prevent cross-episode transitions
        op_agent.n_step_buffer.clear()
        
        stats['episode_returns'].append(episode_return)
        stats['episode_lengths'].append(episode_length)
    
    # Summary statistics
    if verbose:
        print(f"\n✓ Oracle experience collection completed")
        print(f"\nStatistics:")
        print(f"  Episodes: {num_episodes}")
        print(f"  Avg return: {np.mean(stats['episode_returns']):.4f}")
        print(f"  Avg length: {np.mean(stats['episode_lengths']):.1f} steps")
        print(f"  Replay buffer size: {len(op_agent.replay_buffer)}")
        print(f"\nSignal distribution:")
        total_signals = stats['long_signals'] + stats['short_signals'] + stats['neutral_signals']
        print(f"  Long: {stats['long_signals']} ({stats['long_signals']/total_signals*100:.1f}%)")
        print(f"  Short: {stats['short_signals']} ({stats['short_signals']/total_signals*100:.1f}%)")
        print(f"  Neutral: {stats['neutral_signals']} ({stats['neutral_signals']/total_signals*100:.1f}%)")
    

    if save_checkpoint:
        if verbose:
            print(f"\n{'='*60}")
            print("Saving checkpoint...")
            print(f"{'='*60}")

        buffer_path = get_checkpoint_path(
            "phase1.1_oracle", 
            "replay_buffer", 
            checkpoint_dir.split('/')[0]
        ).replace('.pt', '.pkl')
        save_replay_buffer(op_agent.replay_buffer, buffer_path, verbose=verbose)
        

        stats_path = get_checkpoint_path(
            "phase1.1_oracle", 
            "stats", 
            checkpoint_dir.split('/')[0]
        ).replace('.pt', '.json')
        
        stats['metadata'] = {
            'phase': 'phase1.1_oracle',
            'crypto': crypto,
            'num_episodes': num_episodes,
            'replay_buffer_size': len(op_agent.replay_buffer),
            'avg_return': float(np.mean(stats['episode_returns'])),
            'oracle_beta': oracle.beta,
            'oracle_lookforward_window': oracle.lookforward_window,
            'state_dim': state_dim,
            'n_step': training_config.op_n_step, 
            'gamma': training_config.op_gamma,    
            'op_config': {
                'hidden_dims': training_config.op_hidden_dims,
                'learning_rate': training_config.op_learning_rate,
                'gamma': training_config.op_gamma,
                'n_step': training_config.op_n_step,
                'batch_size': training_config.op_batch_size
            }
        }
        save_stats(stats, stats_path, verbose=verbose)
        
        if verbose:
            print(f"\n✓ All checkpoints saved to: {checkpoint_dir}")
            print(f"  - Replay buffer: {buffer_path}")
            print(f"  - Statistics: {stats_path}")
    
    return op_agent, stats


