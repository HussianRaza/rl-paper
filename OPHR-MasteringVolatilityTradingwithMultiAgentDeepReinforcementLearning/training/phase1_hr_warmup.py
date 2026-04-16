import datetime
from typing import Optional
from decimal import Decimal

import numpy as np
from tqdm import tqdm

from env.data.data_handler import DataHandler
from env.base_env import BaseEnv
from env.rl_env import RLEnv
from env.base.action import OptionAction, HedgeAction
from agents.op_agent import OPAgent
from agents.hr_agent import HRAgent
from hedgers.delta_hedger import DeltaThresholdHedger
from config import ConfigManager
from training.checkpoint_utils import (
    save_agent, load_agent, get_checkpoint_path, ensure_checkpoint_dir
)


def _select_atm_straddle(options_chain: dict, underlying_price):
    """Select ATM call/put symbols for a straddle."""
    if not options_chain:
        return None, None

    min_diff = float('inf')
    best_strike = None
    expiry = None

    up = float(underlying_price)
    for symbol, option in options_chain.items():
        diff = abs(float(option.strike_price) - up)
        if diff < min_diff:
            min_diff = diff
            best_strike = option.strike_price
            expiry = option.expiration

    if best_strike is None:
        return None, None

    best_call, best_put = None, None
    for symbol, option in options_chain.items():
        if option.strike_price == best_strike and option.expiration == expiry:
            if ('-C' in symbol) or (hasattr(option, 'type') and str(option.type) == 'OptionTypes.CALL'):
                best_call = symbol
            elif ('-P' in symbol) or (hasattr(option, 'type') and str(option.type) == 'OptionTypes.PUT'):
                best_put = symbol

    return best_call, best_put


def _build_option_action_from_direction(state: dict, position, direction: int) -> OptionAction:
    """Build OptionAction from OP direction."""
    timestamp = state['timestamp']
    option_action = OptionAction(timestamp=timestamp, trades={})

    if direction == 1:
        if len(position.option_positions) == 0:
            options_chain = state.get('options_chain', {})
            if options_chain:
                first_option = next(iter(options_chain.values()))
                call_sym, put_sym = _select_atm_straddle(options_chain, first_option.underlying_price)
                if call_sym and put_sym:
                    option_action.trades[call_sym] = Decimal('1')
                    option_action.trades[put_sym] = Decimal('1')
    elif direction == -1:
        if len(position.option_positions) == 0:
            options_chain = state.get('options_chain', {})
            if options_chain:
                first_option = next(iter(options_chain.values()))
                call_sym, put_sym = _select_atm_straddle(options_chain, first_option.underlying_price)
                if call_sym and put_sym:
                    option_action.trades[call_sym] = Decimal('-1')
                    option_action.trades[put_sym] = Decimal('-1')
    else:
        for sym, pos in position.option_positions.items():
            if pos.net_quantity != 0:
                option_action.trades[sym] = -pos.net_quantity

    return option_action


def _build_hedge_action(state: dict, greeks: tuple, hedger) -> HedgeAction:
    """Build HedgeAction using specified hedger."""
    timestamp = state['timestamp']
    delta, gamma, theta, vega = greeks[0], greeks[1], greeks[2], greeks[3]
    
    market_info = {}
    options_chain = state.get('options_chain', {})
    if options_chain:
        first_option = next(iter(options_chain.values()))
        market_info['mark_price'] = first_option.underlying_price
    
    hedge_qty = hedger.compute_hedge(
        delta, gamma, theta, vega,
        position_info={},
        market_info=market_info
    )
    
    return HedgeAction(timestamp=timestamp, quantity=hedge_qty)


def warmup_hr_agent(
    crypto: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    config_manager: ConfigManager,
    data_paths: dict,
    op_agent: OPAgent = None,
    hr_agent: Optional[HRAgent] = None,
    baseline_hedger: Optional[DeltaThresholdHedger] = None,
    num_episodes: int = None,
    checkpoint_dir: str = "checkpoints/phase1.3_hr_warmup",
    save_every_n_episodes: int = 10,
    save_checkpoint: bool = True,
    load_op_from_phase1_2: bool = False,
    phase1_2_checkpoint: str = None,
    verbose: bool = True
):
    if load_op_from_phase1_2:
        if phase1_2_checkpoint is None:
            phase1_2_checkpoint = get_checkpoint_path("phase1.2_op_offline", "op_agent_final")
        
        if verbose:
            print(f"\n{'='*60}")
            print("Loading OP-Agent from Phase 1.2...")
            print(f"{'='*60}")
        
        if op_agent is None:
            raise ValueError("op_agent must be initialized before loading checkpoint")
        
        load_agent(op_agent, phase1_2_checkpoint, verbose=verbose)
    
    if op_agent is None:
        raise ValueError("Trained OP-Agent is required for HR warm-up training")
    
    env_conf = config_manager.env_config
    train_conf = config_manager.training_config
    
    # Build environment
    data_handler = DataHandler(
        start_date, end_date, crypto,
        data_paths['option_chain'],
        data_paths['perpetual'],
        volatility_ticker_path=data_paths.get('volatility_ticker')
    )
    
    time_config = {
        'episode_length': env_conf.episode_length,
        'option_interval': env_conf.option_interval,
        'include_volatility_tickers': True,
    }
    
    base_env = BaseEnv(data_handler, time_config, crypto=crypto)
    env = RLEnv(base_env)
    
    # Initialize HR-Agent if needed
    if hr_agent is None:
        state, info = env.reset(start_date, end_date)
        feat = np.concatenate([
            state.get('volatility_tickers', np.zeros(48)),
            state.get('features', np.zeros(48))
        ])
        hr_state_dim = len(feat) + 2 + 4  # features + position + greeks
        hr_agent = HRAgent(
            state_dim=hr_state_dim,
            hidden_dims=getattr(train_conf, 'hr_hidden_dims', [128, 128]),
            learning_rate=train_conf.hr_learning_rate,
            gamma=getattr(train_conf, 'hr_gamma', train_conf.op_gamma),
            replay_buffer_size=getattr(train_conf, 'hr_replay_buffer_size', 50000),
            batch_size=train_conf.hr_batch_size,
            epsilon_start=getattr(train_conf, 'hr_epsilon_start', 1.0),
            epsilon_end=getattr(train_conf, 'hr_epsilon_end', 0.05),
            epsilon_decay=getattr(train_conf, 'hr_epsilon_decay', 0.995),
            n_hr=train_conf.hr_n_hr,
            update_frequency=getattr(train_conf, 'hr_update_frequency', 1),
            target_update_frequency=getattr(train_conf, 'hr_target_update_frequency', 100),
        )
    
    # Initialize baseline hedger
    if baseline_hedger is None:
        baseline_hedger = DeltaThresholdHedger(
            delta_threshold=Decimal('0.1'),
            hedge_ratio=Decimal('1.0')
        )
    
    if verbose:
        print("=" * 60)
        print("Phase 1.3: HR-Agent Warm-up Training")
        print("=" * 60)
        print(f"\nOP-Agent: FROZEN (used for option decisions, no updates)")
        print(f"HR-Agent: TRAINING (learns to select hedgers)")
        print(f"Baseline Hedger: DeltaThresholdHedger(threshold=0.1)")
        print(f"\nHR-Agent config:")
        print(f"  State dim: {hr_agent.state_dim}")
        print(f"  n_hr: {hr_agent.n_hr}")
        print(f"  Num hedgers: {len(hr_agent.hedgers)}")
    
    # Warm-up training
    if num_episodes is None:
        num_episodes = getattr(train_conf, 'hr_warmup_episodes', 50)
    
    if verbose:
        print(f"\nWarm-up training for {num_episodes} episodes...")
    
    # Ensure checkpoint directory exists
    ensure_checkpoint_dir(checkpoint_dir)
    
    stats = {
        'hr_rewards': [],
        'hr_decisions': [],
        'losses': []
    }
    
    for episode in tqdm(range(num_episodes), disable=not verbose):
        state, info = env.reset()
        done = False
        steps = 0
        episode_hr_reward = 0
        episode_hr_decisions = 0
        
        hr_agent.reset_decision_counter()
        
        while not done:
            # HR-Agent decision every n_hr steps
            if hr_agent.should_make_decision():
                # Save state for twin environment
                saved_state = env.env.save_state()
                
                # Get HR state
                greeks_tuple = state['greeks']
                hr_state_features = hr_agent.extract_features(
                    state, info['position'], greeks_tuple
                )
                
                # Select hedger (with exploration)
                selected_hedger_idx = hr_agent.select_hedger(
                    state, info['position'], greeks_tuple,
                    epsilon=hr_agent.epsilon
                )
                selected_hedger = hr_agent.hedgers[selected_hedger_idx]
                
                # Execute n_hr steps in MAIN environment
                temp_state = state
                temp_info = info
                temp_done = done
                for _ in range(hr_agent.n_hr):
                    if temp_done:
                        break
                    
                    # OP action (frozen, no exploration)
                    action_idx = op_agent.select_action(temp_state, epsilon=0.0)
                    direction = op_agent.action_to_direction(action_idx)
                    
                    option_action = _build_option_action_from_direction(
                        temp_state, temp_info['position'], direction
                    )
                    hedge_action = _build_hedge_action(
                        temp_state, temp_state['greeks'], selected_hedger
                    )
                    
                    temp_state, _, temp_done, temp_info = env.step(option_action, hedge_action)
                    steps += 1
                
                main_net_value = env.env.get_net_value()
                
                # Restore and execute in TWIN environment
                env.env.restore_state(saved_state)
                
                twin_state = state
                twin_info = info
                twin_done = done
                for _ in range(hr_agent.n_hr):
                    if twin_done:
                        break
                    
                    action_idx = op_agent.select_action(twin_state, epsilon=0.0)
                    direction = op_agent.action_to_direction(action_idx)
                    
                    option_action = _build_option_action_from_direction(
                        twin_state, twin_info['position'], direction
                    )
                    hedge_action = _build_hedge_action(
                        twin_state, twin_state['greeks'], baseline_hedger
                    )
                    
                    twin_state, _, twin_done, twin_info = env.step(option_action, hedge_action)
                
                twin_net_value = env.env.get_net_value()
                
                # Calculate relative reward
                hr_reward = float(main_net_value - twin_net_value)
                episode_hr_reward += hr_reward
                episode_hr_decisions += 1
                
                # Get next HR state
                next_hr_state_features = hr_agent.extract_features(
                    temp_state, temp_info['position'], temp_state['greeks']
                )
                
                # Store transition
                hr_agent.replay_buffer.push(
                    hr_state_features,
                    selected_hedger_idx,
                    hr_reward,
                    next_hr_state_features,
                    temp_done
                )
                
                # Update HR-Agent
                if len(hr_agent.replay_buffer) >= hr_agent.batch_size:
                    if steps % hr_agent.update_frequency == 0:
                        loss = hr_agent.update()
                        if loss is not None:
                            stats['losses'].append(loss)
                
                # Update target network
                if steps % hr_agent.target_update_frequency == 0:
                    hr_agent.update_target_network()
                
                # Decay epsilon
                hr_agent.decay_epsilon()
                
                # Continue from main environment
                env.env.restore_state(saved_state)
                for _ in range(hr_agent.n_hr):
                    if done:
                        break
                    action_idx = op_agent.select_action(state, epsilon=0.0)
                    direction = op_agent.action_to_direction(action_idx)
                    option_action = _build_option_action_from_direction(
                        state, info['position'], direction
                    )
                    hedge_action = _build_hedge_action(
                        state, state['greeks'], selected_hedger
                    )
                    state, reward, done, info = env.step(option_action, hedge_action)
            
            else:
                # Not HR decision step
                action_idx = op_agent.select_action(state, epsilon=0.0)
                direction = op_agent.action_to_direction(action_idx)
                option_action = _build_option_action_from_direction(
                    state, info['position'], direction
                )
                
                greeks_tuple = state['greeks']
                _ = hr_agent.step(state, info['position'], greeks_tuple)
                current_hedger = hr_agent.hedgers[hr_agent.current_hedger_idx]
                hedge_action = _build_hedge_action(state, greeks_tuple, current_hedger)
                
                state, reward, done, info = env.step(option_action, hedge_action)
                steps += 1
        
        stats['hr_rewards'].append(episode_hr_reward / max(1, episode_hr_decisions))
        stats['hr_decisions'].append(episode_hr_decisions)
        
        # Save checkpoint every N episodes
        if save_checkpoint and (episode + 1) % save_every_n_episodes == 0:
            checkpoint_path = get_checkpoint_path(
                f"phase1.3_hr_warmup/episode_{episode+1}", 
                "hr_agent",
                checkpoint_dir.split('/')[0]
            )
            metadata = {
                'phase': 'phase1.3_hr_warmup',
                'crypto': crypto,
                'episode': episode + 1,
                'avg_hr_reward': float(np.mean(stats['hr_rewards'][-10:])) if stats['hr_rewards'] else None,
                'total_episodes': num_episodes
            }
            save_agent(hr_agent, checkpoint_path, metadata=metadata, verbose=False)
            if verbose:
                print(f"  Checkpoint saved: episode_{episode+1}")
    
    if verbose:
        print(f"\n✓ HR-Agent warm-up training completed")
        print(f"\nStatistics:")
        print(f"  Episodes: {num_episodes}")
        print(f"  Avg HR reward: {np.mean(stats['hr_rewards']):.4f}")
        print(f"  Avg decisions per episode: {np.mean(stats['hr_decisions']):.1f}")
        print(f"  Replay buffer size: {len(hr_agent.replay_buffer)}")
        print(f"  Final epsilon: {hr_agent.epsilon:.4f}")
        if stats['losses']:
            print(f"  Avg loss: {np.mean(stats['losses'][-100:]):.6f}")
    
    # Save final checkpoint
    if save_checkpoint:
        if verbose:
            print(f"\n{'='*60}")
            print("Saving final checkpoint...")
            print(f"{'='*60}")
        
        final_path = get_checkpoint_path("phase1.3_hr_warmup", "hr_agent_final", checkpoint_dir.split('/')[0])
        metadata = {
            'phase': 'phase1.3_hr_warmup',
            'crypto': crypto,
            'total_episodes': num_episodes,
            'avg_hr_reward': float(np.mean(stats['hr_rewards'])) if stats['hr_rewards'] else None,
            'final_epsilon': float(hr_agent.epsilon),
            'replay_buffer_size': len(hr_agent.replay_buffer)
        }
        save_agent(hr_agent, final_path, metadata=metadata, verbose=verbose)
    
    return hr_agent


