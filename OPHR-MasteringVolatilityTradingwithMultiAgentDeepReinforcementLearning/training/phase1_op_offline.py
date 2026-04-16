import datetime
from typing import Tuple, Optional
from decimal import Decimal

import numpy as np
from tqdm import tqdm

from env.data.data_handler import DataHandler
from env.base_env import BaseEnv
from env.rl_env import RLEnv
from env.base.action import OptionAction, HedgeAction
from agents.op_agent import OPAgent
from hedgers.delta_hedger import DeltaThresholdHedger
from config import ConfigManager
from training.checkpoint_utils import (
    save_agent, load_agent, save_replay_buffer, load_replay_buffer,
    load_stats, get_checkpoint_path, ensure_checkpoint_dir
)


def _select_atm_straddle(options_chain: dict, underlying_price) -> Tuple[Optional[str], Optional[str]]:
    """Select ATM call/put symbols for a straddle."""
    if not options_chain:
        return None, None

    # Find closest strike (and remember expiry)
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


def _build_actions_from_direction(
    state: dict, 
    position, 
    direction: int,
    baseline_hedger: DeltaThresholdHedger
) -> Tuple[OptionAction, HedgeAction]:
    """Build OptionAction and HedgeAction from OP direction and baseline hedger."""
    timestamp = state['timestamp']

    # Option action
    option_action = OptionAction(timestamp=timestamp, trades={})

    if direction == 1:
        # Enter long gamma if no positions
        if len(position.option_positions) == 0:
            options_chain = state.get('options_chain', {})
            if options_chain:
                first_option = next(iter(options_chain.values()))
                call_sym, put_sym = _select_atm_straddle(options_chain, first_option.underlying_price)
                if call_sym and put_sym:
                    option_action.trades[call_sym] = Decimal('1')
                    option_action.trades[put_sym] = Decimal('1')
    elif direction == -1:
        # Enter short gamma if no positions
        if len(position.option_positions) == 0:
            options_chain = state.get('options_chain', {})
            if options_chain:
                first_option = next(iter(options_chain.values()))
                call_sym, put_sym = _select_atm_straddle(options_chain, first_option.underlying_price)
                if call_sym and put_sym:
                    option_action.trades[call_sym] = Decimal('-1')
                    option_action.trades[put_sym] = Decimal('-1')
    else:
        # Neutral: close all positions
        for sym, pos in position.option_positions.items():
            if pos.net_quantity != 0:
                option_action.trades[sym] = -pos.net_quantity

    # Hedge action using baseline hedger
    greeks = state['greeks']
    delta, gamma, theta, vega = greeks[0], greeks[1], greeks[2], greeks[3]
    
    # Prepare market_info for hedgers
    market_info = {}
    options_chain = state.get('options_chain', {})
    if options_chain:
        first_option = next(iter(options_chain.values()))
        market_info['mark_price'] = first_option.underlying_price
    
    hedge_qty = baseline_hedger.compute_hedge(
        delta, gamma, theta, vega,
        position_info={},
        market_info=market_info
    )
    hedge_action = HedgeAction(timestamp=timestamp, quantity=hedge_qty)

    return option_action, hedge_action


def train_op_offline(
    crypto: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    config_manager: ConfigManager,
    data_paths: dict,
    op_agent: Optional[OPAgent] = None,
    baseline_hedger: Optional[DeltaThresholdHedger] = None,
    num_epochs: int = None,
    checkpoint_dir: str = "checkpoints/phase1.2_op_offline",
    save_every_n_epochs: int = 20,
    save_checkpoint: bool = True,
    load_from_phase1_1: bool = False,
    phase1_1_checkpoint: str = None,
    verbose: bool = True
):
    if load_from_phase1_1:
        if verbose:
            print(f"\n{'='*60}")
            print("Loading replay buffer from Phase 1.1 Oracle...")
            print(f"{'='*60}")
        
        stats_path = get_checkpoint_path("phase1.1_oracle", "stats").replace('.pt', '.json')
        phase1_1_stats = load_stats(stats_path, verbose=verbose)
        
        if 'metadata' not in phase1_1_stats:
            raise ValueError("Phase 1.1 stats file missing metadata. Please re-run Phase 1.1.")
        
        metadata = phase1_1_stats['metadata']
        
        # Validate critical hyperparameters for compatibility
        if 'n_step' in metadata:
            expected_n_step = config_manager.training_config.op_n_step
            if metadata['n_step'] != expected_n_step:
                raise ValueError(
                    f"n_step mismatch! Phase 1.1 used n_step={metadata['n_step']}, "
                    f"but current config has n_step={expected_n_step}. "
                    f"The replay buffer contains {metadata['n_step']}-step transitions "
                    f"and cannot be used with a different n_step value."
                )
        
        if 'gamma' in metadata:
            expected_gamma = config_manager.training_config.op_gamma
            if abs(metadata['gamma'] - expected_gamma) > 1e-6:
                raise ValueError(
                    f"gamma mismatch! Phase 1.1 used gamma={metadata['gamma']}, "
                    f"but current config has gamma={expected_gamma}. "
                    f"The replay buffer's n-step rewards were computed with gamma={metadata['gamma']}."
                )
        
        if op_agent is None:
            state_dim = metadata['state_dim']
            op_config = metadata.get('op_config', {})
            
            if verbose:
                print(f"\nCreating new OP-Agent:")
                print(f"  State dim: {state_dim}")
                print(f"  Hidden dims: {op_config.get('hidden_dims', config_manager.training_config.op_hidden_dims)}")
            
            op_agent = OPAgent(
                state_dim=state_dim,
                hidden_dims=op_config.get('hidden_dims', config_manager.training_config.op_hidden_dims),
                learning_rate=op_config.get('learning_rate', config_manager.training_config.op_learning_rate),
                gamma=op_config.get('gamma', config_manager.training_config.op_gamma),
                n_step=op_config.get('n_step', config_manager.training_config.op_n_step),
                replay_buffer_size=config_manager.training_config.op_replay_buffer_size,
                batch_size=op_config.get('batch_size', config_manager.training_config.op_batch_size)
            )
        
        buffer_path = get_checkpoint_path("phase1.1_oracle", "replay_buffer").replace('.pt', '.pkl')
        op_agent.replay_buffer = load_replay_buffer(buffer_path, verbose=verbose)
        
        if verbose:
            print(f"\n✓ Successfully loaded Phase 1.1 Oracle data")
            print(f"  Crypto: {metadata.get('crypto', 'N/A')}")
            print(f"  Episodes: {metadata.get('num_episodes', 'N/A')}")
            print(f"  Avg return: {metadata.get('avg_return', 'N/A'):.4f}")
    
    if op_agent is None:
        raise ValueError("OP-Agent with pre-filled buffer is required for offline training. Set load_from_phase1_1=True or provide op_agent.")
    
    env_conf = config_manager.env_config
    train_conf = config_manager.training_config

    if baseline_hedger is None:
        baseline_hedger = DeltaThresholdHedger(
            delta_threshold=Decimal('0.1'),
            hedge_ratio=Decimal('1.0')
        )

    # Set epsilon to 0 for offline training (no exploration needed)
    op_agent.epsilon = 0.0
    
    if verbose:
        print("=" * 60)
        print("Phase 1.5: OP-Agent Offline Training")
        print("=" * 60)
        print(f"\nBaseline Hedger: DeltaThresholdHedger")
        print(f"  Delta threshold: {baseline_hedger.delta_threshold}")
        print(f"  Hedge ratio: {baseline_hedger.hedge_ratio}")
        print(f"\nReplay buffer size: {len(op_agent.replay_buffer)}")
        print(f"Batch size: {op_agent.batch_size}")
        print(f"Epsilon: {op_agent.epsilon} (offline training, no exploration)")

    if num_epochs is None:
        num_epochs = getattr(train_conf, 'op_offline_epochs', 100)
    
    num_updates_per_epoch = max(1, len(op_agent.replay_buffer) // op_agent.batch_size)
    
    if verbose:
        print(f"\nOffline training for {num_epochs} epochs")
        print(f"Updates per epoch: {num_updates_per_epoch}")
        print(f"Total updates: {num_epochs * num_updates_per_epoch}")
    
    ensure_checkpoint_dir(checkpoint_dir)
    
    losses = []
    for epoch in tqdm(range(num_epochs), disable=not verbose):
        epoch_losses = []
        
        for _ in range(num_updates_per_epoch):
            if len(op_agent.replay_buffer) >= op_agent.batch_size:
                loss = op_agent.update()
                if loss is not None:
                    epoch_losses.append(loss)
        
        if (epoch + 1) % 10 == 0:
            op_agent.update_target_network()
        
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs} - Avg loss: {avg_loss:.6f}")
        
        if save_checkpoint and (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_path = get_checkpoint_path(
                f"phase1.2_op_offline/epoch_{epoch+1}", 
                "op_agent",
                checkpoint_dir.split('/')[0]
            )
            metadata = {
                'phase': 'phase1.2_op_offline',
                'crypto': crypto,
                'epoch': epoch + 1,
                'avg_loss': float(np.mean(losses[-10:])) if losses else None,
                'total_epochs': num_epochs,
                'replay_buffer_size': len(op_agent.replay_buffer)
            }
            save_agent(op_agent, checkpoint_path, metadata=metadata, verbose=False)
            
            buffer_path = checkpoint_path.replace('op_agent.pt', 'replay_buffer.pkl')
            save_replay_buffer(op_agent.replay_buffer, buffer_path, verbose=False)
            
            if verbose:
                print(f"  Checkpoint saved: epoch_{epoch+1}")
    
    if verbose:
        print(f"\n✓ OP-Agent offline training completed")
        print(f"  Final avg loss: {np.mean(losses[-10:]) if losses else 'N/A':.6f}")
        print(f"  Total training updates: {len(losses) * num_updates_per_epoch}")
    
    if save_checkpoint:
        if verbose:
            print(f"\n{'='*60}")
            print("Saving final checkpoint...")
            print(f"{'='*60}")
        
        final_path = get_checkpoint_path("phase1.2_op_offline", "op_agent_final", checkpoint_dir.split('/')[0])
        metadata = {
            'phase': 'phase1.2_op_offline',
            'crypto': crypto,
            'total_epochs': num_epochs,
            'final_avg_loss': float(np.mean(losses[-10:])) if losses else None,
            'replay_buffer_size': len(op_agent.replay_buffer)
        }
        save_agent(op_agent, final_path, metadata=metadata, verbose=verbose)
        
        buffer_path = final_path.replace('op_agent_final.pt', 'replay_buffer_final.pkl')
        save_replay_buffer(op_agent.replay_buffer, buffer_path, verbose=verbose)

    return op_agent


def train_op_online(
    crypto: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    config_manager: ConfigManager,
    data_paths: dict,
    op_agent: Optional[OPAgent] = None,
    baseline_hedger: Optional[DeltaThresholdHedger] = None,
    checkpoint_dir: str = "checkpoints/phase1.2_op_online",
    save_checkpoint: bool = True,
    verbose: bool = True
):
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


    if op_agent is None:
        state, info = env.reset(start_date, end_date)
        feat = np.concatenate([
            state.get('volatility_tickers', np.zeros(48)),
            state.get('features', np.zeros(48))
        ])
        state_dim = len(feat)
        op_agent = OPAgent(
            state_dim=state_dim,
            hidden_dims=train_conf.op_hidden_dims,
            learning_rate=train_conf.op_learning_rate,
            gamma=train_conf.op_gamma,
            n_step=train_conf.op_n_step,
            replay_buffer_size=train_conf.op_replay_buffer_size,
            batch_size=train_conf.op_batch_size,
            epsilon_start=train_conf.op_epsilon_start,
            epsilon_end=train_conf.op_epsilon_end,
            epsilon_decay=train_conf.op_epsilon_decay,
            update_frequency=train_conf.op_update_frequency,
            target_update_frequency=train_conf.op_target_update_frequency,
        )


    if baseline_hedger is None:
        baseline_hedger = DeltaThresholdHedger(
            delta_threshold=Decimal('0.1'),
            hedge_ratio=Decimal('1.0')
        )

    if verbose:
        print("=" * 60)
        print("Phase 1.5: OP-Agent Online Training (with Baseline Hedger)")
        print("=" * 60)
        print(f"\nBaseline Hedger: DeltaThresholdHedger")
        print(f"  Delta threshold: {baseline_hedger.delta_threshold}")

    num_episodes = getattr(train_conf, 'op_online_episodes', train_conf.op_episodes_per_iter)

    for episode in tqdm(range(num_episodes), disable=not verbose):
        state, info = env.reset()
        done = False
        steps = 0
        episode_return = 0

        while not done:
            # OP action
            action_idx = op_agent.select_action(state)
            direction = op_agent.action_to_direction(action_idx)

            # Build concrete actions with baseline hedger
            option_action, hedge_action = _build_actions_from_direction(
                state, info['position'], direction, baseline_hedger
            )

            # Step env
            next_state, reward, done, next_info = env.step(option_action, hedge_action)

            # Store and update
            op_agent.store_transition(state, action_idx, reward, next_state, done)
            episode_return += reward
            steps += 1

            if steps % op_agent.update_frequency == 0:
                _ = op_agent.update()

            if steps % op_agent.target_update_frequency == 0:
                op_agent.update_target_network()

            op_agent.decay_epsilon()

            state, info = next_state, next_info

    if verbose:
        print("✓ OP-Agent online training completed")
    
    if save_checkpoint:
        if verbose:
            print(f"\n{'='*60}")
            print("Saving final checkpoint...")
            print(f"{'='*60}")
        
        final_path = get_checkpoint_path("phase1.2_op_online", "op_agent_final", checkpoint_dir.split('/')[0])
        metadata = {
            'phase': 'phase1.2_op_online',
            'crypto': crypto,
            'num_episodes': num_episodes,
            'replay_buffer_size': len(op_agent.replay_buffer)
        }
        save_agent(op_agent, final_path, metadata=metadata, verbose=verbose)

    return op_agent

