import datetime
import copy
from typing import Tuple, Optional
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
    save_agent, load_agent, get_checkpoint_path, ensure_checkpoint_dir, save_stats
)


def _select_atm_straddle(options_chain: dict, underlying_price) -> Tuple[Optional[str], Optional[str]]:
    """Select ATM call/put symbols for a straddle."""
    if not options_chain:
        return None, None

    # Find closest strike
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

    return option_action


def _build_hedge_action(state: dict, greeks: tuple, hedger) -> HedgeAction:
    """Build HedgeAction using specified hedger."""
    timestamp = state['timestamp']
    delta, gamma, theta, vega = greeks[0], greeks[1], greeks[2], greeks[3]
    
    # Prepare market_info for PriceMoveHedger
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


class TwinEnvTrainer:
    def __init__(
        self,
        env: RLEnv,
        op_agent: OPAgent,
        hr_agent: HRAgent,
        config: ConfigManager,
        baseline_hedger: Optional[DeltaThresholdHedger] = None
    ):
        """
        Initialize Twin Environment Trainer
        
        Args:
            env: Main environment
            op_agent: OP-Agent
            hr_agent: HR-Agent
            config: Configuration manager
            baseline_hedger: Baseline hedger (default: delta_threshold=0.1)
        """
        self.env = env
        self.op_agent = op_agent
        self.hr_agent = hr_agent
        self.config = config.training_config
        
        # Baseline hedger for twin environment
        if baseline_hedger is None:
            self.baseline_hedger = DeltaThresholdHedger(
                delta_threshold=Decimal('0.1'),
                hedge_ratio=Decimal('1.0')
            )
        else:
            self.baseline_hedger = baseline_hedger
    
    def train_hr_episode(self, start_date=None, end_date=None, verbose=False):
        state, info = self.env.reset(start_date, end_date)
        done = False
        steps = 0
        total_hr_reward = 0
        hr_decisions = 0
        
        self.hr_agent.reset_decision_counter()
        
        while not done:
            if self.hr_agent.should_make_decision():
                saved_state = self.env.env.save_state()
                initial_net_value = self.env.env.get_net_value()
                
                greeks_tuple = state['greeks']
                hr_state_features = self.hr_agent.extract_features(
                    state, info['position'], greeks_tuple
                )
                
                selected_hedger_idx = self.hr_agent.select_hedger(
                    state, info['position'], greeks_tuple,
                    epsilon=self.hr_agent.epsilon
                )
                selected_hedger = self.hr_agent.hedgers[selected_hedger_idx]
                
                temp_state = state
                temp_info = info
                for _ in range(self.hr_agent.n_hr):
                    if done:
                        break
                    
                    action_idx = self.op_agent.select_action(temp_state, epsilon=0.0)
                    direction = self.op_agent.action_to_direction(action_idx)
                    
                    option_action = _build_option_action_from_direction(
                        temp_state, temp_info['position'], direction
                    )
                    hedge_action = _build_hedge_action(
                        temp_state, temp_state['greeks'], selected_hedger
                    )
                    
                    temp_state, _, done, temp_info = self.env.step(option_action, hedge_action)
                    steps += 1
                
                main_net_value = self.env.env.get_net_value()
                
                self.env.env.restore_state(saved_state)
                
                twin_state = state  # Same initial state
                twin_info = info
                twin_done = False
                for _ in range(self.hr_agent.n_hr):
                    if twin_done:
                        break
                    
                    action_idx = self.op_agent.select_action(twin_state, epsilon=0.0)
                    direction = self.op_agent.action_to_direction(action_idx)
                    
                    option_action = _build_option_action_from_direction(
                        twin_state, twin_info['position'], direction
                    )
                    hedge_action = _build_hedge_action(
                        twin_state, twin_state['greeks'], self.baseline_hedger
                    )
                    
                    twin_state, _, twin_done, twin_info = self.env.step(option_action, hedge_action)
                
                twin_net_value = self.env.env.get_net_value()
                
                hr_reward = float(main_net_value - twin_net_value)
                total_hr_reward += hr_reward
                hr_decisions += 1
                
                
                next_hr_state_features = self.hr_agent.extract_features(
                    temp_state, temp_info['position'], temp_state['greeks']
                )
                
                self.hr_agent.replay_buffer.push(
                    hr_state_features,
                    selected_hedger_idx,
                    hr_reward,
                    next_hr_state_features,
                    done
                )
                
                if len(self.hr_agent.replay_buffer) >= self.hr_agent.batch_size:
                    hr_loss = self.hr_agent.update()
                    if verbose and hr_loss is not None:
                        print(f"  HR update - loss: {hr_loss:.4f}, reward: {hr_reward:.4f}")
                
                self.hr_agent.decay_epsilon()
                
                self.env.env.restore_state(saved_state)
                for _ in range(self.hr_agent.n_hr):
                    if done:
                        break
                    action_idx = self.op_agent.select_action(state, epsilon=0.0)
                    direction = self.op_agent.action_to_direction(action_idx)
                    option_action = _build_option_action_from_direction(
                        state, info['position'], direction
                    )
                    hedge_action = _build_hedge_action(
                        state, state['greeks'], selected_hedger
                    )
                    state, reward, done, info = self.env.step(option_action, hedge_action)
                
            else:
                action_idx = self.op_agent.select_action(state, epsilon=0.0)
                direction = self.op_agent.action_to_direction(action_idx)
                option_action = _build_option_action_from_direction(
                    state, info['position'], direction
                )
                
                greeks_tuple = state['greeks']
                _ = self.hr_agent.step(state, info['position'], greeks_tuple)
                current_hedger = self.hr_agent.hedgers[self.hr_agent.current_hedger_idx]
                hedge_action = _build_hedge_action(state, greeks_tuple, current_hedger)
                
                state, reward, done, info = self.env.step(option_action, hedge_action)
                steps += 1
        
        return {
            'steps': steps,
            'hr_decisions': hr_decisions,
            'avg_hr_reward': total_hr_reward / hr_decisions if hr_decisions > 0 else 0,
            'final_net_value': float(self.env.env.get_net_value())
        }
    
    def train_op_episode(self, start_date=None, end_date=None):
        state, info = self.env.reset(start_date, end_date)
        done = False
        steps = 0
        total_reward = 0
        
        self.hr_agent.reset_decision_counter()
        
        while not done:
            greeks_tuple = state['greeks']
            _ = self.hr_agent.step(state, info['position'], greeks_tuple)
            current_hedger = self.hr_agent.hedgers[self.hr_agent.current_hedger_idx]
            
            action_idx = self.op_agent.select_action(state)
            direction = self.op_agent.action_to_direction(action_idx)
            
            option_action = _build_option_action_from_direction(
                state, info['position'], direction
            )
            hedge_action = _build_hedge_action(state, greeks_tuple, current_hedger)
            
            next_state, reward, done, next_info = self.env.step(option_action, hedge_action)
            
            self.op_agent.store_transition(state, action_idx, reward, next_state, done)
            total_reward += reward
            steps += 1
            
            if steps % self.op_agent.update_frequency == 0:
                _ = self.op_agent.update()
            
            if steps % self.op_agent.target_update_frequency == 0:
                self.op_agent.update_target_network()
            
            self.op_agent.decay_epsilon()
            
            state, info = next_state, next_info
        
        return {
            'steps': steps,
            'total_reward': total_reward,
            'final_net_value': float(self.env.env.get_net_value())
        }


def train_iterative(
    crypto: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    config_manager: ConfigManager,
    data_paths: dict,
    op_agent: Optional[OPAgent] = None,
    hr_agent: Optional[HRAgent] = None,
    checkpoint_dir: str = "checkpoints/phase2_iterative",
    save_every_iteration: bool = True,
    save_checkpoint: bool = True,
    load_from_phase1: bool = False,
    phase1_op_checkpoint: str = None,
    phase1_hr_checkpoint: str = None,
    verbose: bool = True
):
    if load_from_phase1:
        if verbose:
            print(f"\n{'='*60}")
            print("Loading agents from Phase 1...")
            print(f"{'='*60}")
        
        if phase1_op_checkpoint is None:
            phase1_op_checkpoint = get_checkpoint_path("phase1.2_op_offline", "op_agent_final")
        if op_agent is not None:
            load_agent(op_agent, phase1_op_checkpoint, verbose=verbose)
        else:
            if verbose:
                print("⚠ OP-Agent not provided, will be initialized below")
        
        if phase1_hr_checkpoint is None:
            phase1_hr_checkpoint = get_checkpoint_path("phase1.3_hr_warmup", "hr_agent_final")
        if hr_agent is not None:
            load_agent(hr_agent, phase1_hr_checkpoint, verbose=verbose)
        else:
            if verbose:
                print("⚠ HR-Agent not provided, will be initialized below")
    
    env_conf = config_manager.env_config
    train_conf = config_manager.training_config
    
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
    
    if op_agent is None or hr_agent is None:
        state, info = env.reset(start_date, end_date)
        feat = np.concatenate([
            state.get('volatility_tickers', np.zeros(48)),
            state.get('features', np.zeros(48))
        ])
        state_dim = len(feat)
        
        if op_agent is None:
            op_agent = OPAgent(
                state_dim=state_dim,
                hidden_dims=train_conf.op_hidden_dims,
                learning_rate=train_conf.op_learning_rate,
                gamma=train_conf.op_gamma,
                n_step=train_conf.op_n_step,
            )
        
        if hr_agent is None:
            hr_state_dim = len(feat) + 2 + 4  # features + position + greeks
            hr_agent = HRAgent(
                state_dim=hr_state_dim,
                learning_rate=train_conf.hr_learning_rate,
                n_hr=train_conf.hr_n_hr,
            )
    
    trainer = TwinEnvTrainer(env, op_agent, hr_agent, config_manager)
    
    if verbose:
        print("=" * 60)
        print("Phase 2: Iterative Training with Twin Environment")
        print("=" * 60)
        print(f"\nThis implements Algorithm 2 from the paper:")
        print(f"  - Alternating training of OP-Agent and HR-Agent")
        print(f"  - HR-Agent uses twin environment for relative reward")
        print(f"  - Baseline hedger: DeltaThresholdHedger(threshold=0.1)")
    
    ensure_checkpoint_dir(checkpoint_dir)
    
    num_iters = train_conf.num_iterations
    op_eps_per_iter = train_conf.op_episodes_per_iter
    hr_eps_per_iter = getattr(train_conf, 'hr_episodes_per_iter', 5)
    
    all_stats = {
        'op_rewards': [],
        'op_values': [],
        'hr_rewards': [],
        'hr_decisions': []
    }
    
    for it in range(num_iters):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Iteration {it+1}/{num_iters}")
            print(f"{'='*60}")
        
        if verbose:
            print(f"\n[OP-Agent Training] {op_eps_per_iter} episodes")
        op_stats = []
        for ep in tqdm(range(op_eps_per_iter), disable=not verbose):
            stats = trainer.train_op_episode()
            op_stats.append(stats)
        
        if verbose:
            avg_reward = np.mean([s['total_reward'] for s in op_stats])
            avg_value = np.mean([s['final_net_value'] for s in op_stats])
            print(f"  OP avg reward: {avg_reward:.4f}, avg final value: {avg_value:.4f}")
        
        all_stats['op_rewards'].append(float(avg_reward))
        all_stats['op_values'].append(float(avg_value))
        
        if verbose:
            print(f"\n[HR-Agent Training] {hr_eps_per_iter} episodes (Twin Env)")
        hr_stats = []
        for ep in tqdm(range(hr_eps_per_iter), disable=not verbose):
            stats = trainer.train_hr_episode(verbose=False)
            hr_stats.append(stats)
        
        if verbose:
            avg_hr_reward = np.mean([s['avg_hr_reward'] for s in hr_stats])
            avg_decisions = np.mean([s['hr_decisions'] for s in hr_stats])
            print(f"  HR avg reward: {avg_hr_reward:.4f}, avg decisions: {avg_decisions:.1f}")
        
        all_stats['hr_rewards'].append(float(avg_hr_reward))
        all_stats['hr_decisions'].append(float(avg_decisions))
        
        if save_every_iteration:
            iter_dir = f"phase2_iterative/iter_{it+1}"
            
            op_path = get_checkpoint_path(iter_dir, "op_agent", checkpoint_dir.split('/')[0])
            hr_path = get_checkpoint_path(iter_dir, "hr_agent", checkpoint_dir.split('/')[0])
            
            op_metadata = {
                'phase': 'phase2_iterative',
                'crypto': crypto,
                'iteration': it + 1,
                'avg_reward': float(avg_reward),
                'avg_value': float(avg_value)
            }
            hr_metadata = {
                'phase': 'phase2_iterative',
                'crypto': crypto,
                'iteration': it + 1,
                'avg_hr_reward': float(avg_hr_reward),
                'avg_decisions': float(avg_decisions)
            }
            
            save_agent(op_agent, op_path, metadata=op_metadata, verbose=False)
            save_agent(hr_agent, hr_path, metadata=hr_metadata, verbose=False)
            
            if verbose:
                print(f"\n  ✓ Checkpoints saved for iteration {it+1}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("✅ Phase 2: Iterative Training Completed!")
        print("=" * 60)
    
    if save_checkpoint:
        if verbose:
            print(f"\n{'='*60}")
            print("Saving final checkpoints...")
            print(f"{'='*60}")
        
        final_op_path = get_checkpoint_path("phase2_iterative/final", "op_agent", checkpoint_dir.split('/')[0])
        final_hr_path = get_checkpoint_path("phase2_iterative/final", "hr_agent", checkpoint_dir.split('/')[0])
        
        op_metadata = {
            'phase': 'phase2_iterative_final',
            'crypto': crypto,
            'total_iterations': num_iters,
            'final_avg_reward': all_stats['op_rewards'][-1] if all_stats['op_rewards'] else None,
            'final_avg_value': all_stats['op_values'][-1] if all_stats['op_values'] else None
        }
        hr_metadata = {
            'phase': 'phase2_iterative_final',
            'crypto': crypto,
            'total_iterations': num_iters,
            'final_avg_hr_reward': all_stats['hr_rewards'][-1] if all_stats['hr_rewards'] else None
        }
        
        save_agent(op_agent, final_op_path, metadata=op_metadata, verbose=verbose)
        save_agent(hr_agent, final_hr_path, metadata=hr_metadata, verbose=verbose)
        
        stats_path = get_checkpoint_path("phase2_iterative/final", "training_stats", checkpoint_dir.split('/')[0]).replace('.pt', '.json')
        save_stats(all_stats, stats_path, verbose=verbose)
    
    return op_agent, hr_agent

