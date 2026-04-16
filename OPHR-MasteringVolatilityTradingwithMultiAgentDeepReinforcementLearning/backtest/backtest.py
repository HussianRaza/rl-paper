import datetime
import os
from typing import Optional, Dict, Any, Tuple
from decimal import Decimal

import numpy as np
import pandas as pd
from tqdm import tqdm

from env.data.data_handler import DataHandler
from env.base_env import BaseEnv
from env.rl_env import RLEnv
from env.base.action import OptionAction, HedgeAction
from agents.op_agent import OPAgent
from agents.hr_agent import HRAgent
from hedgers.delta_hedger import DeltaThresholdHedger
from hedgers.base_hedger import BaseHedger
from config import ConfigManager
from training.checkpoint_utils import load_agent


def _select_atm_straddle(options_chain: dict, underlying_price) -> Tuple[Optional[str], Optional[str]]:
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


class BacktestRunner:   
    def __init__(
        self,
        crypto: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        config_manager: ConfigManager,
        data_paths: dict,
        op_agent: Optional[OPAgent] = None,
        hr_agent: Optional[HRAgent] = None,
        baseline_hedger: Optional[BaseHedger] = None,
        mode: str = 'full',  # 'full', 'op_only', 'baseline'
        verbose: bool = True
    ):
        self.crypto = crypto
        self.start_date = start_date
        self.end_date = end_date
        self.config_manager = config_manager
        self.data_paths = data_paths
        self.op_agent = op_agent
        self.hr_agent = hr_agent
        self.mode = mode
        self.verbose = verbose
        
        # Initialize baseline hedger
        if baseline_hedger is None:
            self.baseline_hedger = DeltaThresholdHedger(
                delta_threshold=Decimal('0.1'),
                hedge_ratio=Decimal('1.0')
            )
        else:
            self.baseline_hedger = baseline_hedger
        
        # Validate mode
        if mode == 'full' and (op_agent is None or hr_agent is None):
            raise ValueError("Both OP-Agent and HR-Agent required for 'full' mode")
        if mode == 'op_only' and op_agent is None:
            raise ValueError("OP-Agent required for 'op_only' mode")
        
        # Build environment
        env_config = config_manager.env_config
        data_handler = DataHandler(
            start_date, end_date, crypto,
            data_paths['option_chain'],
            data_paths['perpetual'],
            volatility_ticker_path=data_paths.get('volatility_ticker')
        )
        
        time_config = {
            'episode_length': env_config.episode_length,
            'option_interval': env_config.option_interval,
            'include_volatility_tickers': True,
        }
        
        base_env = BaseEnv(data_handler, time_config, crypto=crypto)
        self.env = RLEnv(base_env)
        
        # Set agents to evaluation mode
        if self.op_agent is not None:
            self.op_agent.q_network.eval()
        if self.hr_agent is not None:
            self.hr_agent.q_network.eval()
        
        if self.verbose:
            print("=" * 60)
            print(f"Backtest Runner Initialized")
            print("=" * 60)
            print(f"Mode: {mode}")
            print(f"Crypto: {crypto}")
            print(f"Period: {start_date.date()} to {end_date.date()}")
            print(f"OP-Agent: {'Loaded' if op_agent else 'None'}")
            print(f"HR-Agent: {'Loaded' if hr_agent else 'None'}")
            print(f"Baseline Hedger: {type(self.baseline_hedger).__name__}")
    
    def run_episode(self) -> Dict[str, Any]:

        state, info = self.env.reset(self.start_date, self.end_date)
        done = False
        steps = 0
        
        # Tracking
        results = {
            'timestamps': [],
            'net_values': [],
            'rewards': [],
            'actions': [],  # long/neutral/short
            'hedger_selections': [],  # which hedger selected
            'option_positions': [],  # number of option positions
            'perp_positions': [],  # perpetual positions
            'greeks': {
                'delta': [],
                'gamma': [],
                'theta': [],
                'vega': []
            },
            'underlying_prices': []
        }
        
        if self.mode in ['full', 'op_only'] and self.hr_agent is not None:
            self.hr_agent.reset_decision_counter()
        
        with tqdm(total=None, disable=not self.verbose, desc="Backtest") as pbar:
            while not done:
                # Get current info
                timestamp = state['timestamp']
                greeks = state['greeks']
                position = info['position']
                net_value = info['account'].net_value
                
                # Get underlying price
                options_chain = state.get('options_chain', {})
                underlying_price = None
                if options_chain:
                    first_option = next(iter(options_chain.values()))
                    underlying_price = float(first_option.underlying_price)
                
                # Determine actions based on mode
                if self.mode == 'baseline':
                    # Baseline: simple buy-and-hold or rule-based
                    direction = 0  # Neutral for baseline
                    option_action = _build_option_action_from_direction(state, position, direction)
                    hedge_action = _build_hedge_action(state, greeks, self.baseline_hedger)
                    hedger_name = 'baseline'
                
                elif self.mode == 'op_only':
                    # OP-Agent + baseline hedger
                    action_idx = self.op_agent.select_action(state, epsilon=0.0)  # No exploration
                    direction = self.op_agent.action_to_direction(action_idx)
                    option_action = _build_option_action_from_direction(state, position, direction)
                    hedge_action = _build_hedge_action(state, greeks, self.baseline_hedger)
                    hedger_name = 'baseline'
                
                elif self.mode == 'full':
                    # OP-Agent + HR-Agent
                    action_idx = self.op_agent.select_action(state, epsilon=0.0)
                    direction = self.op_agent.action_to_direction(action_idx)
                    option_action = _build_option_action_from_direction(state, position, direction)
                    
                    # HR agent selection
                    _ = self.hr_agent.step(state, position, greeks)
                    current_hedger = self.hr_agent.hedgers[self.hr_agent.current_hedger_idx]
                    hedge_action = _build_hedge_action(state, greeks, current_hedger)
                    hedger_name = f"hedger_{self.hr_agent.current_hedger_idx}"
                
                # Step environment
                next_state, reward, done, next_info = self.env.step(option_action, hedge_action)
                
                # Record results
                results['timestamps'].append(timestamp)
                results['net_values'].append(float(net_value))
                results['rewards'].append(float(reward))
                results['actions'].append(direction)
                results['hedger_selections'].append(hedger_name)
                results['option_positions'].append(len(position.option_positions))
                results['perp_positions'].append(float(position.perpetual_position.net_quantity))
                results['greeks']['delta'].append(float(greeks[0]))
                results['greeks']['gamma'].append(float(greeks[1]))
                results['greeks']['theta'].append(float(greeks[2]))
                results['greeks']['vega'].append(float(greeks[3]))
                results['underlying_prices'].append(underlying_price if underlying_price else np.nan)
                
                state = next_state
                info = next_info
                steps += 1
                pbar.update(1)
        
        # Add summary statistics
        results['total_steps'] = steps
        results['initial_value'] = results['net_values'][0] if results['net_values'] else 0
        results['final_value'] = results['net_values'][-1] if results['net_values'] else 0
        results['total_return'] = results['final_value'] - results['initial_value']
        results['return_pct'] = (results['total_return'] / results['initial_value'] * 100) if results['initial_value'] != 0 else 0
        
        return results
    
    def calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:

        net_values = np.array(results['net_values'])
        rewards = np.array(results['rewards'])
        
        # Returns
        returns = np.diff(net_values)
        returns_pct = returns / net_values[:-1] * 100
        
        # Basic metrics
        metrics = {
            'total_return': results['total_return'],
            'return_pct': results['return_pct'],
            'final_value': results['final_value'],
            'initial_value': results['initial_value'],
            'total_steps': results['total_steps']
        }
        
        # Risk metrics
        if len(returns_pct) > 0:
            metrics['volatility'] = np.std(returns_pct)
            metrics['sharpe_ratio'] = np.mean(returns_pct) / np.std(returns_pct) if np.std(returns_pct) > 0 else 0
            
            # Maximum drawdown
            cummax = np.maximum.accumulate(net_values)
            drawdowns = (net_values - cummax) / cummax * 100
            metrics['max_drawdown'] = np.min(drawdowns)
            
            # Calmar ratio
            avg_return = np.mean(returns_pct)
            metrics['calmar_ratio'] = avg_return / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns_pct[returns_pct < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                metrics['sortino_ratio'] = avg_return / downside_std if downside_std > 0 else 0
            else:
                metrics['sortino_ratio'] = float('inf') if avg_return > 0 else 0
        
        # Position metrics
        actions = np.array(results['actions'])
        metrics['long_ratio'] = np.sum(actions == 1) / len(actions) * 100 if len(actions) > 0 else 0
        metrics['short_ratio'] = np.sum(actions == -1) / len(actions) * 100 if len(actions) > 0 else 0
        metrics['neutral_ratio'] = np.sum(actions == 0) / len(actions) * 100 if len(actions) > 0 else 0
        
        # Average position
        option_positions = np.array(results['option_positions'])
        metrics['avg_option_positions'] = np.mean(option_positions) if len(option_positions) > 0 else 0
        
        return metrics
    
    def generate_report(self, results: Dict[str, Any], save_dir: Optional[str] = None) -> pd.DataFrame:

        metrics = self.calculate_metrics(results)
        
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"Backtest Report - {self.mode.upper()} Mode")
            print("=" * 60)
            print(f"\n📊 Returns:")
            print(f"  Total Return: ${metrics['total_return']:.2f}")
            print(f"  Return %: {metrics['return_pct']:.2f}%")
            print(f"  Initial Value: ${metrics['initial_value']:.2f}")
            print(f"  Final Value: ${metrics['final_value']:.2f}")
            
            print(f"\n📈 Risk Metrics:")
            print(f"  Volatility: {metrics['volatility']:.4f}%")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
            print(f"  Calmar Ratio: {metrics['calmar_ratio']:.4f}")
            print(f"  Sortino Ratio: {metrics['sortino_ratio']:.4f}")
            
            print(f"\n🎯 Position Distribution:")
            print(f"  Long: {metrics['long_ratio']:.1f}%")
            print(f"  Short: {metrics['short_ratio']:.1f}%")
            print(f"  Neutral: {metrics['neutral_ratio']:.1f}%")
            print(f"  Avg Positions: {metrics['avg_option_positions']:.2f}")
            
            print(f"\n⏱️  Duration:")
            print(f"  Total Steps: {metrics['total_steps']}")
        
        # Create DataFrame
        df_metrics = pd.DataFrame([metrics])
        
        # Save if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save metrics
            metrics_path = os.path.join(save_dir, f'metrics_{self.mode}.csv')
            df_metrics.to_csv(metrics_path, index=False)
            
            # Save detailed results
            df_results = pd.DataFrame({
                'timestamp': results['timestamps'],
                'net_value': results['net_values'],
                'reward': results['rewards'],
                'action': results['actions'],
                'hedger': results['hedger_selections'],
                'option_positions': results['option_positions'],
                'perp_position': results['perp_positions'],
                'delta': results['greeks']['delta'],
                'gamma': results['greeks']['gamma'],
                'theta': results['greeks']['theta'],
                'vega': results['greeks']['vega'],
                'underlying_price': results['underlying_prices']
            })
            results_path = os.path.join(save_dir, f'results_{self.mode}.csv')
            df_results.to_csv(results_path, index=False)
            
            if self.verbose:
                print(f"\n💾 Report saved to:")
                print(f"  Metrics: {metrics_path}")
                print(f"  Detailed: {results_path}")
        
        return df_metrics


def run_backtest(
    crypto: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    config_manager: ConfigManager,
    data_paths: dict,
    op_checkpoint: Optional[str] = None,
    hr_checkpoint: Optional[str] = None,
    mode: str = 'full',
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    op_agent = None
    hr_agent = None
    
    if mode in ['full', 'op_only'] and op_checkpoint:
        # Load OP-Agent
        env_config = config_manager.env_config
        train_config = config_manager.training_config
        
        # Dummy initialization to get state_dim
        data_handler = DataHandler(
            start_date, end_date, crypto,
            data_paths['option_chain'],
            data_paths['perpetual'],
            volatility_ticker_path=data_paths.get('volatility_ticker')
        )
        time_config = {
            'episode_length': env_config.episode_length,
            'option_interval': env_config.option_interval,
            'include_volatility_tickers': True,
        }
        base_env = BaseEnv(data_handler, time_config, crypto=crypto)
        env = RLEnv(base_env)
        state, _ = env.reset(start_date, end_date)
        feat = np.concatenate([
            state.get('volatility_tickers', np.zeros(48)),
            state.get('features', np.zeros(48))
        ])
        state_dim = len(feat)
        
        op_agent = OPAgent(
            state_dim=state_dim,
            hidden_dims=train_config.op_hidden_dims,
            learning_rate=train_config.op_learning_rate,
            gamma=train_config.op_gamma,
            n_step=train_config.op_n_step,
        )
        load_agent(op_agent, op_checkpoint, verbose=verbose)
    
    if mode == 'full' and hr_checkpoint:
        # Load HR-Agent
        hr_state_dim = state_dim + 2 + 4  # features + position + greeks
        hr_agent = HRAgent(
            state_dim=hr_state_dim,
            learning_rate=train_config.hr_learning_rate,
            n_hr=train_config.hr_n_hr,
        )
        load_agent(hr_agent, hr_checkpoint, verbose=verbose)
    
    # Run backtest
    runner = BacktestRunner(
        crypto=crypto,
        start_date=start_date,
        end_date=end_date,
        config_manager=config_manager,
        data_paths=data_paths,
        op_agent=op_agent,
        hr_agent=hr_agent,
        mode=mode,
        verbose=verbose
    )
    
    results = runner.run_episode()
    metrics_df = runner.generate_report(results, save_dir=save_dir)
    
    return results, metrics_df

