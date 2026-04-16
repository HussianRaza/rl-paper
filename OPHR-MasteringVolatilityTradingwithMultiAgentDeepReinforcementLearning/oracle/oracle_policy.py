"""
Oracle Policy Implementation

Oracle generates long/short signals by comparing future RV with current IV:
- If future RV >= IV * (1 + β), place long position
- If future RV <= IV * (1 - β), place short position  
- Otherwise, neutral position
"""

import numpy as np
from typing import Tuple, Optional
from decimal import Decimal

from env.base.action import OptionAction, HedgeAction
from hedgers.delta_hedger import DeltaThresholdHedger


class OraclePolicy:

    def __init__(
        self,
        beta: float = 0.1,
        lookforward_window: int = 168, 
        delta_threshold: float = 0.1
    ):

        self.beta = beta
        self.lookforward_window = lookforward_window
        self.hedger = DeltaThresholdHedger(delta_threshold=delta_threshold)
        self.step_count = 0
    
    def calculate_future_rv(
        self,
        state: dict,
        info: dict,
        window: Optional[int] = None
    ) -> float:

        if window is None:
            window = self.lookforward_window

        future_info = info.get('future_info', {})
        
        if not future_info:
            return 0.5
        
        if window <= 3:
            key = 'volatility_next_3h'
        elif window <= 6:
            key = 'volatility_next_6h'
        elif window <= 9:
            key = 'volatility_next_9h'
        elif window <= 12:
            key = 'volatility_next_12h'
        elif window <= 18:
            key = 'volatility_next_18h'
        else:
            key = 'volatility_next_24h'
        
        future_rv = future_info.get(key, 0.0)
        
        try:
            future_rv = float(future_rv)
        except:
            future_rv = 0.5  

        if future_rv <= 0 or future_rv > 10:
            future_rv = 0.5  
        
        return future_rv
    
    def get_current_iv(self, state: dict) -> float:
        options_chain = state.get('options_chain', {})
        
        if options_chain:
            min_diff = float('inf')
            atm_iv = None
            
            first_option = next(iter(options_chain.values()))
            underlying_price = float(first_option.underlying_price)
            
            for symbol, option in options_chain.items():
                strike_diff = abs(float(option.strike_price) - underlying_price)
                
                if strike_diff < min_diff:
                    min_diff = strike_diff
                    if hasattr(option, 'mark_iv') and option.mark_iv > 0:
                        atm_iv = float(option.mark_iv)
            
            if atm_iv is not None and atm_iv > 0:
                return atm_iv
        
        vola_features = state.get('volatility_tickers', np.array([]))
        
        if len(vola_features) > 0 and vola_features[0] > 0:
            return float(vola_features[0])

        return 0.5
    
    def generate_signal(self, state: dict, info: dict) -> int:

        future_rv = self.calculate_future_rv(state, info)
        current_iv = self.get_current_iv(state)
        
        if future_rv >= current_iv * (1 + self.beta):
            return 1  # Long
        elif future_rv <= current_iv * (1 - self.beta):
            return -1  # Short
        else:
            return 0  # Neutral
    
    def select_atm_straddle(
        self,
        options_chain: dict,
        underlying_price: Decimal
    ) -> Tuple[Optional[str], Optional[str]]:

        if not options_chain:
            return None, None
        
        min_diff = float('inf')
        best_call = None
        best_put = None
        
        for symbol, option in options_chain.items():
            strike_diff = abs(option.strike_price - underlying_price)
            
            if strike_diff < min_diff:
                min_diff = strike_diff
                
                if 'C' in symbol or (hasattr(option, 'option_type') and option.option_type == 'call'):
                    best_call = symbol
                    put_symbol = symbol.replace('-C', '-P')
                    if put_symbol in options_chain:
                        best_put = put_symbol
        
        return best_call, best_put
    
    def step(
        self,
        state: dict,
        account,
        position,
        info: dict
    ) -> Tuple[OptionAction, HedgeAction]:

        timestamp = state['timestamp']
        
        direction = self.generate_signal(state, info)
        
        option_action = OptionAction(timestamp=timestamp, trades={})
        
        if direction != 0:
            if len(position.option_positions) == 0:
                options_chain = state.get('options_chain', {})
                if options_chain:
                    first_option = next(iter(options_chain.values()))
                    underlying_price = first_option.underlying_price
                    
                    call_symbol, put_symbol = self.select_atm_straddle(
                        options_chain, underlying_price
                    )
                    
                    if call_symbol and put_symbol:
                        # Buy straddle for long, sell for short
                        quantity = Decimal('1') if direction == 1 else Decimal('-1')
                        option_action.trades[call_symbol] = quantity
                        option_action.trades[put_symbol] = quantity
        else:
            for symbol, pos in position.option_positions.items():
                if pos.net_quantity != 0:
                    option_action.trades[symbol] = -pos.net_quantity
        
        greeks = state['greeks']
        delta, gamma, theta, vega = greeks[0], greeks[1], greeks[2], greeks[3]
        
        hedge_quantity = self.hedger.compute_hedge(
            delta, gamma, theta, vega,
            position_info={},
            market_info={}
        )
        
        hedge_action = HedgeAction(timestamp=timestamp, quantity=hedge_quantity)
        
        self.step_count += 1
        
        return option_action, hedge_action
    
    def reset(self):    
        self.step_count = 0
        self.hedger.reset()

