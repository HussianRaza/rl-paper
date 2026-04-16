import abc
import dataclasses
import datetime
import math
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP
from typing import Dict, Union, List, Tuple
import numpy as np
from env.base.option import Option, OptionTypes
from env.base.options_chain import OptionsChain
from env.base.perpetual import Perpetual

@dataclasses.dataclass
class OptionPosition(abc.ABC):
    symbol: str
    strike: Decimal
    type: OptionTypes
    option: Option
    average_price: Decimal
    net_quantity: Decimal
    fee: Decimal
    unrealized_pnl: Decimal
    expire_cash_flow: Decimal
    realized_pnl: Decimal

    def __init__(self, option: Option):
        self.symbol = option.symbol
        self.strike = option.strike_price
        self.type = option.type
        self.option = option
        self.fee = Decimal(0)
        self.net_quantity = Decimal(0)
        self.average_price = None
        self.unrealized_pnl = Decimal(0)
        self.expire_cash_flow = Decimal(0)
        self.realized_pnl = Decimal(0)
        self.last_valid_price = None

    def trade(self, quantity: Decimal, option: Option):
        if quantity == 0:
            return self.fee

        if quantity > 0:
            trade_price = Decimal(str(option.ask_price))
            if trade_price <= 0:
                if self.last_valid_price is None:
                    return self.fee
                trade_price = self.last_valid_price
        else:
            trade_price = Decimal(str(option.bid_price))
            if trade_price <= 0:
                if self.last_valid_price is None:
                    return self.fee
                trade_price = self.last_valid_price

        self.last_valid_price = trade_price
        self.fee += min(Decimal('0.0003'), abs(trade_price) * Decimal('0.125')) * abs(quantity)

        if self.net_quantity == 0:
            self.net_quantity = quantity
            self.average_price = trade_price
            return self.fee

        if (self.net_quantity > 0 and quantity > 0) or (self.net_quantity < 0 and quantity < 0):
            new_position_size = self.net_quantity + quantity
            current_cost = self.average_price * self.net_quantity
            added_cost = trade_price * quantity
            total_cost = current_cost + added_cost
            self.net_quantity = new_position_size
            self.average_price = total_cost / new_position_size
        else:
            closed_size = min(abs(self.net_quantity), abs(quantity))
            
            if self.net_quantity > 0:
                realized_pnl_part = Decimal(str(closed_size)) * (trade_price - self.average_price)
            else:
                realized_pnl_part = -Decimal(str(closed_size)) * (trade_price - self.average_price)

            self.realized_pnl += realized_pnl_part
            self.net_quantity += quantity

            if (self.net_quantity > 0 and quantity < 0 and abs(quantity) >= closed_size) \
                    or (self.net_quantity < 0 and quantity > 0 and abs(quantity) >= closed_size) \
                    or self.net_quantity == 0:
                if self.net_quantity == 0:
                    self.average_price = None
                else:
                    self.average_price = trade_price

        return self.fee

    def calculate_unrealized_pnl(self, option_snapshot: Option):
        if self.net_quantity == 0 or self.average_price is None:
            self.unrealized_pnl = Decimal('0')
            return self.unrealized_pnl

        mark_price = Decimal(str(option_snapshot.mark_price))
        
        if mark_price <= 0:
            if self.last_valid_price is None:
                return self.unrealized_pnl
            mark_price = self.last_valid_price
        else:
            self.last_valid_price = mark_price

        self.unrealized_pnl = self.net_quantity * (mark_price - self.average_price)
        return self.unrealized_pnl

    def expire(self, underlying_mark_price: Decimal):
        """
        到期结算：对于币本位期权，到期时应该以最后有效价格（BTC单位）进行结算
        
        Args:
            underlying_mark_price: 标的价格（用于判断是否实值）
        """
        if self.net_quantity == 0 or self.average_price is None:
            self.expire_cash_flow = Decimal(0)
            return

        # 判断期权是否实值（in the money）
        is_itm = False
        if self.type == OptionTypes.CALL:
            is_itm = underlying_mark_price > self.strike
        else:  # PUT
            is_itm = underlying_mark_price < self.strike

        # 实值期权：使用最后有效价格（BTC单位）结算
        # 虚值期权：价值为0
        if is_itm and self.last_valid_price is not None:
            # 使用最后有效的市场价格（BTC单位）作为到期结算价
            settlement_price_btc = self.last_valid_price
        else:
            # 虚值期权或无有效价格，到期价值为0
            settlement_price_btc = Decimal(0)

        self.expire_cash_flow = self.net_quantity * settlement_price_btc
        cost_basis = self.net_quantity * self.average_price
        expiry_pnl = self.expire_cash_flow - cost_basis
        self.realized_pnl += expiry_pnl
        self.net_quantity = Decimal(0)
        self.average_price = None
        self.unrealized_pnl = Decimal(0)

@dataclasses.dataclass
class PerpetualPosition(abc.ABC):
    symbol: str
    net_quantity: Decimal
    average_price: Decimal
    fee: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    contract_size: Decimal

    def __init__(self):
        self.symbol = None
        self.net_quantity = Decimal(0)
        self.average_price = None
        self.fee = Decimal(0)
        self.unrealized_pnl = Decimal(0)
        self.realized_pnl = Decimal(0)
        self.contract_size = Decimal('1')

    def trade(self, quantity: Decimal, perpetual: Perpetual):
        self.fee = Decimal(0)
        prev_unrealized_pnl = self.unrealized_pnl

        if quantity == 0:
            return self.fee

        if quantity > 0:
            trade_price = Decimal(str(perpetual.ask_prices[0]))
        else:
            trade_price = Decimal(str(perpetual.bid_prices[0]))

        if self.net_quantity == 0:
            self.net_quantity = quantity
            self.average_price = trade_price
            return self.fee

        if (self.net_quantity > 0 and quantity > 0) or (self.net_quantity < 0 and quantity < 0):
            old_avg_price = self.average_price
            self.average_price = (self.net_quantity * old_avg_price + quantity * trade_price) / (self.net_quantity + quantity)
            self.net_quantity += quantity
            
        else:
            closed_size = min(abs(self.net_quantity), abs(quantity))
            old_avg_price = self.average_price
            remaining_qty = self.net_quantity + quantity

            if self.net_quantity > 0:  # long
                realized_pnl_part = closed_size * (1 / self.average_price - 1 / trade_price) * self.contract_size
            else:  # short
                realized_pnl_part = closed_size * (1 / trade_price - 1 / self.average_price) * self.contract_size

            self.realized_pnl += realized_pnl_part

            if remaining_qty == 0:
                self.net_quantity = Decimal('0')
                self.average_price = None
            elif abs(quantity) > closed_size:
                self.net_quantity = remaining_qty
                self.average_price = trade_price
            else:
                self.net_quantity = remaining_qty

        return self.fee

    def calculate_unrealized_pnl(self, perpetual: Perpetual):
        if self.net_quantity == 0 or self.average_price is None:
            self.unrealized_pnl = Decimal('0')
            return self.unrealized_pnl

        mark_price = Decimal(str(perpetual.mark_price))
        self.unrealized_pnl = self.net_quantity * (1 / self.average_price - 1 / mark_price) * self.contract_size
        return self.unrealized_pnl
    
@dataclasses.dataclass
class Positions(abc.ABC):
    timestamp: datetime.datetime
    perpetual_position: PerpetualPosition
    hedges: List[Tuple[datetime.datetime, Decimal, Decimal]]
    option_positions: Dict[str, OptionPosition]
    delta: Decimal
    gamma: Decimal
    theta: Decimal
    vega: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal

    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.perpetual_position = PerpetualPosition()
        self.hedges = []
        self.option_positions = {}
        self.delta = Decimal('0')
        self.gamma = Decimal('0')
        self.theta = Decimal('0')
        self.vega = Decimal('0')
        self.unrealized_pnl = Decimal('0')
        self.realized_pnl = Decimal('0')
    
    def calculate_realized_pnl(self) -> Tuple[Decimal, Decimal]:
        total_realized = self.perpetual_position.realized_pnl
        fee = self.perpetual_position.fee
        
        for opt_pos in self.option_positions.values():
            total_realized += opt_pos.realized_pnl
            fee += opt_pos.fee
            
        return total_realized, fee

    def calculate_unrealized_pnl(self, options_chain: OptionsChain, perpetual: Perpetual) -> Decimal:
        total_unrealized = self.perpetual_position.calculate_unrealized_pnl(perpetual)
        for symbol, option_position in self.option_positions.items():
            option = options_chain.calls.get(symbol) or options_chain.puts.get(symbol)
            if option is None:
                continue
            total_unrealized += option_position.calculate_unrealized_pnl(option)
        return total_unrealized

    def settle_realized_pnl(self):
        total_realized, fee = self.calculate_realized_pnl()
        
        self.perpetual_position.realized_pnl = Decimal('0')
        self.perpetual_position.fee = Decimal('0')
        for opt_pos in self.option_positions.values():
            opt_pos.realized_pnl = Decimal('0')
            opt_pos.fee = Decimal('0')

        return total_realized - fee

    def update_option_snapshot(self, options_chain: OptionsChain):
        for symbol, option_position in self.option_positions.items():
            option = options_chain.calls.get(symbol) or options_chain.puts.get(symbol)
            if option is not None:
                option_position.option = option

    def get_greeks(self):
        return np.array([self.delta, self.gamma, self.theta, self.vega], dtype=float)

    def get_hedge_history(self, history_length):
        hedge_history = np.zeros((int(history_length), 2), dtype=float)
        return hedge_history
    