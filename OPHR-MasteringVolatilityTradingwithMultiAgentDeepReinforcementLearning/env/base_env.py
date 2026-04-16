import abc
import random
import datetime
import time
import copy
import math
from typing import List, Dict
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP

from click import option

from env.data.data_handler import DataHandler
from env.base.tick import Tick
from env.base.option import OptionTypes
from env.base.perpetual import Perpetual
from env.base.options_chain import OptionsChain
from env.base.action import Action, OptionAction, HedgeAction
from env.base.positions import Positions, OptionPosition, PerpetualPosition
from env.base.account import Account
from env.base.log import Log
from env.config import fee_config

class BaseEnv(abc.ABC):
    def __init__(self, data_handler: DataHandler, config, crypto: str = None, pm_config: dict = None):
        """
        Initialize Base Environment
        
        Args:
            data_handler: Data handler instance
            config: Environment config dict with episode_length, option_interval, etc.
            crypto: Cryptocurrency symbol ('BTC' or 'ETH') - used to load pm_config if not provided
            pm_config: Optional PM config dict (if None, uses default config for crypto)
        """
        self._data_handler = data_handler
        self._episode_length = config['episode_length']
        self._date_range = data_handler.get_date_range(self._episode_length)
        self._option_trade_interval = config['option_interval']
        self._perpetual_contract_size = Decimal('10')
        self._ts = None
        self._tick = None
        self._account = None
        self._log = None
        self._positions = None
        self._start_time = None
        self._end_time = None
        self._just_traded = False
        self._include_volatility_tickers = config.get('include_volatility_tickers', False)
        

        if crypto is None:
            crypto = data_handler._crypto
        self._crypto = crypto


        if pm_config is not None:
            self.pm_config = pm_config
        else:

            try:
                from config import ConfigManager
                config_manager = ConfigManager()
                pm_cfg = config_manager.env_config.get_pm_config(crypto)
                

                self.pm_config = {
                    crypto: {
                        "price_range": pm_cfg.price_range,
                        "min_expiry_delta_shock": pm_cfg.min_expiry_delta_shock,
                        "annualized_move_risk": pm_cfg.annualized_move_risk,
                        "extended_dampener": pm_cfg.extended_dampener,
                        "volatility_range_up": pm_cfg.volatility_range_up,
                        "volatility_range_down": pm_cfg.volatility_range_down,
                        "short_term_vega_power": pm_cfg.short_term_vega_power,
                        "long_term_vega_power": pm_cfg.long_term_vega_power,
                        "delta_total_liquidity_shock_threshold": pm_cfg.delta_total_liquidity_shock_threshold,
                        "max_delta_shock": pm_cfg.max_delta_shock,
                        "min_volatility_for_shock_up": pm_cfg.min_volatility_for_shock_up,
                        "extended_table_factor": pm_cfg.extended_table_factor
                    }
                }
            except:

                self.pm_config = {
                    "BTC": {
                        "price_range": Decimal('0.16'),
                        "min_expiry_delta_shock": Decimal('0.01'),
                        "annualized_move_risk": Decimal('0.075'),
                        "extended_dampener": Decimal('100000'),
                        "volatility_range_up": Decimal('0.50'),
                        "volatility_range_down": Decimal('0.25'),
                        "short_term_vega_power": Decimal('0.30'),
                        "long_term_vega_power": Decimal('0.13'),
                        "delta_total_liquidity_shock_threshold": Decimal('20000000'),
                        "max_delta_shock": Decimal('0.10'),
                        "min_volatility_for_shock_up": Decimal('0.50'),
                        "extended_table_factor": Decimal('1.00')
                    },
                    "ETH": {
                        "price_range": Decimal('0.16'),
                        "min_expiry_delta_shock": Decimal('0.01'),
                        "annualized_move_risk": Decimal('0.075'),
                        "extended_dampener": Decimal('100000'),
                        "volatility_range_up": Decimal('0.50'),
                        "volatility_range_down": Decimal('0.25'),
                        "short_term_vega_power": Decimal('0.30'),
                        "long_term_vega_power": Decimal('0.13'),
                        "delta_total_liquidity_shock_threshold": Decimal('20000000'),
                        "max_delta_shock": Decimal('0.10'),
                        "min_volatility_for_shock_up": Decimal('0.50'),
                        "extended_table_factor": Decimal('1.00')
                    }
                }

        self.reset()

    def reset(self, start_date: datetime.datetime = None, end_date: datetime.datetime = None):
        if not start_date:
            start_date = random.choice(self._date_range)

        self._start_time = datetime.datetime.combine(start_date.date(),
                                                     datetime.datetime.min.time()) + datetime.timedelta(hours=8)

        if end_date:
            self._end_time = datetime.datetime.combine(end_date.date(),
                                                       datetime.datetime.min.time()) + datetime.timedelta(hours=8)
        else:
            self._end_time = self._start_time + datetime.timedelta(days=self._episode_length)

        self._ts = self._start_time
        self._tick = None
        
        # Get initial capital from config (with fallback)
        initial_capital = Decimal('10')  # Default fallback
        if hasattr(self, '_crypto'):
            try:
                from config import ConfigManager
                config_manager = ConfigManager()
                initial_capital = config_manager.env_config.initial_capital
            except:
                pass  # Use default
        
        self._account = Account(timestamp=self._ts, cash_balance=initial_capital)
        self._log = Log(timestamp=self._ts)
        self._positions = Positions(timestamp=self._ts)
        self._account.net_value = self._account.cash_balance

        self._log.add_account_record(
            timestamp=self._ts,
            cash_balance=self._account.cash_balance,
            unrealized_pnl=Decimal('0'),
            realized_pnl=Decimal('0'),
            fee=Decimal('0'),
            total_value=self._account.net_value
        )

        self._log.add_value_record(
            timestamp=self._ts,
            net_value=self._account.net_value,
            mark_price=Decimal('0')
        )

        self._data_handler.reset(self._start_time)
        self._tick = self._data_handler.getNextTick()
        self._option_chain = self._tick.options_chain
        assert self._ts == self._tick.timestamp

        return self._tick, self._account, self._positions, self._log, False

    def step(self, option_action: OptionAction, hedge_action: HedgeAction):
        self.remove_closed_positions()

        if self._ts.hour == 8 and self._ts.minute == 0:
            self._process_expiration()

        if self.checkAction(option_action):
            self._process_option_action(option_action, self._option_chain)
            self._update_greeks(self._tick)
            self._just_traded = True

        if self.checkHedge(hedge_action):
            self._process_hedge_action(hedge_action, self._tick.perpetual)
            self._just_traded = True

        if self._positions.perpetual_position:
            self._calculate_funding_fee(self._positions.perpetual_position.net_quantity, self._tick.perpetual)

        next_tick = self._data_handler.getNextTick()
        self._step(next_tick)
        done = self._ts >= self._end_time

        return self._tick, self._account, self._positions, self._log, done
    
    def save_state(self) -> dict:
        return {
            'timestamp': self._ts,
            'tick': copy.deepcopy(self._tick),
            'account': copy.deepcopy(self._account),
            'positions': copy.deepcopy(self._positions),
            'log': copy.deepcopy(self._log),
            'start_time': self._start_time,
            'end_time': self._end_time,
            'just_traded': self._just_traded,
            'option_chain': copy.deepcopy(self._option_chain),
        }
    
    def restore_state(self, state: dict):
        self._ts = state['timestamp']
        self._tick = copy.deepcopy(state['tick'])
        self._account = copy.deepcopy(state['account'])
        self._positions = copy.deepcopy(state['positions'])
        self._log = copy.deepcopy(state['log'])
        self._start_time = state['start_time']
        self._end_time = state['end_time']
        self._just_traded = state['just_traded']
        self._option_chain = copy.deepcopy(state['option_chain'])
    
    def get_net_value(self) -> Decimal:
        if self._account is None:
            return Decimal('0')
        
        return self._account.net_value

    def _step(self, tick: Tick):
        self._option_chain.timestamp = tick.timestamp
        self._option_chain.calls.update(tick.options_chain.calls)
        self._option_chain.puts.update(tick.options_chain.puts)
        self._positions.update_option_snapshot(self._option_chain)

        if self._just_traded:
            realized_pnl = self._positions.settle_realized_pnl()
            self._account.cash_balance += realized_pnl
            self._just_traded = False

        options_value = Decimal('0')
        for symbol, option_position in self._positions.option_positions.items():
            option = self._option_chain.calls.get(symbol) or self._option_chain.puts.get(symbol)
            if option is None:
                continue
            
            mark_price = Decimal(str(option.mark_price))
            if mark_price <= 0:
                if option_position.last_valid_price is not None:
                    mark_price = option_position.last_valid_price
                else:
                    continue
            else:
                option_position.last_valid_price = mark_price
            
            position_value = mark_price * option_position.net_quantity
            options_value += position_value
        
        # Unrealized PnL (BTC) for perpetual only (inverse contract)
        perp_unrealized_pnl = self._positions.perpetual_position.calculate_unrealized_pnl(tick.perpetual)
        
        # Net value in BTC: cash (BTC) + options marked-to-market (BTC) + perpetual unrealized PnL (BTC)
        self._account.net_value = self._account.cash_balance + options_value + perp_unrealized_pnl

        self._calculate_portfolio_margins()
        self._update_greeks(tick)

        self._log.add_account_record(
            timestamp=tick.timestamp,
            cash_balance=self._account.cash_balance,
            unrealized_pnl=perp_unrealized_pnl,
            realized_pnl=self._positions.realized_pnl,
            fee=Decimal('0'),
            total_value=self._account.net_value
        )

        self._log.add_value_record(
            timestamp=tick.timestamp,
            net_value=self._account.net_value,
            mark_price=tick.perpetual.mark_price
        )

        self._ts = tick.timestamp
        self._tick = tick
        self._account.timestamp = self._ts
        self._positions.timestamp = self._ts
        self._log.timestamp = self._ts

    def checkPositions(self, current_positions: Positions, current_time: datetime.datetime) -> list:
        expiration_option = []
        for symbol, option_position in current_positions.option_positions.items():
            if current_time == option_position.option.expiration:
                expiration_option.append(symbol)
        return expiration_option

    def _process_expiration(self):
        expiration_options = self.checkPositions(self._positions, self._ts)
        underlying_price = self._tick.perpetual.mark_price
        for symbol in expiration_options:
            self._positions.option_positions[symbol].expire(underlying_price)
            self._update_greeks(self._tick)

        if expiration_options:
            self._just_traded = True

    def checkAction(self, action: OptionAction) -> bool:
        if len(action.trades) == 0:
            return False
        return True

    def checkHedge(self, action: HedgeAction) -> bool:
        self._log.hedges.append(action.quantity)
        if action.quantity:
            return True
        return False

    def _process_option_action(self, action: OptionAction, options_chain: OptionsChain):
        total_fee = Decimal('0')
        simulated_positions = copy.deepcopy(self._positions)

        for symbol, qty in action.trades.items():
            option = options_chain.calls.get(symbol) or options_chain.puts.get(symbol)
            if option is None:
                continue

            if symbol in simulated_positions.option_positions:
                fee = simulated_positions.option_positions[symbol].trade(qty, option)
                total_fee += fee
            else:
                simulated_positions.option_positions[symbol] = OptionPosition(option)
                fee = simulated_positions.option_positions[symbol].trade(qty, option)
                total_fee += fee

        simulated_worst_pnl = self._calculate_risk_matrix_worst_pnl(simulated_positions)
        simulated_delta_shock = self._calculate_delta_shock(simulated_positions)
        simulated_roll_shock = self._calculate_roll_shock(simulated_positions)
        simulated_initial_margin = simulated_worst_pnl + simulated_delta_shock + simulated_roll_shock

        additional_margin = simulated_initial_margin - self._account.initial_margin

        if additional_margin > self._account.liquidation_value:
            return

        self._positions = simulated_positions

        options_value = Decimal('0')
        for symbol, option_position in self._positions.option_positions.items():
            option = options_chain.calls.get(symbol) or options_chain.puts.get(symbol)
            if option is None:
                continue
            
            mark_price = Decimal(str(option.mark_price))
            if mark_price <= 0:
                if option_position.last_valid_price is not None:
                    mark_price = option_position.last_valid_price
                else:
                    continue
            else:
                option_position.last_valid_price = mark_price
            
            options_value += mark_price * option_position.net_quantity

        # Unrealized PnL (BTC) for perpetual only
        perp_unrealized_pnl = self._positions.perpetual_position.calculate_unrealized_pnl(self._tick.perpetual)
        self._account.net_value = self._account.cash_balance + options_value + perp_unrealized_pnl

        self._log.add_account_record(
            timestamp=self._ts,
            cash_balance=self._account.cash_balance,
            unrealized_pnl=perp_unrealized_pnl,
            realized_pnl=self._positions.calculate_realized_pnl()[0],
            fee=total_fee,
            total_value=self._account.net_value
        )

        for symbol, qty in action.trades.items():
            if qty == 0:
                continue
            
            option = options_chain.calls.get(symbol) or options_chain.puts.get(symbol)
            if option is None:
                continue

            if qty > 0:
                trade_price = Decimal(str(option.ask_price))
                if trade_price <= 0:
                    if symbol in self._positions.option_positions and self._positions.option_positions[symbol].last_valid_price is not None:
                        trade_price = self._positions.option_positions[symbol].last_valid_price
                    else:
                        continue
                if trade_price <= 0:
                    continue
                option_cost = trade_price * qty
                self._account.cash_balance -= option_cost
            else:
                trade_price = Decimal(str(option.bid_price))
                if trade_price <= 0:
                    if symbol in self._positions.option_positions and self._positions.option_positions[symbol].last_valid_price is not None:
                        trade_price = self._positions.option_positions[symbol].last_valid_price
                    else:
                        continue
                if trade_price <= 0:
                    continue
                option_premium = trade_price * abs(qty)
                self._account.cash_balance += option_premium

        self._account.cash_balance -= total_fee
        self._log.option_fee += total_fee

        for symbol, qty in action.trades.items():
            if qty == 0:
                continue
                
            option = options_chain.calls.get(symbol) or options_chain.puts.get(symbol)
            if option is None:
                continue

            if qty > 0:
                trade_price = option.ask_price
                side = 'long'
            else:
                trade_price = option.bid_price
                side = 'short'

            self._log.add_trade_record(
                timestamp=self._ts,
                symbol=symbol,
                instrument_type='option',
                side=side,
                quantity=abs(qty),
                price=trade_price,
                fee=total_fee,
                trade_id=str(int(time.time()))
            )

            self._log.add_order_record(
                timestamp=self._ts,
                symbol=symbol,
                instrument_type='option',
                order_type='market',
                side=side,
                quantity=abs(qty),
                price=trade_price,
                status='filled'
            )

    def _process_hedge_action(self, hedge_action: HedgeAction, perpetual: Perpetual):
        total_fee = Decimal('0')
        simulated_positions = copy.deepcopy(self._positions)

        if hedge_action.quantity > 0:
            side = 'buy'
            trade_price = perpetual.ask_prices[0] if len(perpetual.ask_prices) > 0 else perpetual.mark_price
        else:
            side = 'sell'
            trade_price = perpetual.bid_prices[0] if len(perpetual.bid_prices) > 0 else perpetual.mark_price

        fee = simulated_positions.perpetual_position.trade(hedge_action.quantity, perpetual)
        total_fee += fee

        simulated_worst_pnl = self._calculate_risk_matrix_worst_pnl(simulated_positions)
        simulated_delta_shock = self._calculate_delta_shock(simulated_positions)
        simulated_roll_shock = self._calculate_roll_shock(simulated_positions)
        simulated_initial_margin = simulated_worst_pnl + simulated_delta_shock + simulated_roll_shock

        additional_margin = simulated_initial_margin - self._account.initial_margin

        if additional_margin > self._account.liquidation_value:
            return

        self._positions = simulated_positions
        self._account.cash_balance -= fee
        self._log.hedge_fee += fee

        self._log.add_trade_record(
            timestamp=self._ts,
            symbol="BTC",
            instrument_type='perpetual',
            side=side,
            quantity=abs(hedge_action.quantity),
            price=trade_price,
            fee=fee,
            trade_id=str(int(time.time()))
        )

        self._log.add_order_record(
            timestamp=self._ts,
            symbol="BTC",
            instrument_type='perpetual',
            order_type='market',
            side=side,
            quantity=abs(hedge_action.quantity),
            price=trade_price,
            status='filled'
        )

        options_value = Decimal('0')
        for symbol, option_position in self._positions.option_positions.items():
            option = self._option_chain.calls.get(symbol) or self._option_chain.puts.get(symbol)
            if option is None:
                continue
            
            mark_price = Decimal(str(option.mark_price))
            if mark_price <= 0:
                if option_position.last_valid_price is not None:
                    mark_price = option_position.last_valid_price
                else:
                    continue
            else:
                option_position.last_valid_price = mark_price
            
            options_value += mark_price * option_position.net_quantity

        # Unrealized PnL (BTC) for perpetual only
        perp_unrealized_pnl = self._positions.perpetual_position.calculate_unrealized_pnl(self._tick.perpetual)
        self._account.net_value = self._account.cash_balance + options_value + perp_unrealized_pnl

        self._log.add_account_record(
            timestamp=self._ts,
            cash_balance=self._account.cash_balance,
            unrealized_pnl=perp_unrealized_pnl,
            realized_pnl=self._positions.calculate_realized_pnl()[0],
            fee=fee,
            total_value=self._account.net_value
        )

    def remove_closed_positions(self):
        has_closed = []
        for symbol, option_position in self._positions.option_positions.items():
            if option_position.net_quantity == 0:
                has_closed.append(symbol)
        for symbol in has_closed:
            del self._positions.option_positions[symbol]

    def _calculate_portfolio_margins(self):
        worst_pnl = self._calculate_risk_matrix_worst_pnl(self._positions)
        delta_shock = self._calculate_delta_shock(self._positions)
        roll_shock = self._calculate_roll_shock(self._positions)
        position_initial_margin = worst_pnl + delta_shock + roll_shock
        self._account.initial_margin = position_initial_margin
        maintenance_margin = position_initial_margin * Decimal('0.80')
        self._account.maintenance_margin = maintenance_margin

    def _calculate_funding_fee(self, positions_perp: Decimal, perpetual: Perpetual):
        funding_fee = positions_perp / perpetual.mark_price * 60 / 480 * perpetual.funding_rate
        self._account.cash_balance -= funding_fee

    def _update_greeks(self, tick: Tick):
        delta, gamma, theta, vega = Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0')
        for symbol, option_position in self._positions.option_positions.items():
            if option_position.net_quantity == 0:
                continue
            option = self._option_chain.calls.get(symbol)
            if option is None:
                option = self._option_chain.puts.get(symbol)
            if option is None:
                continue

            delta += option.delta * option_position.net_quantity
            gamma += option.gamma * option_position.net_quantity
            theta += option.theta * option_position.net_quantity
            vega += option.vega * option_position.net_quantity

        delta -= self._positions.perpetual_position.net_quantity / tick.perpetual.mark_price

        self._positions.delta = delta
        self._positions.gamma = gamma
        self._positions.theta = theta
        self._positions.vega = vega

    def _calculate_risk_matrix_worst_pnl(self, positions):
        main_moves = [-0.16, -0.12, -0.08, -0.04, 0, 0.04, 0.08, 0.12, 0.16]
        vol_shocks = [0.75, 1.0, 1.5]

        pnl_matrix = []
        pnl_list = []
        
        index = self._tick.perpetual.mark_price
        for move in main_moves:
            row = []
            for shock in vol_shocks:
                simulated_pnl = self._simulate_pnl(positions, move, shock)
                row.append(simulated_pnl)
                pnl_list.append(simulated_pnl)
            pnl_matrix.append(row)

        extended_moves = [-0.66, -0.33, 0.50, 1.00, 2.00, 3.00, 4.00, 5.00]
        extended_dampener = self.pm_config[self._crypto]["extended_dampener"]
        factor = self.pm_config[self._crypto]["extended_table_factor"]
        price_range = self.pm_config[self._crypto]["price_range"]
        
        for move in extended_moves:
            move = Decimal(move)
            margin_multiplier = factor * (price_range / abs(Decimal(move)))

            adjusted_simulated_pnl = (
                    self._simulate_pnl(positions, move, 1.5) * margin_multiplier * (1 + move) / (1 + (Decimal(math.copysign(1, move)) * price_range))
            )
            amount_to_be_dampened = min(
                (max(abs(Decimal(move)) / price_range, 1) - 1) * extended_dampener / ((1 + (Decimal(math.copysign(1, move))) * price_range) * index),
                abs(adjusted_simulated_pnl))
            final_pnl = adjusted_simulated_pnl + amount_to_be_dampened
            pnl_list.append(final_pnl)
            
        worst_pnl = min(pnl_list)
        return worst_pnl

    def _simulate_pnl(self, positions, price_move: float, vol_shock: float):
        pnl_total = Decimal('0')

        for symbol, option_position in positions.option_positions.items():
            option = self._option_chain.calls.get(symbol) or self._option_chain.puts.get(symbol)
            if option:
                days_to_expiry = (option.expiration - self._ts).days
                vega_power = (
                    self.pm_config[self._crypto]["short_term_vega_power"]
                    if days_to_expiry < 30
                    else self.pm_config[self._crypto]["long_term_vega_power"]
                )

                if vol_shock < 1.0:
                    simulated_vol = max(
                        option.mark_iv * (1 - (Decimal(30) / Decimal(max(days_to_expiry, 1))) ** vega_power) * Decimal(
                            vol_shock),
                        Decimal('0')
                    )
                else:
                    simulated_vol = max(
                        option.mark_iv * (1 + (Decimal(30) / Decimal(max(days_to_expiry, 1))) ** vega_power) * Decimal(
                            vol_shock),
                        self.pm_config[self._crypto]["min_volatility_for_shock_up"]
                    )

                simulated_price = option.underlying_price * Decimal(1 + price_move)

                delta_adjusted = option.delta * simulated_vol / option.mark_iv
                gamma_adjusted = option.gamma * simulated_vol / option.mark_iv
                vega_adjusted = option.vega * (simulated_vol - option.mark_iv) / option.mark_iv
                theta_adjusted = option.theta

                price_change = simulated_price - option.underlying_price
                pnl = (
                              delta_adjusted * price_change +
                              Decimal('0.5') * gamma_adjusted * (price_change ** 2) +
                              vega_adjusted +
                              theta_adjusted
                      ) * option_position.net_quantity

                pnl_total += pnl

        if positions.perpetual_position:
            perpetual = positions.perpetual_position
            mark_price = self._tick.perpetual.mark_price
            simulated_price = mark_price * Decimal(1 + price_move)
            pnl_perpetual = perpetual.net_quantity * (1 / mark_price - 1 / simulated_price) * self._perpetual_contract_size
            pnl_total += pnl_perpetual
    
        return pnl_total/mark_price

    def _calculate_delta_shock(self, positions):
        delta_shock = Decimal('0')
        index = self._tick.perpetual.mark_price
        delta_total_liquidity_shock_threshold = self.pm_config[self._crypto]["delta_total_liquidity_shock_threshold"]
        increment = Decimal('0.01')
        max_delta_shock = self.pm_config[self._crypto]["max_delta_shock"]

        optPosition = positions.option_positions
        perpPostion = positions.perpetual_position
        delta1 = Decimal('0')
        delta2 = perpPostion.net_quantity/index 
        for symbol, position in optPosition.items():
            if position.net_quantity > 0:
                delta1 += position.net_quantity * position.option.delta
            else:
                delta2 += position.net_quantity * position.option.delta

        delta_for_shock = (
            min(max(delta1 + delta2, delta2), Decimal('0')) if delta2 < 0 
            else max(min(delta1 + delta2, delta2), Decimal('0'))
        )

        part1 = max((delta_for_shock * index - delta_total_liquidity_shock_threshold),
                    Decimal('0')) * delta_for_shock * increment
        part2 = max_delta_shock * index * delta_for_shock
        delta_shock += min(part1, part2) / index

        return delta_shock

    def _calculate_roll_shock(self, positions):
        min_expiry_delta_shock = self.pm_config[self._crypto]["min_expiry_delta_shock"]
        annualized_move_risk = self.pm_config[self._crypto]["annualized_move_risk"]
        index = self._tick.perpetual.mark_price
        roll_shock = Decimal('0')

        grouped_expiry = {}
        for symbol, option_position in positions.option_positions.items():
            expiry = option_position.option.expiration
            grouped_expiry.setdefault(expiry, Decimal('0'))
            grouped_expiry[expiry] += option_position.net_quantity*option_position.option.delta #net delta of each expiry

        for expiry, net_delta_expiry in grouped_expiry.items():
            years_to_expiry = Decimal((expiry - self._ts).days) / Decimal('365')

            minimum_roll_shock = min_expiry_delta_shock * abs(net_delta_expiry)
            annualized_roll_shock = max(
                Decimal(math.exp(annualized_move_risk * years_to_expiry)) - 1,
                min_expiry_delta_shock
            ) * net_delta_expiry

            roll_shock += max(minimum_roll_shock, abs(annualized_roll_shock))

        # perpetual roll shock only use minimum roll shock
        perp_roll_shock = min_expiry_delta_shock * abs(positions.perpetual_position.net_quantity) / index
        roll_shock += perp_roll_shock

        return roll_shock


if __name__ == '__main__':

    start_date = datetime.datetime(2022, 2, 1)
    end_date = datetime.datetime(2023, 1, 31)
    crypto = 'BTC'
    option_chain_path = ''
    perpetual_path = ''
    config = {'episode_length': 14, 'option_interval': 180}


    data_handler = DataHandler(
        start_date, end_date, crypto, option_chain_path, perpetual_path
    )

    data_handler.reset(start_date)


    env = BaseEnv(data_handler, config)


    done = False
    tick, account, position, log, done = env.reset(start_date)
    while not done:

        option_action = OptionAction(timestamp=tick.timestamp, trades={})
        hedge_action = HedgeAction(timestamp=tick.timestamp, quantity=Decimal('0'))

        tick, account, positions, log, done = env.step(option_action, hedge_action)
