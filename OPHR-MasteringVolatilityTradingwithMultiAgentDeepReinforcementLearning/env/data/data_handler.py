import abc
import os
import gc
import time
import datetime
from decimal import Decimal
import json
import polars as pl
import numpy as np
import pandas as pd

from env.base import call
from env.base import put
from env.base.perpetual import PerpetualOHLCV, Perpetual
from env.base.options_chain import OptionsChain
from env.base.volatility_tickers import VolatilityTickers
from env.base.tick import Tick
from typing import Iterable

class DataHandler(abc.ABC):
    """This class is the base class for handling data. It is an abstract class."""
    def __init__(self, start_date: datetime.datetime, end_date: datetime.datetime, crypto: str,
                 options_chain_path: str, perpetual_data_path: str, volatility_ticker_path: str = None) -> None:
        self._crypto = crypto
        self._start_date = start_date
        self._end_date = end_date
        self._current_date = None
        self._ts = None
        self._options_chain_dir = options_chain_path
        self._options_chain_df = None
        self._perpetual_df = pl.read_parquet(perpetual_data_path).to_pandas()
        self._current_perp_index = 0
        
        # 加载volatility_tickers数据（如果提供）
        self._volatility_ticker_df = None
        if volatility_ticker_path:
            self._volatility_ticker_df = pl.read_parquet(volatility_ticker_path).to_pandas()

        self.reset(self._start_date)

    def reset(self, start_time) -> None:
        """
        Reset the data handler to the start date.
        """
        self._ts = start_time
        self._current_date = self._ts.date()
        self._load_data(self._current_date)
        self._current_perp_index = self._perpetual_df[self._perpetual_df['timestamp'] == self._ts].index

    def _load_data(self, date: datetime.date) -> None:
        """
        Load the data for the given date.
        """
        # Release previous data before loading new data
        if hasattr(self, '_options_chain_df'):
            del self._options_chain_df
            gc.collect()

        # Read option chain
        oc_file_name = f"{date.strftime('%Y-%m-%d')}.parquet"
        oc_file_path = os.path.join(self._options_chain_dir, oc_file_name)

        if os.path.isfile(oc_file_path):
            try:
                self._options_chain_df = pl.read_parquet(oc_file_path)
            except Exception:
                self._options_chain_df = pl.DataFrame()
        else:
            self._options_chain_df = pl.DataFrame()

    def get_date_range(self, episode_length: int) -> Iterable[datetime.datetime]:
        return pd.date_range(start=self._start_date,
                             end=self._end_date - datetime.timedelta(days=episode_length), freq='1h')

    def getNextTick(self) -> Tick:
        current_options_chain = self._options_chain_df.filter(pl.col('timestamp') == self._ts)
        calls = dict()
        puts = dict()

        # Get the calls and puts
        for row in current_options_chain.iter_rows(named=True):
            bid_price = row['bid_price']
            ask_price = row['ask_price']

            # Handle cases where ask_price or bid_price might still be None
            spread = (ask_price - bid_price) if ask_price > 0 and bid_price > 0 else -1
            spread_iv = (row['ask_iv'] - row['bid_iv']) if row['ask_iv'] > 0 and row['bid_iv'] > 0 else -1

            args = {
                'symbol': row['symbol'],
                'timestamp': row['timestamp'],
                'strike_price': Decimal(row['strike_price']),
                'expiration': row['expiration'],
                'open_interest': Decimal(row['open_interest']),
                'last_price': Decimal(row['last_price']),
                'bid_price': Decimal(bid_price),
                'bid_amount': Decimal(row['bid_amount']),
                'bid_iv': Decimal(row['bid_iv']),
                'ask_price': Decimal(ask_price),
                'ask_amount': Decimal(row['ask_amount']),
                'ask_iv': Decimal(row['ask_iv']),
                'mark_price': Decimal(row['mark_price']),
                'mark_iv': Decimal(row['mark_iv']),
                'spread': Decimal(spread),
                'spread_iv': Decimal(spread_iv),
                'underlying_price': Decimal(row['underlying_price']),
                'delta': Decimal(row['delta']) - Decimal(row['mark_price']),
                'gamma': Decimal(row['gamma']),
                'theta': Decimal(row['theta']),
                'vega': Decimal(row['vega']),
                'rho': Decimal(row['rho']),
            }
            if row['type'] == 'call':
                calls[row['symbol']] = call.Call(**args)
            else:
                puts[row['symbol']] = put.Put(**args)

        options_chain = OptionsChain(self._ts, calls, puts)

        # 获取当前时间戳的永续合约数据
        current_perp_data = self._perpetual_df[self._perpetual_df['timestamp'] == self._ts]
        if len(current_perp_data) == 0:
            raise ValueError(f"Missing perpetual data for timestamp: {self._ts}")
            
        current_row = current_perp_data.iloc[0]
        
        # 处理volatility_tickers数据
        if self._volatility_ticker_df is not None:
            # 从volatility_ticker_df获取数据
            volatility_mask = (self._volatility_ticker_df['timestamp'] == self._ts)
            if volatility_mask.any():
                current_volatility_data = self._volatility_ticker_df[volatility_mask].iloc[0]
                voti_array = np.array(current_volatility_data.iloc[1:].values.flatten())
                volatility_tickers = VolatilityTickers(
                    timestamp=self._ts, 
                    min_mark_iv=Decimal('0.01'),  # 使用默认值
                    min_mark_iv_delta=Decimal('0'), 
                    features=voti_array
                )
            else:
                # 如果没有找到对应时间戳的数据，使用默认值
                voti_array = np.zeros(48)  # 假设有48个特征
                volatility_tickers = VolatilityTickers(
                    timestamp=self._ts, 
                    min_mark_iv=Decimal('0.01'), 
                    min_mark_iv_delta=Decimal('0'), 
                    features=voti_array
                )
        else:
            # 使用原有的方法
            voti_array = np.array(current_row.iloc[-76:-4].values.flatten())
            if not current_options_chain.is_empty():
                min_iv_index = current_options_chain['mark_iv'].arg_min()
                min_mark_iv = current_options_chain[min_iv_index, 'mark_iv']
                min_mark_iv_delta = current_options_chain[min_iv_index, 'delta']
                volatility_tickers = VolatilityTickers(timestamp=self._ts, min_mark_iv=Decimal(min_mark_iv),
                                                    min_mark_iv_delta=Decimal(min_mark_iv_delta), features=voti_array)
            else:
                volatility_tickers = VolatilityTickers(timestamp=self._ts, min_mark_iv=None, min_mark_iv_delta=None,
                                                    features=voti_array)

        __temp_askp = []
        __temp_bidp = []
        __temp_aska = []
        __temp_bida = []

        for i in range(1, 26):
            __temp_askp.append(current_row[f'ask{i}_price'])
            __temp_bidp.append(current_row[f'bid{i}_price'])
            __temp_aska.append(current_row[f'ask{i}_size'])
            __temp_bida.append(current_row[f'bid{i}_size'])

        features = np.array(current_row.iloc[105:-177]).flatten()
        future_information = {col: current_row[col] for col in current_row.index[-4:]}

        perpetual = Perpetual(
            timestamp=self._ts,
            ask_prices=np.array(__temp_askp),
            bid_prices=np.array(__temp_bidp),
            ask_quantities=np.array(__temp_aska),
            bid_quantities=np.array(__temp_bida),
            features=features,
            mark_price=Decimal(str(current_row['mark_price'])),
            funding_rate=Decimal(str(current_row['funding_rate'])),
            future_information=future_information
        )

        tick = Tick(timestamp=self._ts, options_chain=options_chain, perpetual=perpetual,
                    volatility_tickers=volatility_tickers, open=False)

        self._ts += datetime.timedelta(hours=1)
        self._current_perp_index = self._perpetual_df[self._perpetual_df['timestamp'] == self._ts].index[0] if len(self._perpetual_df[self._perpetual_df['timestamp'] == self._ts]) > 0 else self._current_perp_index + 1

        if self._ts.date() > self._current_date:
            self._current_date = self._ts.date()
            self._load_data(self._current_date)

        return tick
