import abc
import os
import gc
import datetime
import decimal
import json
import logging
import polars as pl
import numpy as np
import pandas as pd

import ray
import time
from ray.util.queue import Queue

from env.base.call import Call
from env.base.put import Put
from env.base.perpetual import PerpetualOHLCV, Perpetual
from env.base.options_chain import OptionsChain
from env.base.volatility_tickers import VolatilityTickers
from env.base.tick import Tick
from typing import Iterable

log_format = '%(asctime)s - %(filename)s - Line: %(lineno)d - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

@ray.remote
class DataHandler:
    def __init__(self, start_date: datetime.datetime, end_date: datetime.datetime, crypto: str,
                 options_chain_path: str, perpetual_data_path: str, data_queue: Queue):
        self._crypto = crypto
        self._start_date = start_date
        self._end_date = end_date
        self._current_date = None
        self._ts = None
        self._options_chain_dir = options_chain_path
        self._options_chain_df = None
        self._perpetual_df = pl.read_parquet(perpetual_data_path)
        self._data_queue = data_queue
        self._producing_flag = False

    def reset(self, start_time):
        self._producing_flag = False
        self._ts = start_time
        self._current_date = self._ts.date()
        self._load_data(self._current_date)
        # Clear the data queue if needed
        while not self._data_queue.empty():
            self._data_queue.get_nowait()

    def _load_data(self, date: datetime.date):
        # Read option chain
        if hasattr(self, '_options_chain_df'):
            del self._options_chain_df
            gc.collect()

        oc_file_name = f"{date.strftime('%Y-%m-%d')}.parquet"
        oc_file_path = os.path.join(self._options_chain_dir, oc_file_name)

        if os.path.isfile(oc_file_path):
            try:
                self._options_chain_df = pl.read_parquet(oc_file_path)
            except Exception as e:
                logger.error(f"Error loading Option Chain data: {e}")
                self._options_chain_df = pl.DataFrame()
        else:
            logger.error(f"Option Chain File not found: {oc_file_path}")
            self._options_chain_df = pl.DataFrame()

    def start_producing(self):
        self._producing_flag = True
        while self._producing_flag and self._ts <= self._end_date:
            tick = self.getNextTick()
            self._data_queue.put(tick)
            self._ts += datetime.timedelta(minutes=5)
            if self._ts.date() > self._current_date:
                self._current_date = self._ts.date()
                self._load_data(self._current_date)

    def stop_producing(self):
        self._producing_flag = False

    def get_date_range(self, episode_length: int) -> Iterable[datetime.datetime]:
        return pd.date_range(start=self._start_date,
                             end=self._end_date - datetime.timedelta(days=episode_length), freq='5T')

    def getNextTick(self) -> 'Tick':
        calls = dict()
        puts = dict()

        current_options_chain = self._options_chain_df.filter(pl.col('timestamp') == self._ts)

        if not current_options_chain.is_empty():
            min_iv_index = current_options_chain['mark_iv'].arg_min()
            min_mark_iv = current_options_chain[min_iv_index, 'mark_iv']
            min_mark_iv_delta = current_options_chain[min_iv_index, 'delta']
            volatility_tickers = VolatilityTickers(timestamp=self._ts, min_mark_iv=min_mark_iv,
                                                   min_mark_iv_delta=min_mark_iv_delta)
        else:
            volatility_tickers = VolatilityTickers(timestamp=self._ts, min_mark_iv=None, min_mark_iv_delta=None)
            logger.info(f"No options chain data available at: {self._ts}")

        # Get the calls and puts
        for row in current_options_chain.iter_rows(named=True):
            bid_price = row['bid_price']
            ask_price = row['ask_price']

            # Handle cases where ask_price or bid_price might still be None
            spread = (ask_price - bid_price) if ask_price is not None and bid_price is not None else None
            spread_iv = (row['ask_iv'] - row['bid_iv']) if row['ask_iv'] is not None and row[
                'bid_iv'] is not None else None

            args = {
                'symbol': row['symbol'],
                'timestamp': row['timestamp'],
                'strike_price': row['strike_price'],
                'expiration': row['expiration'],
                'open_interest': row['open_interest'],
                'last_price': row['last_price'],
                'bid_price': bid_price,
                'bid_amount': row['bid_amount'],
                'bid_iv': row['bid_iv'],
                'ask_price': ask_price,
                'ask_amount': row['ask_amount'],
                'ask_iv': row['ask_iv'],
                'mark_price': row['mark_price'],
                'mark_iv': row['mark_iv'],
                'spread': spread,
                'spread_iv': spread_iv,
                'underlying_price': row['underlying_price'],
                'delta': row['delta'],
                'gamma': row['gamma'],
                'theta': row['theta'],
                'vega': row['vega'],
                'rho': row['rho'],
            }
            if row['type'] == 'call':
                calls[row['symbol']] = Call(**args)
            else:
                puts[row['symbol']] = Put(**args)

        options_chain = OptionsChain(self._ts, calls, puts)

        current_row = self._perpetual_df.filter(pl.col('timestamp') == self._ts)

        __temp_askp = []
        __temp_bidp = []
        __temp_aska = []
        __temp_bida = []

        for i in range(1, 26):
            __temp_askp.append(current_row[f'ask{i}_price'].item())
            __temp_bidp.append(current_row[f'bid{i}_price'].item())
            __temp_aska.append(current_row[f'ask{i}_size'].item())
            __temp_bida.append(current_row[f'bid{i}_size'].item())

        features = current_row[:, 105:-9].to_numpy().flatten()
        future_information = {col: current_row[col].item() for col in current_row.columns[-9:]}

        perpetual = Perpetual(
            timestamp=self._ts,
            ask_prices=np.array(__temp_askp),
            bid_prices=np.array(__temp_bidp),
            ask_quantities=np.array(__temp_aska),
            bid_quantities=np.array(__temp_bida),
            features=features,
            mark_price=current_row['mark_price'].item(),
            funding_rate=current_row['funding_rate'].item(),
            future_information=future_information
        )

        tick = Tick(timestamp=self._ts, options_chain=options_chain, perpetual=perpetual,
                    volatility_tickers=volatility_tickers)

        return tick