import abc
import dataclasses
import datetime
import logging
from decimal import Decimal
import pandas as pd
from typing import List

log_format = '%(asctime)s - %(filename)s - Line: %(lineno)d - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Log(abc.ABC):
    def __init__(self, timestamp: datetime.datetime):
        self._timestamp = timestamp
        self._step_daily = 0
        self._exercise_pnl = 0
        self._realized_pnl = 0
        self._fee = 0
        self.hedges = []
        self.option_fee = Decimal('0')
        self.hedge_fee = Decimal('0')
        
        self.exercise_history = pd.DataFrame(columns=[
            'time',
            'symbol',
            'side',
            'amount',
            'opening_cash_flow',
            'settlement_cash_flow',
            'net_cash_flow',
            'settlement_price',
            'fees'
        ])

        self.account_history = pd.DataFrame(columns=[
            'timestamp',
            'cash_balance',
            'unrealized_pnl',
            'realized_pnl',
            'total_value',
        ])

        self.order_history = pd.DataFrame(columns=[
            'timestamp',
            'symbol',
            'instrument_type',
            'order_type',
            'side',
            'quantity',
            'price',
            'status'
        ])

        self.trade_history = pd.DataFrame(columns=[
            'timestamp',
            'symbol',
            'instrument_type',
            'side',
            'quantity',
            'price',
            'fee',
            'trade_id'
        ])

        self.position_history = pd.DataFrame(columns=[
            'symbol',
            'open_time',
            'average_entry_price',
            'average_closed_price',
            'quantity',
            'position_pnl',
            'position_roi',
            'closed_time',
            'side'
        ])

        self.value_history = pd.DataFrame(columns=[
            'timestamp',
            'net_value',
            'mark_price'
        ])

    def add_exercise_record(self, timestamp: datetime.datetime, symbol: str, side: str,
                          amount: Decimal, opening_cf: Decimal, settlement_cf: Decimal,
                          net_cf: Decimal, settlement_price: Decimal, fees: Decimal):
        new_row = pd.DataFrame([{
            'time': pd.Timestamp(timestamp),
            'symbol': str(symbol),
            'side': str(side),
            'amount': float(amount),
            'opening_cash_flow': float(opening_cf),
            'settlement_cash_flow': float(settlement_cf),
            'net_cash_flow': float(net_cf),
            'settlement_price': float(settlement_price),
            'fees': float(fees)
        }])
        # 确保数据类型一致
        for col in self.exercise_history.columns:
            new_row[col] = new_row[col].astype(self.exercise_history[col].dtype)
        self.exercise_history = pd.concat([self.exercise_history, new_row], ignore_index=True)

    def add_account_record(self, timestamp: datetime.datetime, cash_balance: Decimal,
                         unrealized_pnl: Decimal, realized_pnl: Decimal, fee: Decimal, total_value: Decimal):
        new_row = pd.DataFrame([{
            'timestamp': pd.Timestamp(timestamp),
            'cash_balance': float(cash_balance),
            'unrealized_pnl': float(unrealized_pnl),
            'realized_pnl': float(realized_pnl),
            'total_value': float(total_value)
        }])
        # 确保数据类型一致
        for col in self.account_history.columns:
            new_row[col] = new_row[col].astype(self.account_history[col].dtype)
        self.account_history = pd.concat([self.account_history, new_row], ignore_index=True)

    def add_order_record(self, timestamp: datetime.datetime, symbol: str,
                       instrument_type: str, order_type: str, side: str,
                       quantity: Decimal, price: Decimal, status: str):
        new_row = pd.DataFrame([{
            'timestamp': pd.Timestamp(timestamp),
            'symbol': str(symbol),
            'instrument_type': str(instrument_type),
            'order_type': str(order_type),
            'side': str(side),
            'quantity': float(quantity),
            'price': float(price),
            'status': str(status)
        }])
        # 确保数据类型一致
        for col in self.order_history.columns:
            new_row[col] = new_row[col].astype(self.order_history[col].dtype)
        self.order_history = pd.concat([self.order_history, new_row], ignore_index=True)

    def add_trade_record(self, timestamp: datetime.datetime, symbol: str,
                       instrument_type: str, side: str, quantity: Decimal,
                       price: Decimal, fee: Decimal, trade_id: str):
        new_row = pd.DataFrame([{
            'timestamp': pd.Timestamp(timestamp),
            'symbol': str(symbol),
            'instrument_type': str(instrument_type),
            'side': str(side),
            'quantity': float(quantity),
            'price': float(price),
            'fee': float(fee),
            'trade_id': str(trade_id)
        }])
        # 确保数据类型一致
        for col in self.trade_history.columns:
            new_row[col] = new_row[col].astype(self.trade_history[col].dtype)
        self.trade_history = pd.concat([self.trade_history, new_row], ignore_index=True)

    def add_position_record(self, position):
        new_row = pd.DataFrame([{
            'symbol': str(position.symbol),
            'open_time': pd.Timestamp(position.open_time) if position.open_time else None,
            'average_entry_price': float(position.average_entry_price) if position.average_entry_price else None,
            'average_closed_price': float(position.average_closed_price) if position.average_closed_price else None,
            'quantity': float(position.quantity),
            'position_pnl': float(position.position_pnl) if position.position_pnl else 0,
            'position_roi': float(position.position_roi) if position.position_roi else 0,
            'closed_time': pd.Timestamp(position.closed_time) if position.closed_time else None,
            'side': str(position.side)
        }])
        # 确保数据类型一致
        for col in self.position_history.columns:
            new_row[col] = new_row[col].astype(self.position_history[col].dtype)
        self.position_history = pd.concat([self.position_history, new_row], ignore_index=True)

    def add_value_record(self, timestamp: datetime.datetime, net_value: Decimal, mark_price: Decimal):
        """记录净值和标记价格历史"""
        new_row = pd.DataFrame([{
            'timestamp': pd.Timestamp(timestamp),
            'net_value': float(net_value),
            'mark_price': float(mark_price)
        }])
        # 确保数据类型一致
        for col in self.value_history.columns:
            new_row[col] = new_row[col].astype(self.value_history[col].dtype)
        self.value_history = pd.concat([self.value_history, new_row], ignore_index=True)

    @property
    def net_value_history(self):
        """兼容旧代码的属性访问方式"""
        return self.value_history['net_value'].tolist()

    @property
    def mark_price_history(self):
        """兼容旧代码的属性访问方式"""
        return self.value_history['mark_price'].tolist()

    def get_net_value_change(self) -> float:
        """安全地获取净值变化
        如果历史记录不足两条，返回0
        """
        if len(self.value_history) < 2:
            return 0.0
        return float(self.value_history.iloc[-1]['net_value'] - self.value_history.iloc[-2]['net_value'])

    def get_mark_price_change(self) -> float:
        """安全地获取标记价格变化
        如果历史记录不足两条，返回0
        """
        if len(self.value_history) < 2:
            return 0.0
        return float(self.value_history.iloc[-1]['mark_price'] - self.value_history.iloc[-2]['mark_price'])

