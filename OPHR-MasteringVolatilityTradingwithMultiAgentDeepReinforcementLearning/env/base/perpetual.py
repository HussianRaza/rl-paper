import abc
import dataclasses
import datetime
import numpy as np
import decimal
from typing import Optional, Text, Dict


@dataclasses.dataclass
class Perpetual(abc.ABC):
    timestamp: datetime.datetime
    ask_prices: np.ndarray
    bid_prices: np.ndarray
    ask_quantities: np.ndarray
    bid_quantities: np.ndarray
    mark_price: decimal.Decimal
    funding_rate : decimal.Decimal
    features: np.ndarray
    future_information: Dict


@dataclasses.dataclass
class PerpetualOHLCV(abc.ABC):
    timestamp: datetime.datetime
    open: decimal.Decimal
    high: decimal.Decimal
    low: decimal.Decimal
    close: decimal.Decimal
    volume: int
    funding_rate : decimal.Decimal
    mark_price : decimal.Decimal
