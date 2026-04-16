import abc
import dataclasses
import datetime
import decimal
import enum
import math
from typing import Optional, Text
import numpy as np


class OptionTypes(enum.Enum):
    PUT = 0
    CALL = 1


@dataclasses.dataclass
class Option(abc.ABC):
    symbol: Text
    timestamp: datetime.datetime
    strike_price: decimal.Decimal
    expiration: datetime.datetime
    open_interest: int

    last_price: decimal.Decimal
    bid_price: decimal.Decimal
    bid_amount: decimal.Decimal
    bid_iv: decimal.Decimal
    ask_price: decimal.Decimal
    ask_amount: decimal.Decimal
    ask_iv: decimal.Decimal
    mark_price: decimal.Decimal
    mark_iv: decimal.Decimal
    spread: decimal.Decimal
    spread_iv: decimal.Decimal

    underlying_price: decimal.Decimal
    delta: decimal.Decimal
    theta: decimal.Decimal
    gamma: decimal.Decimal
    rho: decimal.Decimal
    vega: decimal.Decimal

    def __post_init__(self):
        if self.__class__ == Option:
            raise TypeError('Cannot instantiate abstract class.')

    def time_to_expiration(self) -> float:
        """Calculate the time to expiration of the option in days."""
        time_to_expiration = self.expiration - self.timestamp
        return time_to_expiration.total_seconds() / (60 * 60 * 24)

    def log_moneyness(self) -> float:
        """Calculate the log moneyness of the option."""
        return math.log(self.underlying_price / self.strike_price)

    def getNumDaysLeft(self) -> int:
        """Determine the number of days between the current date/time and expiration date / time.
          :return: number of days between curDateTime and expDateTime.
        """
        return (self.expiration - self.timestamp).days

    def getMidPrice(self) -> decimal.Decimal:
        """Calculate the mid price for the option."""
        return (self.bid_price + self.ask_price) / decimal.Decimal(2.0)

    # TODO: fix features
    def get_features(self) -> np.array:
        return np.array([self.open_interest, self.last_price, self.bid_price, self.bid_amount, self.bid_iv, self.ask_price, self.ask_amount, self.ask_iv, self.mark_price, self.mark_iv, self.spread, self.spread_iv, self.underlying_price, self.delta, self.theta, self.gamma, self.rho, self.vega, self.position])