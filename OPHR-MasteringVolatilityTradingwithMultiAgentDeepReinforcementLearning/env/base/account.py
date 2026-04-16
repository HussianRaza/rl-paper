import abc
import dataclasses
import datetime
from decimal import Decimal


@dataclasses.dataclass
class Account(abc.ABC):
    timestamp: datetime.datetime
    cash_balance: Decimal
    net_value: Decimal
    liquidation_value: Decimal

    # Added for S:PM mode
    initial_margin: Decimal = Decimal('0')
    maintenance_margin: Decimal = Decimal('0')
    im_percentage = Decimal('0')
    mm_percentage = Decimal('0')

    def __init__(self, timestamp: datetime.datetime, cash_balance: Decimal):
        self.timestamp = timestamp
        self.cash_balance = cash_balance
        self.net_value = cash_balance
        self.liquidation_value = cash_balance
