import abc
import dataclasses
import datetime
from decimal import Decimal


@dataclasses.dataclass
class Trade(abc.ABC):
    timestamp: datetime.datetime
    size: Decimal
