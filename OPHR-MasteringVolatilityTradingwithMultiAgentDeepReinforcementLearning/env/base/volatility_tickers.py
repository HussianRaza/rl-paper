import abc
import dataclasses
import datetime
import decimal
import numpy as np

@dataclasses.dataclass
class VolatilityTickers(abc.ABC):
    timestamp: datetime.datetime
    min_mark_iv: decimal.Decimal
    min_mark_iv_delta: decimal.Decimal
    features: np.array