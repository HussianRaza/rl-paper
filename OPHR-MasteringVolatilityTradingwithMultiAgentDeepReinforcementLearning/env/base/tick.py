import abc
import dataclasses
import datetime
from typing import Union
from env.base.options_chain import OptionsChain
from env.base.perpetual import Perpetual, PerpetualOHLCV
from env.base.volatility_tickers import VolatilityTickers


@dataclasses.dataclass
class Tick(abc.ABC):
    timestamp: datetime.datetime
    perpetual: Union[Perpetual, PerpetualOHLCV]
    options_chain: OptionsChain
    volatility_tickers: VolatilityTickers
    open: bool
