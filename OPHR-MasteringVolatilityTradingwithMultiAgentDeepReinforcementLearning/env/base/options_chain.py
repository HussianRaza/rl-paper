import abc
import dataclasses
import datetime
from typing import Dict
from env.base.call import Call
from env.base.put import Put


@dataclasses.dataclass
class OptionsChain(abc.ABC):
    """This class defines the basic type for the backtester or live trader -- an option chain snapshot."""
    timestamp: datetime.datetime
    calls: Dict[str, Call]
    puts: Dict[str, Put]
