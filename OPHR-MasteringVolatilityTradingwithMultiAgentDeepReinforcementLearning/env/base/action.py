import abc
import dataclasses
import datetime
from typing import Dict
import decimal
from typing import Optional, Text
from env.base.call import Call
from env.base.put import Put
from env.base.option import Option


@dataclasses.dataclass
class Action(abc.ABC):
    timestamp: datetime.datetime


@dataclasses.dataclass
class OptionAction(Action):
    trades: Dict[str, decimal.Decimal]


@dataclasses.dataclass
class HedgeAction(Action):
    quantity: decimal.Decimal

