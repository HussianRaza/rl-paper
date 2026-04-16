import dataclasses
from env.base import option


@dataclasses.dataclass
class Call(option.Option):
    """This class defines a CALL option, which inherits from the Option class."""
    type: option.OptionTypes = option.OptionTypes.CALL
