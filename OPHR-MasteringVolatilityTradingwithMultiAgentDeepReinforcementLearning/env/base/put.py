import dataclasses
from env.base import option


@dataclasses.dataclass
class Put(option.Option):
    """This class defines a PUT option, which inherits from the Option class."""
    type: option.OptionTypes = option.OptionTypes.PUT
