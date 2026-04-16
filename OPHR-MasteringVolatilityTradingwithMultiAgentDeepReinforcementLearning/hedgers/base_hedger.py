from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Tuple

class BaseHedger(ABC):
    
    def __init__(self, config: dict):
        self.config = config
    
    @abstractmethod
    def compute_hedge(
        self,
        delta: Decimal,
        gamma: Decimal,
        theta: Decimal,
        vega: Decimal,
        position_info: dict,
        market_info: dict
    ) -> Decimal:
        pass
    
    def __call__(
        self,
        delta: Decimal,
        gamma: Decimal,
        theta: Decimal,
        vega: Decimal,
        position_info: dict = None,
        market_info: dict = None
    ) -> Decimal:
        if position_info is None:
            position_info = {}
        if market_info is None:
            market_info = {}
            
        return self.compute_hedge(
            delta, gamma, theta, vega,
            position_info, market_info
        )
    
    def reset(self):
        pass



