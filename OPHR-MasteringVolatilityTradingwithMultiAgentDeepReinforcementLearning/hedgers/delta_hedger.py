from decimal import Decimal
from hedgers.base_hedger import BaseHedger


class DeltaThresholdHedger(BaseHedger):
    def __init__(self, delta_threshold: float = 0.1, hedge_ratio: float = 1.0):
        config = {
            'delta_threshold': delta_threshold,
            'hedge_ratio': hedge_ratio
        }
        super().__init__(config)
        
        self.delta_threshold = Decimal(str(delta_threshold))
        self.hedge_ratio = Decimal(str(hedge_ratio))
    
    def compute_hedge(
        self,
        delta: Decimal,
        gamma: Decimal,
        theta: Decimal,
        vega: Decimal,
        position_info: dict,
        market_info: dict
    ) -> Decimal:
        if not isinstance(delta, Decimal):
            delta = Decimal(str(delta))

        abs_delta = abs(delta)
        
        if abs_delta > self.delta_threshold:
            hedge_amount = -delta * self.hedge_ratio
            return hedge_amount
        else:
            return Decimal('0')
    
    def __repr__(self):
        return f"DeltaThresholdHedger(threshold={self.delta_threshold}, ratio={self.hedge_ratio})"


class BaselineHedger(DeltaThresholdHedger):
    def __init__(self):
        super().__init__(delta_threshold=0.1, hedge_ratio=1.0)

