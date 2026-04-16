from decimal import Decimal
from typing import Optional

from hedgers.base_hedger import BaseHedger


class PriceMoveHedger(BaseHedger):
    def __init__(self, price_move_threshold: float = 0.02, hedge_ratio: float = 1.0):
        super().__init__({
            'price_move_threshold': price_move_threshold,
            'hedge_ratio': hedge_ratio,
        })
        self.price_move_threshold = Decimal(str(price_move_threshold))
        self.hedge_ratio = Decimal(str(hedge_ratio))
        self._last_ref_price: Optional[Decimal] = None

    def reset(self):
        self._last_ref_price = None

    def _should_hedge(self, current_price: Decimal) -> bool:
        if self._last_ref_price is None:
            self._last_ref_price = current_price
            return False
        if self._last_ref_price <= 0 or current_price <= 0:
            return False
        move = abs(current_price - self._last_ref_price) / self._last_ref_price
        return move > self.price_move_threshold

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
        mark_price = market_info.get('mark_price', None)
        if mark_price is None:
            return Decimal('0')
        if not isinstance(mark_price, Decimal):
            mark_price = Decimal(str(mark_price))

        if self._should_hedge(mark_price):
            hedge_amount = -delta * self.hedge_ratio
            self._last_ref_price = mark_price
            return hedge_amount
        return Decimal('0')

    def __repr__(self):
        pct = (self.price_move_threshold * Decimal('100')).quantize(Decimal('0.01'))
        return f"PriceMoveHedger(threshold={pct}%)"
