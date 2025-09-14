from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class OrderRequest:
    symbol: str            # e.g., NSE:INFY
    side: str              # BUY or SELL
    quantity: int
    price: Optional[float] = None  # None for market order
    product: str = "CNC"           # CNC (delivery) or MIS (intraday)
    variety: str = "regular"
    order_type: str = "MARKET"     # MARKET or LIMIT
    tag: Optional[str] = None


@dataclass
class OrderResult:
    ok: bool
    order_id: Optional[str]
    raw: Dict[str, Any]
    error: Optional[str] = None


class BrokerBase:
    def __init__(self) -> None:
        pass

    def connect(self) -> None:
        raise NotImplementedError

    def quote(self, exchange: str, tradingsymbol: str) -> Dict[str, Any]:
        raise NotImplementedError

    def margins(self) -> Dict[str, Any]:
        raise NotImplementedError

    def place_order(self, req: OrderRequest) -> OrderResult:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

