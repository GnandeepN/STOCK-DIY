from __future__ import annotations

from typing import Dict, Any, Optional
from dataclasses import dataclass
from .base import BrokerBase, OrderRequest, OrderResult


@dataclass
class PaperBroker(BrokerBase):
    """A simple in-memory paper broker that immediately 'fills' market orders."""

    def __init__(self) -> None:
        super().__init__()
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._id = 0

    def connect(self) -> None:
        return

    def quote(self, exchange: str, tradingsymbol: str) -> Dict[str, Any]:
        # Not a real quote; return empty
        return {}

    def margins(self) -> Dict[str, Any]:
        return {}

    def place_order(self, req: OrderRequest) -> OrderResult:
        self._id += 1
        oid = str(self._id)
        self._orders[oid] = {
            "symbol": req.symbol,
            "side": req.side,
            "quantity": req.quantity,
            "status": "COMPLETE",
        }
        return OrderResult(ok=True, order_id=oid, raw={"order_id": oid})

    def cancel_order(self, order_id: str) -> bool:
        return True

