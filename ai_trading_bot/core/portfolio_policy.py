from __future__ import annotations

from typing import Dict, Any
from ai_trading_bot.core.config import SELL_BEHAVIOR, ALLOW_SHORTS, ALLOW_SHORT_WHEN_LONG


def decide_action(signal: Dict[str, Any], holding: Dict[str, Any] | None) -> Dict[str, Any]:
    """Decide EXIT/SHORT/IGNORE/BUY given a signal row and an optional holding.

    signal keys expected: ticker, signal (BUY/SELL), recommended_product, recommended_reason, proba_up, rr
    holding keys expected (optional): quantity, product, average_price
    """
    side = str(signal.get("signal") or "").upper()
    product = str(signal.get("recommended_product") or ("MIS" if side == "SELL" else "CNC"))
    reason_parts: list[str] = []

    owns = bool(holding and int(holding.get("quantity") or 0) > 0)

    if side == "SELL":
        if owns:
            # We hold a long position
            reason_parts.append("holding_long→exit")
            if SELL_BEHAVIOR == "exit_only" or SELL_BEHAVIOR == "exit_or_ignore":
                return {"action": "EXIT", "product": "CNC", "reason": ", ".join(reason_parts)}
            if SELL_BEHAVIOR == "exit_then_short":
                if not ALLOW_SHORTS or not (ALLOW_SHORT_WHEN_LONG is False):
                    # policy says only exit
                    return {"action": "EXIT", "product": "CNC", "reason": ", ".join(reason_parts)}
                # Exit first; caller may enqueue SHORT after exit if desired
                return {"action": "EXIT", "product": "CNC", "reason": ", ".join(reason_parts) + "; short_after_exit"}
            if SELL_BEHAVIOR == "short_only":
                # avoid hedging same scrip
                if not ALLOW_SHORT_WHEN_LONG:
                    return {"action": "EXIT", "product": "CNC", "reason": ", ".join(reason_parts)}
                return {"action": "SHORT", "product": "MIS", "reason": ", ".join(reason_parts)}
        else:
            # No holding
            if ALLOW_SHORTS and SELL_BEHAVIOR in ("short_only", "exit_then_short"):
                return {"action": "SHORT", "product": "MIS", "reason": "no_holding→short_ok"}
            return {"action": "IGNORE", "product": product, "reason": "policy_veto"}

    # BUY or others: default to BUY with the product recommender
    return {"action": "BUY", "product": product, "reason": signal.get("recommended_reason")}

