from __future__ import annotations

"""
notify.py — Lightweight Telegram approval helper using long-polling.

Environment variables (or .env in project root):
  TELEGRAM_BOT_TOKEN=<token>
  TELEGRAM_CHAT_ID=<your chat id>

Usage:
  ok = request_approval_via_telegram(
          title="Place Order",
          text="BUY INFY x10 @ ~1500, SL=1450, TP=1550",
          timeout=120,
      )

Returns True on Approve, False on Reject/timeout/error.
"""

import os
import time
import json
from typing import Optional
from pathlib import Path

import requests  # type: ignore

from ai_trading_bot.core.config import BASE_DIR


def _env(key: str) -> Optional[str]:
    # Load .env lazily if needed
    val = os.getenv(key)
    if val:
        return val
    try:
        env_path = BASE_DIR / "environment" / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.strip()
    except Exception:
        pass
    return None


def _get_base_url() -> Optional[str]:
    token = _env("TELEGRAM_BOT_TOKEN")
    if not token:
        return None
    return f"https://api.telegram.org/bot{token}"


def _get_chat_id() -> Optional[str]:
    return _env("TELEGRAM_CHAT_ID")


def _telegram_available() -> bool:
    return _get_base_url() is not None and _get_chat_id() is not None


def request_approval_via_telegram(title: str, text: str, timeout: int = 120) -> bool:
    """Send an approval request with inline buttons; wait for a response.

    Returns True if approved, False otherwise (reject/timeout/error/unavailable).
    """
    base = _get_base_url()
    chat_id = _get_chat_id()
    if not base or not chat_id:
        return False

    try:
        # Determine current offset to avoid old updates
        r = requests.get(f"{base}/getUpdates", timeout=10)
        r.raise_for_status()
        data = r.json()
        last_update_id = max([u.get("update_id", 0) for u in data.get("result", [])] + [0])
        offset = last_update_id + 1

        unique = str(int(time.time() * 1000))
        msg = f"{title}\n\n{text}"
        keyboard = {
            "inline_keyboard": [[
                {"text": "Approve ✅", "callback_data": f"APPROVE:{unique}"},
                {"text": "Reject ❌", "callback_data": f"REJECT:{unique}"},
            ]]
        }

        sr = requests.post(
            f"{base}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "reply_markup": keyboard, "parse_mode": "HTML"},
            timeout=10,
        )
        sr.raise_for_status()

        # Poll for callback
        t_end = time.time() + max(10, int(timeout))
        while time.time() < t_end:
            ur = requests.get(f"{base}/getUpdates", params={"timeout": 10, "offset": offset}, timeout=15)
            ur.raise_for_status()
            res = ur.json().get("result", [])
            if not res:
                continue
            for upd in res:
                offset = max(offset, upd.get("update_id", offset) + 1)
                cq = upd.get("callback_query")
                if not cq:
                    continue
                data = cq.get("data", "")
                if data.endswith(f":{unique}"):
                    # Acknowledge callback to Telegram
                    try:
                        requests.post(f"{base}/answerCallbackQuery", json={"callback_query_id": cq.get("id"), "text": "Received"}, timeout=10)
                    except Exception:
                        pass
                    approved = data.startswith("APPROVE:")
                    # Send a visible chat confirmation
                    try:
                        ack = "Approved ✅" if approved else "Rejected ❌"
                        requests.post(
                            f"{base}/sendMessage",
                            json={"chat_id": chat_id, "text": f"Thanks. {ack}\n\n{text}", "parse_mode": "HTML"},
                            timeout=10,
                        )
                    except Exception:
                        pass
                    # Optionally, remove buttons by editing reply markup to empty
                    try:
                        msg = cq.get("message", {})
                        mid = msg.get("message_id")
                        if mid:
                            requests.post(
                                f"{base}/editMessageReplyMarkup",
                                json={"chat_id": chat_id, "message_id": mid, "reply_markup": {"inline_keyboard": []}},
                                timeout=10,
                            )
                    except Exception:
                        pass
                    return approved
        return False
    except Exception:
        return False


def request_approval(title: str, text: str, timeout: int = 120) -> bool:
    """Wrapper: tries Telegram; returns False if not approved in time."""
    return request_approval_via_telegram(title, text, timeout=timeout)


def send_message(text: str) -> bool:
    """Send a plain text message to Telegram chat. Returns True on success."""
    base = _get_base_url()
    chat_id = _get_chat_id()
    if not base or not chat_id:
        return False
    try:
        r = requests.post(
            f"{base}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        r.raise_for_status()
        return True
    except Exception:
        return False
