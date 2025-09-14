from __future__ import annotations

"""
refresh_kite_session.py — Daily reminder + interactive refresh of Zerodha (Kite) access_token.

What it does (single run):
 1) Sends a Telegram message with the Kite login URL at start.
 2) Waits for you to reply in Telegram with the `request_token` (or paste the whole redirect URL).
 3) Uses KiteConnect.generate_session(api_key, api_secret, request_token) to get a new access_token.
 4) Saves access_token to secrets/kite_session.json (overwrites existing).
 5) Sends a Telegram confirmation and prints logs locally.

Scheduling:
 - Run with cron/launchd at 05:00 IST (recommended), or use the built-in scheduler:
     python refresh_kite_session.py --schedule 05:00
 - Without --schedule it runs the flow once and exits.
"""

import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple
from urllib.parse import urlparse, parse_qs

import requests  # type: ignore


from ai_trading_bot.core.config import BASE_DIR

SESS_PATH = BASE_DIR / "secrets" / "kite_session.json"


def _load_env_file():
    try:
        env_path = BASE_DIR / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k and v and os.getenv(k) is None:
                    os.environ[k.strip()] = v.strip()
    except Exception:
        pass


def _env(key: str) -> Optional[str]:
    return os.getenv(key)


def _tg_base() -> Optional[str]:
    tok = _env("TELEGRAM_BOT_TOKEN")
    if not tok:
        return None
    return f"https://api.telegram.org/bot{tok}"


def _tg_chat() -> Optional[str]:
    return _env("TELEGRAM_CHAT_ID")


def _kite_creds() -> Tuple[str, str]:
    k = _env("KITE_API_KEY") or ""
    s = _env("KITE_API_SECRET") or ""
    return k, s


def send_tg_message(text: str) -> bool:
    base = _tg_base(); chat = _tg_chat()
    if not base or not chat:
        print("[telegram] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return False
    try:
        r = requests.post(f"{base}/sendMessage", json={"chat_id": chat, "text": text}, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"[telegram] send error: {e}")
        return False


def get_updates_offset() -> int:
    base = _tg_base()
    if not base:
        return 0
    try:
        r = requests.get(f"{base}/getUpdates", timeout=10)
        r.raise_for_status()
        data = r.json().get("result", [])
        return (max([u.get("update_id", 0) for u in data] + [0]) + 1) if data else 0
    except Exception:
        return 0


def wait_for_request_token(timeout_sec: int = 900) -> Optional[str]:
    base = _tg_base(); chat = _tg_chat()
    if not base or not chat:
        print("[telegram] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return None
    try:
        offset = get_updates_offset()
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            r = requests.get(f"{base}/getUpdates", params={"timeout": 10, "offset": offset}, timeout=15)
            r.raise_for_status()
            res = r.json().get("result", [])
            if not res:
                continue
            for upd in res:
                offset = max(offset, upd.get("update_id", offset) + 1)
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue
                if str(msg.get("chat", {}).get("id")) != str(chat):
                    continue
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                token = extract_request_token(text)
                if token:
                    return token
        return None
    except Exception as e:
        print(f"[telegram] poll error: {e}")
        return None


def extract_request_token(text: str) -> Optional[str]:
    # Accept either a raw token or a full redirect URL containing request_token param.
    text = text.strip()
    # Try URL parse
    try:
        pr = urlparse(text)
        if pr.scheme and pr.netloc:
            qs = parse_qs(pr.query)
            rt = qs.get("request_token", [None])[0]
            if rt:
                return rt
    except Exception:
        pass
    # Fallback: token-like string (alphanumeric ~32+ chars)
    if 16 <= len(text) <= 64 and text.replace("_", "").replace("-", "").isalnum():
        return text
    return None


def obtain_login_url(api_key: str) -> Optional[str]:
    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=api_key)
        return kite.login_url()
    except Exception as e:
        print(f"[kite] cannot build login URL: {e}")
        return None


def generate_session(api_key: str, api_secret: str, request_token: str) -> Optional[str]:
    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=api_key)
        data = kite.generate_session(request_token, api_secret=api_secret)
        return data.get("access_token")
    except Exception as e:
        print(f"[kite] generate_session failed: {e}")
        return None


def save_access_token(token: str) -> None:
    SESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESS_PATH.write_text(json.dumps({"access_token": token}, indent=2))


def once_flow(timeout_sec: int = 900) -> int:
    api_key, api_secret = _kite_creds()
    if not api_key or not api_secret:
        print("Missing KITE_API_KEY or KITE_API_SECRET in environment/.env")
        return 2

    url = obtain_login_url(api_key)
    if not url:
        print("Could not generate login URL.")
        return 3

    send_tg_message(
        "Zerodha token refresh — please log in and reply with request_token:\n" + url
    )
    print("Sent Telegram reminder with login URL. Waiting for request_token reply…")

    token = wait_for_request_token(timeout_sec=timeout_sec)
    if not token:
        print("Timed out waiting for request_token. Try again later.")
        send_tg_message("Token refresh timed out. Please rerun the refresh script when ready.")
        return 4

    access = generate_session(api_key, api_secret, token)
    if not access:
        send_tg_message("Token refresh failed. Invalid request_token or API error.")
        return 5

    save_access_token(access)
    send_tg_message("Access token updated ✅. You are good for today’s trading session.")
    print(f"Saved new access_token to {SESS_PATH}")
    return 0


def sleep_until(hhmm: str, tz: str = "Asia/Kolkata") -> None:
    try:
        import pytz
        tzinfo = pytz.timezone(tz)
    except Exception:
        tzinfo = None
    h, m = hhmm.split(":")
    h = int(h); m = int(m)
    while True:
        now = datetime.utcnow() if tzinfo is None else datetime.now(tzinfo)
        target = (now.replace(hour=h, minute=m, second=0, microsecond=0))
        if target <= now:
            target = target + timedelta(days=1)
        wait = (target - now).total_seconds()
        if wait > 0:
            time.sleep(min(wait, 3600))
        else:
            break
        # loop until within 60s
        if wait <= 60:
            time.sleep(max(1, int(wait)))
            break


def main() -> int:
    _load_env_file()
    ap = argparse.ArgumentParser(description="Daily Zerodha token refresh via Telegram")
    ap.add_argument("--schedule", default=None, help="Run daily at HH:MM (e.g., 05:00) in Asia/Kolkata")
    ap.add_argument("--timeout", type=int, default=900, help="Seconds to wait for request_token reply")
    args = ap.parse_args()

    if args.schedule:
        print(f"Scheduling daily token refresh at {args.schedule} IST… (Ctrl+C to stop)")
        while True:
            sleep_until(args.schedule)
            try:
                once_flow(timeout_sec=args.timeout)
            except Exception:
                # keep scheduler alive
                pass
            # brief pause before scheduling next
            time.sleep(5)
    else:
        return once_flow(timeout_sec=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

