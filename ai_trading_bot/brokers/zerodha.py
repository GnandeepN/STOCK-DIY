from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from .base import BrokerBase, OrderRequest, OrderResult


class ZerodhaBroker(BrokerBase):
    """Light wrapper over KiteConnect (Zerodha) with minimal deps.

    Requires: pip install kiteconnect
    Credentials: set env vars KITE_API_KEY, KITE_API_SECRET or provide in ctor.
    Session persistence: secrets/kite_session.json (stores access_token).
    """

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 session_path: Optional[Path] = None) -> None:
        super().__init__()

        # Attempt to load .env if env vars are missing
        def _load_env_file():
            try:
                # Always load from project root: environment/.env
                base_dir = Path(__file__).resolve().parent.parent.parent
                env_path = base_dir / "environment" / ".env"
                if env_path.exists():
                    for line in env_path.read_text().splitlines():
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip(); v = v.strip()
                        if k and v and os.getenv(k) is None:
                            os.environ[k] = v
            except Exception:
                pass

        if not (os.getenv("KITE_API_KEY") and os.getenv("KITE_API_SECRET")):
            _load_env_file()

        self.api_key = api_key or os.getenv("KITE_API_KEY", "").strip()
        self.api_secret = api_secret or os.getenv("KITE_API_SECRET", "").strip()
        self.session_path = Path(session_path) if session_path else Path(__file__).resolve().parent.parent / "secrets" / "kite_session.json"
        self.kite = None

    # --- session utils ---
    def _load_session(self) -> Optional[str]:
        try:
            if self.session_path.exists():
                data = json.loads(self.session_path.read_text())
                return data.get("access_token")
        except Exception:
            pass
        return None

    def _save_session(self, access_token: str) -> None:
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_path.write_text(json.dumps({"access_token": access_token}, indent=2))

    # --- public API ---
    def connect(self) -> None:
        try:
            from kiteconnect import KiteConnect
        except Exception as e:  # pragma: no cover
            raise RuntimeError("kiteconnect is not installed. `pip install kiteconnect`.") from e

        self.kite = KiteConnect(api_key=self.api_key)
        token = self._load_session()
        if token:
            self.kite.set_access_token(token)
            return
        # First-time: user must obtain request_token via login URL and call obtain_access_token()
        login_url = self.kite.login_url()
        raise RuntimeError(f"No access_token. Visit login URL to get request_token: {login_url}\n"
                           f"Then call ZerodhaBroker.obtain_access_token(request_token) once.")

    def obtain_access_token(self, request_token: str) -> str:
        if not self.kite:
            from kiteconnect import KiteConnect
            self.kite = KiteConnect(api_key=self.api_key)
        data = self.kite.generate_session(request_token, api_secret=self.api_secret)
        access_token = data["access_token"]
        self.kite.set_access_token(access_token)
        self._save_session(access_token)
        return access_token

    def quote(self, exchange: str, tradingsymbol: str) -> Dict[str, Any]:
        assert self.kite is not None
        inst = f"{exchange}:{tradingsymbol}"
        q = self.kite.quote([inst])
        return q.get(inst, {})

    def margins(self) -> Dict[str, Any]:
        assert self.kite is not None
        return self.kite.margins(segment="equity")

    def place_order(self, req: OrderRequest) -> OrderResult:
        assert self.kite is not None
        exch, tsym = req.symbol.split(":", 1) if ":" in req.symbol else ("NSE", req.symbol)
        try:
            order_id = self.kite.place_order(
                variety=req.variety,
                exchange=exch,
                tradingsymbol=tsym,
                transaction_type=req.side.upper(),
                quantity=int(req.quantity),
                product=req.product,
                order_type=req.order_type.upper(),
                price=req.price if req.order_type.upper() == "LIMIT" else None,
                validity="DAY",
                tag=req.tag or "ai_trading_bot",
            )
            return OrderResult(ok=True, order_id=str(order_id), raw={"order_id": order_id})
        except Exception as e:
            return OrderResult(ok=False, order_id=None, raw={}, error=str(e))

    def cancel_order(self, order_id: str) -> bool:
        assert self.kite is not None
        try:
            self.kite.cancel_order(variety="regular", order_id=order_id)
            return True
        except Exception:
            return False

    # --- 🔹 new methods ---
    def get_positions(self) -> Dict[str, Any]:
        """Fetch current positions (intraday + overnight)."""
        assert self.kite is not None
        return self.kite.positions()

    def get_holdings(self) -> List[Dict[str, Any]]:
        """Fetch demat holdings (long-term CNC investments)."""
        assert self.kite is not None
        return self.kite.holdings()
