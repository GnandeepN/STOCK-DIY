from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from ai_trading_bot.core.config import REPORTS_DIR

SNAP_DIR = REPORTS_DIR / "portfolio_snapshots"


def build_equity_curve() -> pd.DataFrame:
    files = sorted(SNAP_DIR.glob("snapshot_*.csv"))
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
            total = float((df["value"].fillna(0)).sum())
            date = f.stem.replace("snapshot_", "")
            rows.append(dict(date=date, equity=total))
        except Exception:
            pass
    curve = pd.DataFrame(rows).sort_values("date")
    return curve


def save_chart(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(pd.to_datetime(df["date"]), df["equity"], marker="o")
    plt.title("Portfolio Equity")
    plt.xlabel("Date")
    plt.ylabel("Equity (₹)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)


def main() -> int:
    df = build_equity_curve()
    out = REPORTS_DIR / "equity_curve.png"
    save_chart(df, out)
    print(f"Saved chart → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

