from brokers import ZerodhaBroker
import pandas as pd

b = ZerodhaBroker()
b.connect()

holdings = b.get_holdings()

if not holdings:
    print("No holdings found.")
else:
    df = pd.DataFrame(holdings)[["tradingsymbol", "exchange", "quantity", "average_price", "last_price", "pnl"]]
    df.rename(columns={
        "tradingsymbol": "Symbol",
        "exchange": "Exch",
        "quantity": "Qty",
        "average_price": "Avg Price",
        "last_price": "LTP",
        "pnl": "PnL"
    }, inplace=True)
    print(df.to_string(index=False))