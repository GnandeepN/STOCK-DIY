"""
tickers_custom.py
Your own hand‑picked watchlist. These are merged with other lists in config.py
and de‑duplicated automatically. Feel free to add/remove tickers here.
"""

# Expanded custom list (NSE suffix .NS)
CORE_TICKERS = [
    # Large cap diversified
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS",

    # Tech and IT services
    "HCLTECH.NS", "TECHM.NS", "WIPRO.NS",

    # Consumer and staples
    "HINDUNILVR.NS", "ASIANPAINT.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TITAN.NS",

    # Autos
    "MARUTI.NS", "M&M.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",

    # Financials
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "ICICIPRULI.NS", "BANKBARODA.NS", "PNB.NS",

    # Pharma & healthcare
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",

    # Metals & energy & infra
    "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "HINDALCO.NS",
    "ADANIPORTS.NS", "ADANIENT.NS", "ONGC.NS", "BPCL.NS",

    # Materials & others
    "ULTRACEMCO.NS", "GRASIM.NS", "UPL.NS", "TATACONSUM.NS",
]

# Midcap additions
MIDCAP_TICKERS = [
    # IT & Engineering services
    "COFORGE.NS", "MPHASIS.NS", "PERSISTENT.NS", "LTIM.NS", "TATAELXSI.NS",
    # Industrials & manufacturing
    "POLYCAB.NS", "DIXON.NS", "ASTRAL.NS", "BALKRISIND.NS", "VOLTAS.NS",
    # Pharma & chemicals
    "DEEPAKNTR.NS", "LAURUSLABS.NS", "AUROPHARMA.NS", "LUPIN.NS", "TORNTPHARM.NS",
    # Financials & others
    "MUTHOOTFIN.NS", "CANFINHOME.NS", "INDHOTEL.NS", "ZYDUSLIFE.NS", "TATAPOWER.NS",
    # New curated midcaps (only if not already present)
    "SRF.NS", "PIIND.NS", "IRCTC.NS", "CONCOR.NS",
    "IEX.NS", "NAVINFLUOR.NS", "AIAENG.NS", "AFFLE.NS", "METROPOLIS.NS",
    "GLAND.NS", "GUJGASLTD.NS", "CROMPTON.NS", "PVRINOX.NS", "TIINDIA.NS",
    "NYKAA.NS", "LODHA.NS",
    # User additions (validated NSE/Yahoo symbols)
    "HDFCAMC.NS", "MAXHEALTH.NS", "CUMMINSIND.NS", "INDUSTOWER.NS", "MARICO.NS",
    "BSE.NS", "HINDPETRO.NS", "NHPC.NS", "IDEA.NS", "SBICARD.NS",
    "BHARATFORG.NS", "ASHOKLEY.NS", "OFSS.NS", "NMDC.NS", "PRESTIGE.NS",
    "YESBANK.NS", "ALKEM.NS", "OIL.NS", "COLPAL.NS", "TORNTPOWER.NS",
    # TIINDIA.NS already present represents Tube Investments (remove alt Yahoo-incompatible symbol)
    "MRF.NS", "IDFCFIRSTB.NS", "GODREJPROP.NS", "OBEROIRLTY.NS",
    "SUPREMEIND.NS", "PHOENIXLTD.NS", "SUZLON.NS", "TATACOMM.NS", "PETRONET.NS",
    # User holdings additions
    "ARVIND.NS", "FEDERALBNK.NS", "GOKEX.NS", "IREDA.NS", "NEWGEN.NS", "PGEL.NS",
]

# Final custom universe (config.py de-duplicates across all sources)
TICKERS = CORE_TICKERS + MIDCAP_TICKERS

# Tip: Add more by appending strings like "FOO.NS" to the list above.
