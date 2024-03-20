from datetime import datetime

import pandas as pd

API_KEY = "PKZ99M8PEGYY37QKHD21"
API_SECRET = "PZoafr0EDDdNtIxVmtJboR9oWNaQeVhxp6LjERXe"
SLEEPTIME = "24H"
WEBHOOK = 'https://discord.com/api/webhooks/1217652897131663360/h-eefnR2x-n2NMyDiiBJSBTkD4yA2y2p6nduoEPgfmBxoIrJ_BU_s6qR0MvVOX2WHCu1'
PAPER = True
if PAPER:
    BASE_URL = "https://paper-api.alpaca.markets"
else:
    BASE_URL = "https://api.alpaca.markets"

# DO NOT CHANGE
ALPACA_CREDS = {"API_KEY": API_KEY, "API_SECRET": API_SECRET, "PAPER": True}

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 3, 14)
# SYMBOLS = list(pd.read_csv('data/companies.csv')['Ticker'])
# SYMBOLS = ["SPY", "AAPL", "MSFT"] # for testing
SYMBOLS = ["SPY", "SMH", "SOXX", "XLK", "IYW", "FTEC"]
# SYMBOLS = ["SPY", "AAPL", "MSFT"]  # for testing
CASH_AT_RISK = 0.5
