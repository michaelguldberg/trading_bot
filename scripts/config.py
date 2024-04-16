from datetime import datetime

import pandas as pd

API_KEY = "API KEY"
API_SECRET = "API SECRET"
SLEEPTIME = "24H"
WEBHOOK = "DISCORD URL"
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
