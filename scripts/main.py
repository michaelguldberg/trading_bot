# from alpaca_broker import AlpacaBroker
from datetime import datetime

from config import (
    ALPACA_CREDS,
    CASH_AT_RISK,
    END_DATE,
    SLEEPTIME,
    START_DATE,
    SYMBOLS,
    WEBHOOK,
)
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from inter_day_strategy import MLSentimentTrader, MultiInterDayTrader

broker = Alpaca(ALPACA_CREDS)
inter_day_strategy = MultiInterDayTrader(
    name="mlstrat",
    broker=broker,
    parameters={
        "symbols": SYMBOLS,
        "cash_at_risk": CASH_AT_RISK,
        "sleeptime": SLEEPTIME,
        "discord_webhook_url": WEBHOOK,
    },
)

# intra_day_strategy = MultiInterDayTrader(
#     name="mlstrat",
#     broker=broker,
#     parameters={
#         "symbols": SYMBOLS,
#         "cash_at_risk": CASH_AT_RISK,
#         "sleeptime": "1M",
#         "discord_webhook_url": WEBHOOK,
#     },
# )

# inter_day_strategy.backtest(
#     YahooDataBacktesting,
#     START_DATE,
#     END_DATE,
#     parameters={
#         "symbols": SYMBOLS,
#         "cash_at_risk": CASH_AT_RISK,
#         "sleeptime": SLEEPTIME,
#     },
# )

# #Live Trading
trader = Trader()
trader.add_strategy(inter_day_strategy)
trader.run_all()
