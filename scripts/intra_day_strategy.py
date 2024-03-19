import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import ta
import yfinance as yf
from alpaca_trade_api import REST
from alpaca_trade_api.rest import APIError
from config import (
    ALPACA_CREDS,
    BASE_URL,
    CASH_AT_RISK,
    END_DATE,
    SLEEPTIME,
    START_DATE,
    SYMBOLS,
    WEBHOOK,
)
from finbert_utils import estimate_sentiment
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.strategies.strategy import Strategy
from ml_utils import train_model, train_random_forest_model
from scipy.signal import argrelextrema
from timedelta import Timedelta