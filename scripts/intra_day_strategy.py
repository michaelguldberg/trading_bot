import os
import pickle
from datetime import datetime

from lumibot.entities import Asset
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


class InterDayTrader(Strategy):
    def __init__(
        self,
        *args,
        broker=None,
        minutes_before_closing=1,
        minutes_before_opening=60,
        sleeptime="1M",
        stats_file=None,
        risk_free_rate=None,
        benchmark_asset="SPY",
        backtesting_start=None,
        backtesting_end=None,
        quote_asset=...,
        starting_positions=None,
        filled_order_callback=None,
        name=None,
        budget=None,
        parameters=...,
        buy_trading_fees=...,
        sell_trading_fees=...,
        force_start_immediately=False,
        discord_webhook_url=None,
        account_history_db_connection_str=None,
        strategy_id=None,
        discord_account_summary_footer=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            broker=broker,
            minutes_before_closing=minutes_before_closing,
            minutes_before_opening=minutes_before_opening,
            sleeptime=sleeptime,
            stats_file=stats_file,
            risk_free_rate=risk_free_rate,
            benchmark_asset=benchmark_asset,
            backtesting_start=backtesting_start,
            backtesting_end=backtesting_end,
            quote_asset=quote_asset,
            starting_positions=starting_positions,
            filled_order_callback=filled_order_callback,
            name=name,
            budget=budget,
            parameters=parameters,
            buy_trading_fees=buy_trading_fees,
            sell_trading_fees=sell_trading_fees,
            force_start_immediately=force_start_immediately,
            discord_webhook_url=discord_webhook_url,
            account_history_db_connection_str=account_history_db_connection_str,
            strategy_id=strategy_id,
            discord_account_summary_footer=discord_account_summary_footer,
            **kwargs,
        )
