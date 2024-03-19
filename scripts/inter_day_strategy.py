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


class MLSentimentTrader(Strategy):
    def initialize(
        self,
        symbol: str = "SPY",
        cash_at_risk: float = 0.5,
        sleeptime: str = "24H",
        discord_webhook_url=None,
    ):
        self.symbol = symbol
        self.sleeptime = sleeptime
        self.discord_webhook_url = discord_webhook_url
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(
            base_url=BASE_URL,
            key_id=ALPACA_CREDS["API_KEY"],
            secret_key=ALPACA_CREDS["API_SECRET"],
        )

    def position_sizing(self):
        cash = self.get_cash()
        last_price = round(self.get_last_price(self.symbol), 2)
        quantity = round((cash * self.cash_at_risk) / last_price, 2)
        return cash, last_price, quantity

    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime("%Y-%m-%d"), three_days_prior.strftime("%Y-%m-%d")

    def get_sentiment(self):
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment

    def log_message(self, message, color=None, broadcast=True):
        return super().log_message(message, color, broadcast)

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()

        if cash > last_price:
            if sentiment == "positive" and probability > 0.999:
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.20,
                    stop_loss_price=last_price * 0.95,
                )
                self.submit_order(order)
                self.last_trade = "buy"
            elif sentiment == "negative" and probability > 0.999:
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * 0.8,
                    stop_loss_price=last_price * 1.05,
                )
                self.submit_order(order)
                self.last_trade = "sell"


class InterDayTrader(Strategy):
    def initialize(
        self,
        symbol: str = "SPY",
        cash_at_risk: float = 0.5,
        sleeptime: str = "24H",
        discord_webhook_url=None,
    ):
        self.symbol = symbol
        self.sleeptime = sleeptime
        self.discord_webhook_url = discord_webhook_url
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(
            base_url=BASE_URL,
            key_id=ALPACA_CREDS["API_KEY"],
            secret_key=ALPACA_CREDS["API_SECRET"],
        )

    def position_sizing(self):
        cash = self.get_cash()
        last_price = round(self.get_last_price(self.symbol), 2)
        quantity = round((cash * self.cash_at_risk) / last_price)
        return cash, last_price, quantity

    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime("%Y-%m-%d"), three_days_prior.strftime("%Y-%m-%d")

    def get_sentiment(self):
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment

    def log_message(self, message, color=None, broadcast=True):
        return super().log_message(message, color, broadcast)

    def calculate_and_prepare(self, stock_data):
        probability, sentiment = self.get_sentiment()
        if sentiment == "neutral":
            sentiment = 0
        elif sentiment == "positive":
            sentiment = 1
        else:
            sentiment = 2
        stock_data["RSI"] = ta.momentum.rsi(stock_data["Close"])
        stock_data["SMA"] = stock_data["Close"].rolling(window=20).mean()
        stock_data["MACD"] = ta.trend.macd_diff(stock_data["Close"])
        stock_data["Sentiment"] = sentiment
        stock_data["Sentiment_Probability"] = probability
        for days in [3, 5, 20, 50, 100, 200]:
            stock_data[f"{days}_Day_%_Change"] = stock_data["Close"].pct_change(days)

        local_min = stock_data.iloc[
            argrelextrema(stock_data.Close.values, np.less_equal, order=5)[0]
        ]["Close"]
        local_max = stock_data.iloc[
            argrelextrema(stock_data.Close.values, np.greater_equal, order=5)[0]
        ]["Close"]
        stock_data["Extrema"] = 1
        stock_data.loc[local_min.index, "Extrema"] = 0
        stock_data.loc[local_max.index, "Extrema"] = 2
        return stock_data.dropna()

    def on_trading_iteration(self):
        training_years = 10
        end_date = self.get_datetime()
        start_date = end_date - Timedelta(days=365 * training_years + 20)
        interval = "1d" if self.sleeptime == "24H" else "1m"
        stock_data = yf.download(
            self.symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
        )

        stock_data = self.calculate_and_prepare(stock_data)
        features = [
            "Close",
            "RSI",
            "SMA",
            "3_Day_%_Change",
            "5_Day_%_Change",
            "20_Day_%_Change",
            "MACD",
        ]
        model = train_random_forest_model(stock_data, features, "Extrema")
        signal = int(model.predict(pd.DataFrame(stock_data[features].iloc[-1, :]).T)[0])
        probability = max(
            model.predict_proba(pd.DataFrame(stock_data[features].iloc[-1, :]).T)[0]
        )
        # print(f'{signal}: {probability}')
        cash, last_price, quantity = self.position_sizing()
        quantity = quantity * probability
        if signal == 2:
            order = self.create_order(
                self.symbol,
                quantity,
                "sell",
                type="bracket",
                take_profit_price=round(last_price * 0.8),
                stop_loss_price=round(last_price * 1.05),
            )
            self.submit_order(order)
        elif signal == 0:
            order = self.create_order(
                self.symbol,
                quantity,
                "buy",
                type="bracket",
                take_profit_price=round(last_price * 1.20),
                stop_loss_price=round(last_price * 0.95),
            )
            self.submit_order(order)


class MultiInterDayTrader(InterDayTrader):
    def initialize(self, symbols, cash_at_risk=0.5, sleeptime: str = "24H"):
        self.symbols = symbols
        super().initialize(
            symbol=symbols[0], cash_at_risk=cash_at_risk, sleeptime=sleeptime
        )

    def on_trading_iteration(self):
        for symbol in self.symbols:
            self.symbol = symbol
            super().on_trading_iteration()


if __name__ == "__main__":
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
    inter_day_strategy.backtest(
        YahooDataBacktesting,
        START_DATE,
        END_DATE,
        parameters={
            "symbols": SYMBOLS,
            "cash_at_risk": CASH_AT_RISK,
            "sleeptime": SLEEPTIME,
        },
    )
