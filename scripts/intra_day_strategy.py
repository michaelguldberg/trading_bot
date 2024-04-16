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
from lumibot.traders import Trader
from lumibot.strategies.strategy import Strategy
from ml_utils import train_KNN_model, train_random_forest_model
from scipy.signal import argrelextrema
from timedelta import Timedelta


class IntraDayTrader(Strategy):
    def initialize(
        self,
        symbols: str = ["SPY"],
        cash_at_risk: float = 0.5,
        sleeptime: str = "1M",
        discord_webhook_url=None,
    ):
        self.symbols = symbols
        self.sleeptime = sleeptime
        self.discord_webhook_url = discord_webhook_url
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(
            base_url=BASE_URL,
            key_id=ALPACA_CREDS["API_KEY"],
            secret_key=ALPACA_CREDS["API_SECRET"],
        )
        self.minutes_before_opening = 5
        self.dict_of_models = {}
        self.signal_probabilities = []
        self.features = [
            "Close",
            "TSI",
            "SMA",
            "3_Day_%_Change",
            "5_Day_%_Change",
            "20_Day_%_Change",
            "MACD",
        ]

    def position_sizing(self, symbol, probability):
        cash = self.get_cash()
        last_price = round(self.get_last_price(symbol), 2)
        quantity = round((cash * self.cash_at_risk * probability) / last_price, 2)
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
        stock_data["TSI"] = ta.momentum.tsi(stock_data["Close"])
        stock_data["SMA"] = stock_data["Close"].rolling(window=20).mean()
        stock_data["MACD"] = ta.trend.macd_diff(stock_data["Close"])
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
    
    def train_model(self, symbol, training_days=7):
        end_date = self.get_datetime()
        start_date = end_date - Timedelta(days=training_days)
        interval = "1d" if self.sleeptime in ["24H", "1D"] else "1m"
        stock_data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
        )
        stock_data = self.calculate_and_prepare(stock_data)
        model = train_random_forest_model(stock_data, self.features, "Extrema")
        signal = int(model.predict(pd.DataFrame(stock_data[self.features].iloc[-1, :]).T)[0])
        probability = max(
            model.predict_proba(pd.DataFrame(stock_data[self.features].iloc[-1, :]).T)[0]
        )
        return stock_data, model, signal, probability
    
    def place_order(self, symbol, signal, quantity, last_price):
        if signal == 2:
            order = self.create_order(
                symbol,
                quantity,
                "sell",
                type="bracket",
                take_profit_price=round(last_price * 0.8),
                stop_loss_price=round(last_price * 1.05),
                time_in_force="day",
            )
            self.submit_order(order)
        elif signal == 0:
            order = self.create_order(
                symbol,
                quantity,
                "buy",
                type="bracket",
                take_profit_price=round(last_price * 1.20),
                stop_loss_price=round(last_price * 0.95),
                time_in_force="day",
                )
            self.submit_order(order)

    def before_starting_trading(self, training_days=7):
        self.dict_of_models = {}
        for symbol in self.symbols:
            end_date = self.get_datetime()
            start_date = end_date - Timedelta(days=training_days)
            interval = "1d" if self.sleeptime in ["24H", "1D"] else "1m"
            stock_data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
            )
            raw_data = stock_data
            stock_data = self.calculate_and_prepare(stock_data)
            model = train_random_forest_model(stock_data, self.features, "Extrema")
            self.dict_of_models[symbol] = model
        
    
    def on_trading_iteration(self):
        for symbol, model in self.dict_of_models.items():
            stock_data = yf.download(
                symbol,
                start=self.get_datetime() - Timedelta(minutes=400),
                end=self.get_datetime(),
                interval="1m",
                progress=False,
            )
            stock_data["TSI"] = ta.momentum.tsi(stock_data["Close"])
            stock_data["SMA"] = stock_data["Close"].rolling(window=20).mean()
            stock_data["MACD"] = ta.trend.macd_diff(stock_data["Close"])
            for minutes in [3, 5, 20]:
                stock_data[f"{minutes}_Day_%_Change"] = stock_data["Close"].pct_change(minutes)
            signal = int(model.predict(pd.DataFrame(stock_data[self.features].iloc[-1, :]).T)[0])
            probability = max(
            model.predict_proba(pd.DataFrame(stock_data[self.features].iloc[-1, :]).T)[0]
            )
            print(stock_data)
            print(signal)
            if signal in [0, 2]:
                self.signal_probabilities.append(probability)
                normalized_probability = probability
                if len(self.signal_probabilities) not in [0,1] and min(self.signal_probabilities) != max(self.signal_probabilities):
                    min_prob = min(self.signal_probabilities)
                    max_prob = max(self.signal_probabilities)
                    normalized_probability = (probability - min_prob) / (max_prob - min_prob)
                cash, last_price, quantity = self.position_sizing(symbol, normalized_probability)
                self.place_order(symbol, signal, quantity, last_price)

if __name__ == "__main__":
    broker = Alpaca(ALPACA_CREDS)
    intra_day_strategy = IntraDayTrader(
        name="intra_day_trader",
        broker=broker,
        sleeptime='1M',
        parameters={
            "symbols": SYMBOLS,
            "cash_at_risk": CASH_AT_RISK,
            "sleeptime": '1M',
            "discord_webhook_url": WEBHOOK,
        },
    )
    backtest = intra_day_strategy.backtest(
        YahooDataBacktesting,
        datetime(2024,3,5),
        datetime(2024,3,10),
        budget=500,
        parameters={
            "symbols": ['SPY'],
            "cash_at_risk": CASH_AT_RISK,
            "sleeptime": '1M',
        },
    )
    # trader = Trader()
    # trader.add_strategy(intra_day_strategy)
    # trader.run_all()