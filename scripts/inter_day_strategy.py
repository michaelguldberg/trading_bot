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
from ml_utils import train_KNN_model, train_random_forest_model
from scipy.signal import argrelextrema
import random
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
        symbols: str = ["SPY"],
        cash_at_risk: float = 0.5,
        sleeptime: str = "24H",
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
        self.signal_probabilities = []

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
    
    def train_model(self, symbol, training_years=10):
        end_date = self.get_datetime()
        start_date = end_date - Timedelta(days=365 * training_years + 20)
        interval = "1d" if self.sleeptime in ["24H", "1D"] else "1m"
        stock_data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
        )
        stock_data = self.calculate_and_prepare(stock_data)
        features = [
            "Close",
            "TSI",
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
                )
            self.submit_order(order)

    def on_trading_iteration(self):
        for symbol in self.symbols:
            stock_data, model, signal, probability = self.train_model(symbol=symbol, training_years=10)
            if signal in [0, 2]:
                self.signal_probabilities.append(probability)
                normalized_probability = probability
                if len(self.signal_probabilities) not in [0,1] and min(self.signal_probabilities) != max(self.signal_probabilities):
                    min_prob = min(self.signal_probabilities)
                    max_prob = max(self.signal_probabilities)
                    normalized_probability = (probability - min_prob) / (max_prob - min_prob)
                cash, last_price, quantity = self.position_sizing(symbol, normalized_probability)
                self.place_order(symbol, signal, quantity, last_price)
            

def get_random_dates(years=range(2015,2025)):
    date_periods = []
    for i in range(5):
        start_year = random.choice(years)
        end_year = random.choice(range(start_year, years[-1]))
        if start_year == end_year:
            start_month = random.choice(range(1,13))
            end_month = random.choice(range(start_month, 13))
            if start_month == end_month:
                start_day = random.choice(range(1, 31))
                end_day = random.choice(range(start_day, 31))
        else:
            start_month = random.choice(range(1,13))
            end_month = random.choice(range(1,13))
            start_day = random.choice(range(1,31))
            end_day = random.choice(range(1,31))
        date_periods.append((datetime(start_year, start_month, start_day), datetime(end_year, end_month, end_day)))
    return date_periods

if __name__ == "__main__":
    broker = Alpaca(ALPACA_CREDS)
    inter_day_strategy = InterDayTrader(
        name="inter_day_trader",
        broker=broker,
        parameters={
            "symbols": SYMBOLS,
            "cash_at_risk": CASH_AT_RISK,
            "sleeptime": SLEEPTIME,
            "discord_webhook_url": WEBHOOK,
        },
    )
    # backtest = inter_day_strategy.backtest(
    #     YahooDataBacktesting,
    #     START_DATE,
    #     END_DATE,
    #     budget=500,
    #     parameters={
    #         "symbols": SYMBOLS,
    #         "cash_at_risk": CASH_AT_RISK,
    #         "sleeptime": SLEEPTIME,
    #     },
    # )
    date_periods = [
        (datetime(2024, 1, 1), datetime(2024, 1, 31)),
        (datetime(2024, 2, 1), datetime(2024, 2, 28)),
        (datetime(2023, 1, 1), datetime(2023, 1, 31)),
        (datetime(2019, 1, 1), datetime(2019, 3, 13)),
        # Add more periods as needed
    ]
    # date_periods = get_random_dates()

    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame(columns=['Start Date', 'End Date', 'CAGR', 'Volatility', 'Sharpe', 'Max Drawdown', 'ROMAD', 'Total Return'])

    for start_date, end_date in date_periods:
        # Assuming inter_day_strategy.backtest(...) is your function call
        backtest = inter_day_strategy.backtest(
            YahooDataBacktesting,
            start_date,
            end_date,
            budget=500,
            parameters={
                "symbols": SYMBOLS,
                "cash_at_risk": CASH_AT_RISK,
                "sleeptime": SLEEPTIME,
            },
        )
        
        # Extract results
        cagr = backtest['cagr']
        volatility = backtest['volatility']
        sharpe = backtest['sharpe']
        max_drawdown = backtest['max_drawdown']['drawdown']
        romad = backtest['romad']
        total_return = backtest['total_return']
        
        # Prepare a new row as a DataFrame to concatenate
        new_row = pd.DataFrame({
            'Start Date': [start_date],
            'End Date': [end_date],
            'CAGR': [cagr],
            'Volatility': [volatility],
            'Sharpe': [sharpe],
            'Max Drawdown': [max_drawdown],
            'ROMAD': [romad],
            'Total Return': [total_return]
        })
        # Concatenate the new row
        results_df = pd.concat([results_df, new_row], ignore_index=True)

    # Display the DataFrame
    print(results_df)
    
