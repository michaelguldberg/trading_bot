import os
import pickle
import requests
import yfinance as yf
from datetime import datetime, timedelta, UTC

from lumibot.entities import Asset
import numpy as np
import pandas as pd
import ta
from alpaca_trade_api import REST
from alpaca.data.timeframe import TimeFrame
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
from lumibot.backtesting import YahooDataBacktesting, PandasDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from lumibot.strategies.strategy import Strategy
from ml_utils import train_KNN_model, train_random_forest_model, train_random_forest_regressor, train_test
from scipy.signal import argrelextrema
from timedelta import Timedelta
import pytz


class CryptoTrader(Strategy):
    def initialize(
        self,
        symbols: str = ["BTC-USD"],
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
        self.model = 0
        self.signal_probabilities = []
        self.close = 'close'
        self.features = [
            self.close,
            "RSI",
            "SMA",
            "3_Minute_%_Change",
            "5_Minute_%_Change",
            "20_Minute_%_Change",
            "MACD",
        ]
        self.bitcoin_data = yf.download(
                'BTC-USD',
                start=datetime.now(UTC) - timedelta(minutes=400),
                end=datetime.now(UTC),
                interval="1m",
                progress=False,
            )
        self.set_market('24/7')

    def position_sizing(self, symbol, probability):
        pass

    def get_dates(self):
        pass

    def get_sentiment(self):
        pass

    def log_message(self, message, color=None, broadcast=True):
        return super().log_message(message, color, broadcast)

    def calculate_and_prepare(self, stock_data, pre_trading=True):
        stock_data["RSI"] = ta.momentum.rsi(stock_data[self.close])
        stock_data["SMA"] = stock_data[self.close].rolling(window=20).mean()
        stock_data["MACD"] = ta.trend.macd_diff(stock_data[self.close])
        for days in [3, 5, 20]:
            stock_data[f"{days}_Minute_%_Change"] = stock_data[self.close].pct_change(days)
        if pre_trading:
            local_min = stock_data.iloc[
                argrelextrema(stock_data[self.close].values, np.less_equal, order=5)[0]
            ][self.close]
            local_max = stock_data.iloc[
                argrelextrema(stock_data[self.close].values, np.greater_equal, order=5)[0]
            ][self.close]
            stock_data["Extrema"] = 1
            stock_data.loc[local_min.index, "Extrema"] = 0
            stock_data.loc[local_max.index, "Extrema"] = 2
            stock_data['Future_Close'] = stock_data[self.close].shift(-20)
        return stock_data
    
    def train_model(self, symbol, training_days=7):
        pass

    def before_starting_trading(self, training_days=7):
        pass
        
    def get_historical_data(self):
        # bitcoin_data = yf.download(
        #         'BTC-USD',
        #         start=self.get_datetime().astimezone(UTC) - timedelta(days=7),
        #         end=self.get_datetime().astimezone(UTC),
        #         interval="1m",
        #         progress=False,
        # )
        # return bitcoin_data
        utc = pytz.UTC
        end_date = datetime.now().astimezone(utc)
        start_date = end_date - timedelta(days=7)

        # Alpaca expects timestamps in ISO format
        start_date_iso = start_date.isoformat()
        end_date_iso = end_date.isoformat()

        client = CryptoHistoricalDataClient()
        request_params = CryptoBarsRequest(
            symbol_or_symbols=['BTC/USD'],
            timeframe=TimeFrame.Minute,
            start=start_date_iso,
            end=end_date_iso
        )
        btc_bars = client.get_crypto_bars(request_params).df
        return btc_bars

    
    def get_current_data(self, existing_df):
        coin_api_key = 'e2653b1f-0e7e-4133-85ab-2c6e4ecd4f60'
        url = "https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': coin_api_key,
        }
        # Assuming Bitcoin's symbol is 'BTC' and we are fetching data for the last 5 minutes, each minute
        params = {
            'start': '1',
            'limit': 5,
            'symbol':'BTC',
            'convert':'USD',
        }
        
        # Make the API call
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        print(data)
        new_data_df = pd.DataFrame(data['data']['BTC']['quotes'])
        print(new_data_df)
        
        # Convert timestamps to a uniform format and set as index
        new_data_df['Timestamp'] = pd.to_datetime(new_data_df['Timestamp'])
        existing_df['Timestamp'] = pd.to_datetime(existing_df['Timestamp'])
        new_data_df.set_index('Timestamp', inplace=True)
        existing_df.set_index('Timestamp', inplace=True)
        
        # Append new rows that don't exist in the existing DataFrame
        combined_df = existing_df.append(new_data_df[~new_data_df.index.isin(existing_df.index)])
        
        return combined_df.reset_index()
    
    def print_stuff(self, prediction):
        live = self.get_datetime().astimezone(UTC).strftime("%B %d, %Y, %I:%M %p %Z")
        date_str = self.bitcoin_data.iloc[-1,:].name[1]
        recent = date_str.strftime("%B %d, %Y, %I:%M %p %Z")
        time_delay = self.get_datetime().astimezone(UTC) - self.bitcoin_data.iloc[-1,:].name[1]
        curr_date = self.bitcoin_data.iloc[-1,:]
        width_label = 30
        width_value = 40
        print()
        print(f'{"Current Live Time:":<{width_label}}{live:>{width_value}}')
        print(f'{"Most Recent BTC Data:":<{width_label}}{recent:>{width_value}}')
        print(f'{"Time Delay:":<{width_label}}{str(time_delay):>{width_value}}')
        print(f'{"Current Price:":<{width_label}}{curr_date["close"]:>{width_value}.2f}')
        print(f'{"Predicted Price in 20 Mins:":<{width_label}}{prediction:>{width_value}.2f}')
        print(f'{"RSI:":<{width_label}}{curr_date["RSI"]:>{width_value}.2f}')
        print(f'{"MACD:":<{width_label}}{curr_date["MACD"]:>{width_value}.2f}')
        print(f'{"3_Minute_%_Change:":<{width_label}}{curr_date["3_Minute_%_Change"] * 100:>{width_value}.2f}%')
        print(f'{"5_Minute_%_Change:":<{width_label}}{curr_date["5_Minute_%_Change"] * 100:>{width_value}.2f}%')
        print(f'{"20_Minute_%_Change:":<{width_label}}{curr_date["20_Minute_%_Change"] * 100:>{width_value}.2f}%')

    def on_trading_iteration(self):
        self.bitcoin_data = self.get_historical_data()
        # self.bitcoin_data = self.get_current_data(self.bitcoin_data)
        self.bitcoin_data = self.calculate_and_prepare(self.bitcoin_data)
        self.model = train_random_forest_regressor(self.bitcoin_data.iloc[:-1,:].dropna(), self.features, 'Future_Close')
        prediction = self.model.predict(pd.DataFrame(self.bitcoin_data[self.features].iloc[-1, :]).T)[0]
        # probability = max(
        #     self.model.predict_proba(pd.DataFrame(self.bitcoin_data[self.features].iloc[-1, :]).T)[0]
        # )
        # test_df = train_test(self.bitcoin_data, self.features, 'Future_Close')
        # test_df['%_change'] = (test_df.y_pred - test_df.Close) / test_df.Close
        # test_df['actual_%_change'] = (test_df.y_actual - test_df.Close) / test_df.Close
        # errors_df = test_df[test_df['%_change'] * test_df['actual_%_change'] < 0]
        # errors_df['error_magnitude'] = abs(errors_df.y_pred - errors_df.y_actual)
        # # print(errors_df.describe())
        # print(test_df[abs(test_df.y_pred - test_df.y_actual) > 1000])
        
        last_price = self.bitcoin_data[self.features].iloc[-1, :][self.close]
        if prediction > last_price * 1.001:
            signal = 0
        elif prediction < last_price * 0.999:
            signal = 2
        else:
            signal = 1
        # print(signal)
        # print(probability)
        self.print_stuff(prediction)
        if signal == 2:
            order = self.create_order(
                Asset(symbol='BTC', asset_type=Asset.AssetType.CRYPTO),
                0.1,
                "sell",
                type="market",
                time_in_force="gtc",
                quote=Asset(symbol='USD', asset_type='crypto')
            )
            self.submit_order(order)
        elif signal == 0:
            order = self.create_order(
                Asset(symbol='BTC', asset_type=Asset.AssetType.CRYPTO),
                0.1,
                "buy",
                type="market",
                time_in_force="gtc",
                quote=Asset(symbol='USD', asset_type='crypto')
            )
            self.submit_order(order)

if __name__ == "__main__":
    broker = Alpaca(ALPACA_CREDS)
    intra_day_strategy = CryptoTrader(
        name="crypto_trader",
        broker=broker,
        sleeptime='1M',
        parameters={
            "symbols": ['BTC-USD'],
            "cash_at_risk": CASH_AT_RISK,
            "sleeptime": '1M',
            "discord_webhook_url": WEBHOOK,
        },
    )
    # backtest = intra_day_strategy.backtest(
    #     YahooDataBacktesting,
    #     datetime(2024,3,23),
    #     datetime(2024,3,26),
    #     budget=500,
    #     quote_asset= Asset(symbol="USD", asset_type="crypto"),
    #     parameters={
    #         "symbols": [Asset(symbol='BTC', asset_type=Asset.AssetType.CRYPTO)],
    #         "cash_at_risk": CASH_AT_RISK,
    #         "sleeptime": '1M',
    #     },
    # )
    trader = Trader()
    trader.add_strategy(intra_day_strategy)
    trader.run_all()


