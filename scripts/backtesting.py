from lumibot.backtesting import YahooDataBacktesting

class YahooDataBacktesting(YahooDataBacktesting):
    def get_data(self, symbol, start_date, end_date, interval):
        return 1
