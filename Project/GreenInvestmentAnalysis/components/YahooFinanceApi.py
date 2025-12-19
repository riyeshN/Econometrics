import yfinance as yfinance
import pandas as pd

class YahooFinanceApi:

    def __init__(self, tickers: list, start_date : str, end_date : str):
        self.__tickers = tickers
        self.__start_date = start_date
        self.__end_date = end_date
        self.__security_data_frame = pd.DataFrame()
        self.__security_value_percent_change = pd.DataFrame()

    def initiate(self):
        self.__get_data_frame_for_tickers()
        self.__set_price_return_change()

    def __get_data_frame_for_tickers(self):
        data = yfinance.download(self.__tickers,
                                 start=self.__start_date, end=self.__end_date, auto_adjust=True)
        data = data.Close
        self.__security_data_frame = data

    def __set_price_return_change(self):
        if not self.__security_data_frame.empty:
            self.__security_value_percent_change = self.__security_data_frame.pct_change()

    def get_security_value_percent_change_df(self):
        return self.__security_value_percent_change.copy()

