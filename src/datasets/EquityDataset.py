"""Implement the BTCDataset class."""
import os
import pandas as pd
import quandl
import zipfile
import numpy as np

from .BaseDataset import BaseDataset, data_folder


tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'FB', 'NFLX', 'PYPL', 'IBM']


class EquityDataset(BaseDataset):

    def __init__(self, root_folder=data_folder, tickers=tickers,
                 replace_existing=False, min_date='2000-01-01', max_date='2020-01-01'):
        self.root_folder = root_folder
        self.tickers = tickers
        self._data = None
        self._timeseries = None
        self.replace_existing = replace_existing
        self.min_date = min_date
        self.max_date = max_date

        api_key = os.environ.get('quandl_api_key', None)
        if api_key is None:
            raise EnvironmentError('You must specify a Quandl API key in your environment '
                             'as variable "quandl_api_key" to use this dataset.\n'
                             'Use command "export quandl_api_key=\'YOUR_API_KEY\'"')
        quandl.ApiConfig.api_key = api_key

    def load_data(self):
        folder = os.path.join(self.root_folder, 'financial/equities/')
        os.makedirs(folder, exist_ok=True)
        zip_path = os.path.join(folder, 'wiki_prices_1day.zip')

        # Download table
        if not os.path.exists(zip_path) or self.replace_existing:
            print('Downloading dataset...')
            quandl.export_table('WIKI/PRICES',
                                qopts={'columns': ['ticker', 'date', 'close']},
                                ticker=self.tickers,
                                date={'gte': self.min_date, 'lte': self.max_date},
                                filename=zip_path)

        # Unzip downloaded table
        print('Unzipping dataset...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            content = zip_ref.namelist()[0]
            print(f'Extracting "{content}"')
            zip_ref.extract(content, path=folder)

        csv_path = os.path.join(folder, content)
        data = pd.read_csv(csv_path)

        timeseries = {}
        for ticker in self.tickers:
            d = data[data['ticker'] == ticker]
            d.set_index('date', inplace=True)
            d.sort_index(axis=0, inplace=True)
            ts = d['close']
            ts.rename(f'{ticker}_close', inplace=True)
            timeseries[ticker] = ts

        self._data = data
        self._timeseries = timeseries

    @property
    def data(self):
        if self._data is None:
            self.load_data()
        return self._data

    @property
    def timeseries_dict(self):
        if self._timeseries is None:
            self.load_data()
        return self._timeseries

    @property
    def timeseries(self):
        return [np.array(ts) for ts in self.timeseries_dict.values()]
