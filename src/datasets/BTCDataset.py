"""Implement the BTCDataset class."""
import os
import pandas as pd

from .BaseDataset import BaseDataset, data_folder


class BTCDataset(BaseDataset):

    def __init__(self, root_folder=data_folder):
        self.root_folder = root_folder

        self._data = None

    def load_data(self):
        path = os.path.join(self.root_folder, 'financial/btcusd/gemini_BTCUSD_1hr.csv')
        data = pd.read_csv(path)
        data['Unix'] = pd.to_datetime(data['Unix'], unit='s')
        data.set_index('Unix', inplace=True)
        data.sort_index(axis=0, inplace=True)
        self._data = data

    @property
    def data(self):
        if self._data is None:
            self.load_data()
        return self._data

    @property
    def timeseries(self):
        df = self.data
        return df['Close']
