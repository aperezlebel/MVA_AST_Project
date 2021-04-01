"""Implement the ArtificalDataset class."""
import os
import pandas as pd
import quandl
import zipfile
import numpy as np

from .BaseDataset import BaseDataset, data_folder

class ArtificalDataset(BaseDataset):

    def __init__(self, length, sinu_period, tria_period, plat_period):
        self.length = length
        self.sinu_period = sinu_period
        self.tria_period = tria_period
        self.plat_period = plat_period
        self._data = None
    
    def load_data(self):
        data = [np.zeros(self.length)]
        if self.sinu_period != 0:
            motif_sinu = [np.sin(2 * i * np.pi / (self.sinu_period + 1)) for i in range(self.sinu_period)]
            sinu = np.tile(motif_sinu[:-1], self.length // len(motif_sinu[:-1]) + 1)[:self.length]
            data[0] += sinu
        if self.tria_period != 0:
            motif_tria = [i / (self.tria_period // 2 - 1) for i in range(self.tria_period // 2 - 1)] + [1 - i / (self.tria_period - self.tria_period//2) for i in range(self.tria_period - self.tria_period//2 + 1)]
            tria = np.tile(motif_tria[:-1], self.length // len(motif_tria[:-1]) + 1)[:self.length]
            data[0] += tria
        if self.plat_period != 0:
            motif_plat = [-1] * (self.plat_period // 2) + [1] * (self.plat_period - self.plat_period // 2)
            plat = np.tile(motif_plat[:-1], self.length // len(motif_plat[:-1]) + 1)[:self.length]
            data[0] += plat
        data[0] += 4
        self._data = data

    @property
    def data(self):
        if self._data is None:
            self.load_data()
        return self._data

    @property
    def timeseries(self):
        df = self.data
        return df