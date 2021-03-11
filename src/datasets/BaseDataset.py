"""Implement the BaseDataset class."""
import os
from abc import ABC, abstractmethod


data_folder = os.path.abspath('src/data/')


class BaseDataset(ABC):
    """Base abstract class for datasets."""

    pass

