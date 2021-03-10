"""Implement the BaseMethod class."""
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator


class BaseMethod(ABC, BaseEstimator):
    """Base abstract class for creating methods."""

    pass
