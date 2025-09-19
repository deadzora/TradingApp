
from abc import ABC, abstractmethod
import pandas as pd
class BaseData(ABC):
    @abstractmethod
    def fetch_history(self, symbol: str, period: str = "6mo", interval: str = "1d"):
        ...
