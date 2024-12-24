from abc import ABC, abstractmethod

import pandas as pd


class MetricRunner(ABC):
    @abstractmethod
    def run(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        pass
