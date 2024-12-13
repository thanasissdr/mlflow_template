from typing import Any

import numpy as np
import pandas as pd


class Trainer:
    model: Any

    def run(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series):
        self.model.train(X, y)
