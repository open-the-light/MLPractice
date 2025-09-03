import pandas as pd
import numpy as np

class MultipleLinearRegression:
    def __init__(self, data: pd.DataFrame, y_name: str) -> None:
        self.y = data[y_name]
        self.x = data.loc[:, data.columns != y_name]

        self.n_parameters = self.x.shape[1]
        self.n_obs = self.x.shape[0]
        self.ws = np.zeros(self.n_parameters)
        self.b = 0
        self.alpha = 0.01

    def get_parameter_estimates(self) -> tuple[np.ndarray, float]:
        return self.ws, self.b

    