import pandas as pd

class SimpleLinearRegression:
    def __init__(self) -> None:
        self.w_estimate = 0
        self.b_estimate = 0

    def get_parameter_estimates(self) -> tuple[float, float]:
        return self.w_estimate, self.b_estimate

    def calculate_cost(self, x: pd.Series, y: pd.Series, w: float, b: float) -> float:
        m = len(x)

        cost = 0

        for i in range(m):
            cost += ((w * x[i] + b) - y[i])**2
        return cost / 2*m

    def calculate_gradients(self, x: pd.Series, y: pd.Series, w: float, b: float) -> tuple[float, float]:
        return 0, 0

    def gradient_descent(self):
        pass

    def train_model(self, data: pd.DataFrame, x_name: str, y_name: str, n_iters: int) -> None:
        x = data[x_name]
        y = data[y_name]

        print(x)
        print(y)

    def predict(self, x: pd.DataFrame):
        pass