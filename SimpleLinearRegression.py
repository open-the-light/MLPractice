import pandas as pd

class SimpleLinearRegression:
    def __init__(self, data: pd.DataFrame, x_name: str, y_name: str) -> None:
        self.w = 0
        self.b = 0
        self.alpha = 0.01

        self.x = data[x_name]
        self.y = data[y_name]

        self.m = len(self.x)

    def get_parameter_estimates(self) -> tuple[float, float]:
        return self.w, self.b

    def calculate_cost(self) -> float:
        cost = sum(((self.w * self.x + self.b) - self.y)**2)
        return cost / (2*self.m)

    def calculate_gradients(self) -> tuple[float, float]:
        
        dj_dw = sum(((self.w * self.x + self.b) - self.y) * self.x)
        dj_db = sum((self.w * self.x + self.b) - self.y)
        
        return (dj_dw / self.m, dj_db / self.m)

    def train_model(self, w: float = 0, b: float = 0, alpha: float = 0.01, n_iters: int = 10000) -> None:

        self.w = w
        self.b = b
        self.alpha = alpha

        iter = 1
        while iter <= n_iters:
            dj_dw, dj_db = self.calculate_gradients()

            self.w = self.w - self.alpha * dj_dw
            self.b = self.b - self.alpha * dj_db
            
            if iter % 100 == 0:
                print(f"Iter {iter}: w = {self.w}, b = {self.b}, cost = {self.calculate_cost()}")
            iter += 1
                


    def predict(self, x: pd.DataFrame):
        pass