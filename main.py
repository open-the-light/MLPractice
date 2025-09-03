from Models.SimpleLinearRegression import SimpleLinearRegression
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    boston = pd.read_csv("datasets/BostonHousing.csv")

    var = "RM"
    plt.scatter(boston[var], boston["MEDV"])
    plt.show()

    test_data = pd.DataFrame(
        {
            'x': [1.0, 2.0],
            'y': [300.0, 500.0]
        }
    )

    lr_model = SimpleLinearRegression(boston, var, "MEDV")
    lr_model.train_model(alpha=0.04, n_iters=10000, stop_criteria=0.00001)

    w, b, = lr_model.get_parameter_estimates()

    comp_model = LinearRegression()
    comp_model.fit(boston[[var]], boston[["MEDV"]])
    print(f"Coefficients: {comp_model.coef_}")
    print(f"Intercept: {comp_model.intercept_}")

    x = np.linspace(0, 12, 100)
    plt.scatter(boston[var], boston["MEDV"])
    plt.plot(x, w * x + b, "-r")
    plt.plot(x, comp_model.coef_[0] * x + comp_model.intercept_, "-g")
    plt.show()

    boston_copy = boston.copy()
    boston_copy["lr_pred"] = lr_model.predict(boston[var])
    print(boston_copy[[var, "MEDV", "lr_pred"]])


if __name__ == "__main__":
    main()
