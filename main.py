from SimpleLinearRegression import SimpleLinearRegression
import sklearn.datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    boston = pd.read_csv("datasets/Boston.csv", index_col=0)

    var = "dis"
    plt.scatter(boston[var], boston["medv"])
    plt.show()

    test_data = pd.DataFrame(
        {
            'x': [1.0, 2.0],
            'y': [300.0, 500.0]
        }
    )

    lr_model = SimpleLinearRegression(boston, var, "medv")
    lr_model.train_model(n_iters=1000)

    w, b, = lr_model.get_parameter_estimates()

    x = np.linspace(0, 15, 100)
    plt.scatter(boston[var], boston["medv"])
    plt.plot(x, w * x + b, "-r")
    plt.show()


if __name__ == "__main__":
    main()
