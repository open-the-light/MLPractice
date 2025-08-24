from SimpleLinearRegression import SimpleLinearRegression
import sklearn.datasets
import pandas as pd

def main():
    boston = pd.read_csv("datasets/Boston.csv", index_col=0)
    print(boston.head())
    print(boston.describe())

    lr_model = SimpleLinearRegression()
    lr_model.train_model(boston, "age", "medv", 10000)


if __name__ == "__main__":
    main()
