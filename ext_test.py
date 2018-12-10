import pandas as pd
import numpy as np

def kaggle_test():
    df = pd.read_csv("./kaggle_test.csv")
    y = df.label
    X = df.iloc[:, 1:]

    X, y = np.array(X), np.array(y)
    X = X.reshape((-1, 28, 28, 1))

    return X, y
