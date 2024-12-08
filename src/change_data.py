import pandas as pd
import numpy as np

def run_change_data():
    X_test = pd.read_csv('../data/test_data.csv')

    # Change at least two feature values.
    # For example, swap the values of the first two features for all rows, or just add random noise.
    # Let's assume features are numeric.
    if X_test.shape[1] >= 2:
        # Swap feature 0 and 1
        temp = X_test.iloc[:, 0].copy()
        X_test.iloc[:, 0] = X_test.iloc[:, 1]
        X_test.iloc[:, 1] = temp

        # Or add random noise to one column
        # X_test.iloc[:, 1] = X_test.iloc[:, 1] + np.random.normal(0, 1, size=len(X_test))

    X_test.to_csv('../data/test_data_changed.csv', index=False)