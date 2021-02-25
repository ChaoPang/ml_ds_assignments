# import system module
from os import path
import inflect

# import data related modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import product, permutations
from numpy.linalg import norm

# import plot modules
import matplotlib.pyplot as plt
import seaborn as sns

FEATURE_NAMES = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year made',
                 'bias term']

INPUT_FOLDER = 'hw2-data/Gaussian_process'
X_TRAIN = 'X_train.csv'
Y_TRAIN = 'y_train.csv'
X_TEST = 'X_test.csv'
Y_TEST = 'y_test.csv'

# +
# Loading the training data
x_train_pd = pd.read_csv(path.join(INPUT_FOLDER, X_TRAIN), sep=',', header=None)
y_train_pd = pd.read_csv(path.join(INPUT_FOLDER, Y_TRAIN), sep=',', header=None)

x_train = x_train_pd.to_numpy()
y_train = y_train_pd.to_numpy()

# Loading the test data
x_test_pd = pd.read_csv(path.join(INPUT_FOLDER, X_TEST), sep=',', header=None)
y_test_pd = pd.read_csv(path.join(INPUT_FOLDER, Y_TEST), sep=',', header=None)

x_test = x_test_pd.to_numpy()
y_test = y_test_pd.to_numpy()


# -

def rbf_kernel(v_1, v_2, b):
    return -np.exp(np.linalg.norm(v_1 - v_2) ** 2 / b)


def gaussian_process_regression(x_train, y_train, x_test, b, sigma_2):
    n, m = np.shape(x_train)
    n_t, _ = np.shape(x_test)

    K_1 = np.reshape([rbf_kernel(i, j, b) for i, j in product(x_train, x_train)], (n, n))
    K_2 = np.reshape([rbf_kernel(i, j, b) for i, j in product(x_test, x_train)], (n_t, n))
    K_3 = np.reshape([rbf_kernel(i, j, b) for i, j in product(x_test, x_test)], (n_t, n_t))

    mean = K_2 @ np.linalg.inv(np.identity(n) * sigma_2 + K_1) @ y_train
    cov = sigma_2 + K_3 - K_2 @ np.linalg.inv(np.identity(n) * sigma_2 + K_1) @ K_2.T

    return mean, cov


def compute_rmse(y_true, y_pred):
    num_of_rows, _ = np.shape(y_true)
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / num_of_rows)


b_values = list(range(5, 17, 2))
sigma_2_values = [s / 10 for s in range(1, 11)]

metrics = []
for b, sigma_2 in product(b_values, sigma_2_values):
    mean, cov = gaussian_process_regression(x_train, y_train, x_test, b, sigma_2)
    rmse = compute_rmse(y_test, mean)
    metrics.append((b, sigma_2, rmse))
#     print(f'b: {b} sigma: {sigma_2} rmse: {rmse}')

metrics

sorted(metrics, key=lambda t: t[2])
