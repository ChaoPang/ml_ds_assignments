# import system module
from os import path

# import data related modules
import numpy as np
import pandas as pd

# import plot modules
import matplotlib.pyplot as plt
import seaborn as sns

from assignment_2.assignment_2 import INPUT_FOLDER, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, \
    gaussian_process_regression, explore_hyperparameters_cross_validation


def main():
    """
    Main function
    :return:
    """

    # Loading the training data
    x_train_gpr = pd.read_csv(path.join(INPUT_FOLDER, X_TRAIN), sep=',', header=None).to_numpy()
    y_train_gpr = pd.read_csv(path.join(INPUT_FOLDER, Y_TRAIN), sep=',', header=None).to_numpy()

    # Loading the test data
    x_test_gpr = pd.read_csv(path.join(INPUT_FOLDER, X_TEST), sep=',', header=None).to_numpy()
    y_test_gpr = pd.read_csv(path.join(INPUT_FOLDER, Y_TEST), sep=',', header=None).to_numpy()

    # Problem 3(a): calculate the RMSEs in the test set using Gaussian Process with different
    # combinations of b and sigma squared
    explore_hyperparameters_cross_validation(x_test_gpr, x_train_gpr, y_test_gpr, y_train_gpr)

    # Problem 3(c): explore the 4th dimension only and plot the predictive mean of each training
    # data point
    # Take the 4th dimension only
    x_train_gpr_car_weight = x_train_gpr[:, 3:4]
    mean, cov = gaussian_process_regression(x_train_gpr_car_weight, y_train_gpr,
                                            x_train_gpr_car_weight,
                                            5, 2)

    # +
    fig, ax = plt.subplots(figsize=(12, 9))
    # Set common labels
    ax.set_xlabel('Car weight')
    ax.set_ylabel('Miles per gallon')

    sns.scatterplot(ax=ax, x=np.squeeze(x_train_gpr_car_weight), y=np.squeeze(y_train_gpr))
    sns.lineplot(ax=ax, x=np.squeeze(x_train_gpr_car_weight), y=np.squeeze(mean))

    # Save the plot
    plt.savefig(path.join(INPUT_FOLDER, 'car_weight_rmse.png'))

    # Clear the figure
    plt.clf()


if __name__ == '__main__':
    """
    Author: Chao Pang 
    """
    main()

# +
x_train_gpr_car_weight = x_train_gpr[:, 3:4]
mean, cov = gaussian_process_regression(x_train_gpr_car_weight, y_train_gpr,
                                        x_train_gpr_car_weight,
                                        b=0.0001, sigma_square=2)

# # +
fig, ax = plt.subplots(figsize=(12, 9))
# Set common labels
ax.set_xlabel('Car weight')
ax.set_ylabel('Miles per gallon')

sns.scatterplot(ax=ax, x=np.squeeze(x_train_gpr_car_weight), y=np.squeeze(y_train_gpr))
sns.lineplot(ax=ax, x=np.squeeze(x_train_gpr_car_weight), y=np.squeeze(mean))
# -


