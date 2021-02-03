# import system module
from os import path
import inflect

# import data related modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# import plot modules
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FOLDER = 'hw1-data'
X_TRAIN = 'X_train.csv'
Y_TRAIN = 'y_train.csv'
X_TEST = 'X_test.csv'
Y_TEST = 'y_test.csv'


def normalize_data(x, y):
    """
    Scale the features data using the mean and standard deviation for features except the last
    feature because it's added for the bias term. Scale the labels using the mean.

    :param x:
    :param y:
    :return:
    """

    # Scale the features using mean and standard deviation except for the last feature
    x_scaler = StandardScaler()
    scaled_x = np.column_stack([x_scaler.fit_transform(x[:, 0:6]), x[:, 6:7]])

    # Scale the label using mean
    y_scaler = StandardScaler(with_std=False)
    scaled_y = y_scaler.fit_transform(y)

    return scaled_x, scaled_y


def compute_ridge_regression_solution(x, y, _lambda):
    """
    Compute the ridge regression solution
    
    :param x: feature matrix
    :param y: labels
    :param _lambda: hyperparameter for regularization term in ridge regression
    :return: the ridge regression solution
    """

    _, m = np.shape(x)

    # Compute the ridge regression solution
    inverse = np.linalg.inv(np.identity(m) * _lambda + np.matmul(x.transpose(), x))
    w_rr = np.matmul(np.matmul(inverse, x.transpose()), y)
    return w_rr


def compute_degree_of_freedom(x, _lambda):
    """
    Compute the degree of freedom for lambda given the matrix x
    :param x:
    :param _lambda:
    :return:
    """
    # Run SVD
    u, s, v_t = np.linalg.svd(x)
    # Compute the degree of freedom for lambda based on the equation given by the lecture slide
    degree_of_freedom = np.sum(s ** 2 / (s ** 2 + _lambda))
    return degree_of_freedom


def plot_df_lambda_w_rr(x, y, lambdas):
    """
    Plot the lambda's degree of freedom against the ridge regression solution

    :param x:
    :param y:
    :param lambdas:
    :return:
    """

    # Get the number of features for creating labels
    _, num_features = np.shape(x)
    w_rr_df_lambda_labels = [f'w{str(i)}' for i in range(1, num_features + 1)] + ['df']

    # compute the degree of freedom for lambda and the corresponding ridge regression solution
    w_rr_df_lambda_list = []
    for lambda_val in lambdas:
        w_rr = compute_ridge_regression_solution(x, y, lambda_val)
        df_lambda = compute_degree_of_freedom(x, lambda_val)
        w_rr_df_lambda_list.append(np.concatenate((np.squeeze(w_rr), [df_lambda])))
    w_rr_df_lambda_pd = pd.DataFrame(w_rr_df_lambda_list, columns=w_rr_df_lambda_labels)

    # Unpivot all feature columns into a single column for plotting
    w_rr_df_lambda_pd_unpivoted = pd.melt(w_rr_df_lambda_pd, id_vars=w_rr_df_lambda_labels[-1],
                                          value_vars=w_rr_df_lambda_labels[0:-1])

    # Computing the lower bound and upper bound for the x axis limit
    x_axis_lower_bound = int(np.floor(w_rr_df_lambda_pd_unpivoted['df'].min()))
    x_axis_upper_bound = int(np.ceil(w_rr_df_lambda_pd_unpivoted['df'].max()))
    fig, ax = plt.subplots()
    ax.set_xlim(x_axis_lower_bound, x_axis_upper_bound + 1)

    # Plot the lines
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.lineplot(ax=ax, data=w_rr_df_lambda_pd_unpivoted, x='df', y='value', hue='variable')

    # Set the label to the corresponding line
    for feature_name in w_rr_df_lambda_pd_unpivoted.variable.unique().tolist():
        sub = w_rr_df_lambda_pd_unpivoted[w_rr_df_lambda_pd_unpivoted['variable'] == feature_name]
        df, label, weight_value = sub[sub.df == sub.df.max()].iloc[0].tolist()
        plt.text(df + 0.1, weight_value - 0.1, label, horizontalalignment='left', size=12)

    # Save the plot
    plt.savefig(path.join(INPUT_FOLDER, 'df_lambda_weight.png'))
    # Clear the figure
    plt.clf()


def compute_rmse(w_rr, x, y):
    """
    Compute root mean squared error for the given x and y
    :param w_rr:
    :param x:
    :param y:
    :return:
    """
    num_of_rows, _ = np.shape(x)
    y_hat = np.matmul(x, w_rr)
    return np.sqrt(np.sum((y_hat - y) ** 2) / num_of_rows)


def plot_lambda_rmse(x_train, y_train, x_test, y_test, lambdas):
    """
    Plot the root mean squared error (RMSE) on the test set as a function of lambda
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param lambdas:
    :return:
    """
    # Get the number of rows in the test set
    num_of_rows, _ = np.shape(y_test)
    lambda_rmse = []
    for lambda_value in lambdas:
        # Compute the ridge regression solution
        w_rr = compute_ridge_regression_solution(x_train, y_train, lambda_value)
        # Compute the rmse in the test set
        lambda_rmse.append((lambda_value, compute_rmse(w_rr, x_test, y_test)))

    # Create a pandas dataframe for ploting
    lambda_rmse_pd = pd.DataFrame(lambda_rmse, columns=['lambda', 'rmse'])

    # Plot rmse against lambda
    sns.lineplot(data=lambda_rmse_pd, x='lambda', y='rmse')

    # Save the plot
    plt.savefig(path.join(INPUT_FOLDER, 'lambda_rmse.png'))
    # Clear the figure
    plt.clf()


def create_polynomial_terms(x, polynomial_order):
    """
    Create ploynomial terms based on the given ploynomial order, there is no interaction between
    features
    :param x:
    :param polynomial_order:
    :return:
    """
    polynomial_terms = np.hstack([x[:, 0:6] ** (i + 1) for i in range(polynomial_order)])
    return np.hstack([polynomial_terms, x[:, 6:7]])


def plot_polynomial_lambda_rmse(x_train, y_train, x_test, y_test, polynomial_orders, lambdas):
    """
    Plot the root mean squared error (RMSE) on the test set as a function of lambda for
    different ridge regression solutions calculated using different polynomial orders

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param polynomial_orders:
    :param lambdas:
    :return:
    """

    # Create an inflect engine for translating the number to an ordinal representation
    p = inflect.engine()

    polynomial_order_lambda_rmse = []
    # Iterate through all polynomial orders
    for polynomial_order in polynomial_orders:

        # Translate the number to an ordinal representation
        ordinal_order = p.ordinal(polynomial_order)

        # Create polynomial features for the training and test sets
        augmented_x_train = create_polynomial_terms(x_train, polynomial_order)
        augmented_x_test = create_polynomial_terms(x_test, polynomial_order)

        # Iterate through all lambda values
        for lambda_value in lambdas:
            # Compute the ridge regression solution
            w_rr = compute_ridge_regression_solution(augmented_x_train, y_train,
                                                     lambda_value)
            # Compute the rmse in the test set
            polynomial_order_lambda_rmse.append((f'{ordinal_order} order', lambda_value,
                                                 compute_rmse(w_rr, augmented_x_test,
                                                              y_test)))

    # Create a pandas dataframe for ploting
    polynomial_order_lambda_rmse_pd = pd.DataFrame(polynomial_order_lambda_rmse,
                                                   columns=['order', 'lambda', 'rmse'])

    # Plot rmse against lambda
    sns.lineplot(data=polynomial_order_lambda_rmse_pd, x='lambda', y='rmse', hue='order')

    # Set the label to the corresponding line
    for ordinal_order in polynomial_order_lambda_rmse_pd.order.unique().tolist():
        sub = polynomial_order_lambda_rmse_pd[
            polynomial_order_lambda_rmse_pd['order'] == ordinal_order]
        _, lambda_value, rmse = sub[sub['lambda'] == sub['lambda'].median()].iloc[0].tolist()
        plt.text(lambda_value, rmse, ordinal_order, horizontalalignment='left', size='large')

    # Save the plot
    plt.savefig(path.join(INPUT_FOLDER, 'polynomial_lambda_rmse.png'))

    # Clear the figure
    plt.clf()


def main():
    """
    Main function
    :return:
    """

    # Loading the training data
    x_train_pd = pd.read_csv(path.join(INPUT_FOLDER, X_TRAIN), sep=',', header=None)
    y_train_pd = pd.read_csv(path.join(INPUT_FOLDER, Y_TRAIN), sep=',', header=None)

    x_train = x_train_pd.to_numpy()
    y_train = y_train_pd.to_numpy()

    scaled_x_train, scaled_y_train = normalize_data(x_train, y_train)

    # Loading the test data
    x_test_pd = pd.read_csv(path.join(INPUT_FOLDER, X_TEST), sep=',', header=None)
    y_test_pd = pd.read_csv(path.join(INPUT_FOLDER, Y_TEST), sep=',', header=None)

    x_test = x_test_pd.to_numpy()
    y_test = y_test_pd.to_numpy()

    scaled_x_test, scaled_y_test = normalize_data(x_test, y_test)

    # Problem 3 Task (a) for plotting the w_rr as a function of df(lambda)
    plot_df_lambda_w_rr(x=scaled_x_train,
                        y=scaled_y_train,
                        lambdas=range(0, 5001))

    # Problem 3 Task (c) for plotting the rmse as a function of lambda
    plot_lambda_rmse(x_train=scaled_x_train,
                     y_train=scaled_y_train,
                     x_test=scaled_x_test,
                     y_test=scaled_y_test,
                     lambdas=range(0, 51))

    # Problem 3 Task (d) for plotting the rmse as a function of lambda for w_rr calculated for
    # different polynomial orders
    plot_polynomial_lambda_rmse(x_train=scaled_x_train,
                                y_train=scaled_y_train,
                                x_test=scaled_x_test,
                                y_test=scaled_y_test,
                                polynomial_orders=range(1, 4),
                                lambdas=range(0, 101))


if __name__ == '__main__':
    main()
