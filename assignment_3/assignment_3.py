# import system module
from os import path

# import data related modules
import numpy as np

np.random.seed(100)
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics.pairwise import euclidean_distances

# import plot modules
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FOLDER = 'hw3-data'
PROB_1_X = path.join(INPUT_FOLDER, 'Prob1_X.csv')
PROB_1_Y = path.join(INPUT_FOLDER, 'Prob1_y.csv')


# ## least square solution
# $ w_{ls} = (X^{T}X)^{-1}X^{T}y $

class AdaBoostLinearClassifier:
    def __init__(self, iteration):
        self._iteration = iteration
        self._data_dists = []
        self._classifiers = []
        self._alphas = []
        self._epsilons = []
        self._training_errors = []
        self._training_error_upper_bounds = []

    @property
    def train_errors(self):
        return self._training_errors

    @property
    def training_error_upper_bounds(self):
        return self._training_error_upper_bounds

    @property
    def data_dists(self):
        return self._data_dists

    @property
    def alphas(self):
        return self._alphas

    @property
    def epsilons(self):
        return self._epsilons

    def _compute_training_error_upper_bound(self):
        return np.exp(-2 * np.sum((1 / 2 - np.asarray(self._epsilons)) ** 2))

    def _reset_state(self):
        self._data_dists.clear()
        self._classifiers.clear()
        self._alphas.clear()
        self._epsilons.clear()
        self._training_errors.clear()
        self._training_error_upper_bounds.clear()

    def train(self, x_train, y_train):

        self._reset_state()

        n, _ = np.shape(x_train)
        # initial distribution on the data
        current_data_dist = np.ones(n) / n

        for t in range(0, self._iteration):
            # Save the current data distribution for the t_th iteration
            self._data_dists.append(current_data_dist)
            # Sample data using the current data distribution
            x_t, y_t = sample_data(x_train, y_train, current_data_dist)
            # Calculate the least square solution
            w_ls_t = least_square_solution(x_t, y_t)

            # compute weight error on the entire dataset
            y_hat = np.sign(x_train @ w_ls_t)
            epsilon_t = np.sum(current_data_dist * np.squeeze((y_hat != y_train).astype(int)))
            # if epsilon is larger than 0.5, flip the sign of the prediction
            if epsilon_t > 0.5:
                w_ls_t *= -1
                # re-calculate the prediction and error
                y_hat = np.sign(x_train @ w_ls_t)
                epsilon_t = np.sum(current_data_dist * np.squeeze((y_hat != y_train).astype(int)))

            # Calculate alpha given epislon
            alpha_t = (1 / 2) * np.log((1 - epsilon_t) / epsilon_t)
            # Update the data distribution using the alpha
            data_dist_hat = current_data_dist * np.squeeze(np.exp(-alpha_t * y_hat * y_train))
            current_data_dist = data_dist_hat / np.sum(data_dist_hat)

            self._classifiers.append(w_ls_t)
            self._alphas.append(alpha_t)
            self._epsilons.append(epsilon_t)

            training_error = 1 - accuracy_score(y_train, self.predict(x_train))
            training_error_upper_bound = self._compute_training_error_upper_bound()
            print(
                f'Iteration {t} Training Error {training_error} Upper bound {training_error_upper_bound}')
            self._training_errors.append((t, training_error))
            self._training_error_upper_bounds.append((t, training_error_upper_bound))

        return self

    def predict(self, x_test):
        """
        Predict using the all classifiers
        :param x_test:
        :return:
        """
        classifier_Ws = np.hstack(self._classifiers)
        classifier_alphas = np.vstack(self._alphas)
        boosting_y_hat = np.sign(np.sign(x_test @ classifier_Ws) @ classifier_alphas)
        return boosting_y_hat


def least_square_solution(x_train, y_train):
    """
    The least square solution given a pair of x and y
    :param x_train:
    :param y_train:
    :return:
    """
    return np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train


def sample_data(x_train, y_train, data_dist):
    """
    Sample from x and y using the data distribution
    :param x_train:
    :param y_train:
    :param data_dist:
    :return:
    """
    n, _ = np.shape(x_train)
    all_indexes = list(range(0, n))
    sampled_indexes = np.random.choice(all_indexes, size=n, replace=True, p=data_dist)
    return x_train[sampled_indexes], y_train[sampled_indexes]


def adaboost_linear_classifier(x_train, y_train):
    """

    :param x_train:
    :param y_train:
    :return:
    """
    classifier = AdaBoostLinearClassifier(iteration=2500)
    boosting_y_hat = classifier.train(x_train, y_train).predict(x_train)

    print(f'The confusion matrix:\n {confusion_matrix(y_train, boosting_y_hat)}')

    print(f'The accuracy for adaboost linear regression {accuracy_score(y_train, boosting_y_hat)}')

    training_error_pd = pd.DataFrame(classifier.train_errors, columns=['iteration', 'error'])
    training_error_upper_bound_pd = pd.DataFrame(classifier.training_error_upper_bounds,
                                                 columns=['iteration', 'upper_bound'])

    training_history_pd = pd.melt(
        training_error_pd.merge(training_error_upper_bound_pd, on='iteration'),
        id_vars='iteration', var_name='type',
        value_vars=['error', 'upper_bound'])

    # plt.subplots(figsize=(14, 8))
    sns.lineplot(data=training_history_pd, x='iteration', y='value', hue='type')
    # Save the plot
    plt.savefig(path.join(INPUT_FOLDER, 'training_error.png'))
    # Clear the figure
    plt.clf()

    # plt.subplots(figsize=(14, 8))
    plt.stem(np.average(np.vstack(classifier.data_dists), axis=0), use_line_collection=True,
             linefmt='-.')
    plt.savefig(path.join(INPUT_FOLDER, 'stem.png'))
    # Clear the figure
    plt.clf()

    alpha_pd = pd.DataFrame(classifier.alphas, columns=['alpha']).reset_index()
    epsilon_pd = pd.DataFrame(classifier.epsilons, columns=['epsilon']).reset_index()

    sns.lineplot(data=alpha_pd, x='index', y='alpha')
    plt.savefig(path.join(INPUT_FOLDER, 'alpha.png'))
    # Clear the figure
    plt.clf()

    sns.lineplot(data=epsilon_pd, x='index', y='epsilon')
    plt.savefig(path.join(INPUT_FOLDER, 'epsilon.png'))
    # Clear the figure
    plt.clf()


def k_means(data, k_cluster, num_of_iterations=20):
    """
    Train a k_means algorithm
    :param data:
    :param k_cluster:
    :param num_of_iterations:
    :return:
    """
    history = []
    _, m = np.shape(data)
    k_means_mus = np.random.rand(k_cluster, m)
    k_means_class = None
    for iteration in range(num_of_iterations):
        all_distance = euclidean_distances(data, k_means_mus, squared=True)
        k_means_class = np.argmin(all_distance, axis=1)
        learning_objective = np.sum(np.min(all_distance, axis=1))
        for i in range(k_cluster):
            k_means_mus[i] = np.mean(data[k_means_class == i], axis=0)
        history.append((iteration + 1, str(k_cluster) + '-cluster', learning_objective))
    return k_means_mus, np.reshape(k_means_class, (-1, 1)), history


def plot_k_means():
    n = 500
    # Sample data from a Gaussian mixture model
    w = [0.2, 0.5, 0.3]
    mu = np.asarray([[0, 0], [3, 0], [0, 3]])
    cov = np.reshape([1, 0, 0, 1], (2, 2))

    randomly_generated_indexes = np.random.choice(a=[0, 1, 2], size=n, p=w)
    unique, counts = np.unique(randomly_generated_indexes, return_counts=True)
    data = []
    for index, count in zip(unique, counts):
        print(f'{index}: {count}')
        data.append(np.random.multivariate_normal(mean=mu[index], cov=cov, size=count))
    data = np.concatenate(data)

    training_history = []
    for k_cluster in range(2, 6):
        _, k_means_classes, history = k_means(data, k_cluster)
        training_history.extend(history)
        # Plot K-means clusters for k is in (3, 5)
        if k_cluster in [3, 5]:
            plot_k_means_cluster(data, k_means_classes, k_cluster)
    training_history_pd = pd.DataFrame(training_history,
                                       columns=['iteration', 'K', 'learning_objective'])
    training_history_pd['K'] = training_history_pd.K.apply(str)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.lineplot(ax=ax, data=training_history_pd, x='iteration', y='learning_objective', hue='K')
    # Save the plot
    plt.savefig(path.join(INPUT_FOLDER, 'k_means.png'))
    # Clear the figure
    plt.clf()


def plot_k_means_cluster(data, k_means_classes, k_cluster):
    """
    Plot k_means clusters

    :param data:
    :param k_means_classes:
    :param k_cluster:
    :return:
    """
    k_means_cluster_pd = pd.DataFrame(np.hstack([data, k_means_classes]),
                                      columns=['x1', 'x2', 'cluster'])
    k_means_cluster_pd.cluster = k_means_cluster_pd.cluster.apply(
        lambda c: 'cluster-' + str(int(c)))

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.scatterplot(data=k_means_cluster_pd, x='x1', y='x2', hue='cluster', ax=ax)
    k_means_centroids = k_means_cluster_pd.groupby('cluster').mean()
    sns.scatterplot(data=k_means_centroids, x='x1', y='x2', ec='black', color='black', s=80,
                    legend=False, ax=ax)

    # Save the plot
    plt.savefig(path.join(INPUT_FOLDER, f'k_means_{k_cluster}_clusters.png'))
    # Clear the figure
    plt.clf()


def main():
    # Load data for problem 1
    x = pd.read_csv(PROB_1_X, header=None, sep=',').values
    y = pd.read_csv(PROB_1_Y, header=None, sep=',').values
    adaboost_linear_classifier(x, y)

    # Problem 2
    plot_k_means()


if __name__ == '__main__':
    """
    Author: Chao Pang 
    """
    main()
