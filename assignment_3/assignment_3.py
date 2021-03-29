# import system module
from os import path

# import data related modules
import numpy as np

import pandas as pd

from scipy.stats import multivariate_normal

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics.pairwise import euclidean_distances

# import plot modules
import matplotlib.pyplot as plt
import seaborn as sns

# +
INPUT_FOLDER = 'hw3-data'
PROB_1_X = path.join(INPUT_FOLDER, 'Prob1_X.csv')
PROB_1_Y = path.join(INPUT_FOLDER, 'Prob1_y.csv')

PROB_3_X_TRAIN = path.join(INPUT_FOLDER, 'Prob3_Xtrain.csv')
PROB_3_Y_TRAIN = path.join(INPUT_FOLDER, 'Prob3_ytrain.csv')
PROB_3_X_TEST = path.join(INPUT_FOLDER, 'Prob3_Xtest.csv')
PROB_3_Y_TEST = path.join(INPUT_FOLDER, 'Prob3_ytest.csv')


# -

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
    training_history_pd = training_history_pd.rename(columns={'value': 'training_error'})

    fig, ax = plt.subplots(figsize=(12, 9))

    # plt.subplots(figsize=(14, 8))
    sns.lineplot(data=training_history_pd, x='iteration', y='training_error', hue='type', ax=ax)
    # Save the plot
    plt.savefig(path.join(INPUT_FOLDER, 'training_error.png'))
    # Clear the figure
    plt.clf()

    plt.stem(np.average(np.vstack(classifier.data_dists), axis=0), use_line_collection=True,
             linefmt='-.')
    plt.savefig(path.join(INPUT_FOLDER, 'stem.png'))
    # Clear the figure
    plt.clf()

    alpha_pd = pd.DataFrame(classifier.alphas, columns=['alpha']).reset_index()
    epsilon_pd = pd.DataFrame(classifier.epsilons, columns=['epsilon']).reset_index()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.lineplot(data=alpha_pd, x='index', y='alpha', ax=ax)
    plt.savefig(path.join(INPUT_FOLDER, 'alpha.png'))
    # Clear the figure
    plt.clf()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.lineplot(data=epsilon_pd, x='index', y='epsilon', ax=ax)
    plt.savefig(path.join(INPUT_FOLDER, 'epsilon.png'))
    # Clear the figure
    plt.clf()


def train_k_means(data, k_cluster, num_of_iterations=20):
    """
    Train a train_k_means algorithm
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


def plot_k_means_training_history():
    """
    Plot k_means training objective and clusters for k equals 3 and 5
    :return:
    """
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
    for k in range(2, 6):
        _, k_means_classes, history = train_k_means(data, k)
        training_history.extend(history)
        # Plot K-means clusters for k is in (3, 5)
        if k in [3, 5]:
            plot_k_means_cluster(data, k_means_classes, k)
    training_history_pd = pd.DataFrame(training_history,
                                       columns=['iteration', 'K', 'learning_objective'])
    training_history_pd['K'] = training_history_pd.K.apply(str)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.lineplot(ax=ax, data=training_history_pd, x='iteration', y='learning_objective', hue='K')
    # Save the plot
    plt.savefig(path.join(INPUT_FOLDER, 'train_k_means.png'))
    # Clear the figure
    plt.clf()


def plot_k_means_cluster(data, k_means_classes, k):
    """
    Plot train_k_means clusters

    :param data:
    :param k_means_classes:
    :param k:
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
    plt.savefig(path.join(INPUT_FOLDER, f'k_means_{k}_clusters.png'))
    # Clear the figure
    plt.clf()


class GaussianMixtureModel:

    def __init__(self, name, initial_mu, initial_sigma, initial_pi, k_cluster, num_of_iterations):
        self._name = name
        self._mu = initial_mu
        self._sigma = initial_sigma
        self._pi = initial_pi
        self._k_cluster = k_cluster
        self._num_of_iterations = num_of_iterations
        self._learning_objectives = []

    @property
    def learning_objectives(self):
        return self._learning_objectives

    def train(self, data):

        n, m = np.shape(data)

        for iteration in range(self._num_of_iterations):
            # calculate the probabilities
            probabilities = np.asarray(
                [multivariate_normal.pdf(data, mean=self._mu[i], cov=self._sigma[i],
                                         allow_singular=True)
                 for i in range(self._k_cluster)]).T
            log_marginal_prob = np.sum(np.log(np.sum(probabilities * self._pi, axis=1)))
            self._learning_objectives.append((iteration + 1, log_marginal_prob))

            # E step
            phi = probabilities * self._pi / np.expand_dims(
                np.sum(probabilities * self._pi, axis=1), axis=1)
            # M step
            n_k = np.sum(phi, axis=0)
            self._pi = n_k / n
            self._mu = phi.T @ data / np.expand_dims(n_k, axis=1)

            sigma_new = []
            for i in range(self._k_cluster):
                phi_k = np.expand_dims(phi[:, i], axis=1)
                sigma_k = (phi_k * (data - self._mu[i])).T @ (
                        data - self._mu[i]) / n_k[i]
                sigma_new.append(sigma_k)
            self._sigma = np.asarray(sigma_new)

    def calculate_log_prob(self, data):
        """
        Calculate the log probability of the data
        :param data:
        :return:
        """
        # calculate the probabilities
        probabilities = np.asarray(
            [multivariate_normal.pdf(data, mean=self._mu[i], cov=self._sigma[i],
                                     allow_singular=True)
             for i in range(self._k_cluster)]).T
        return np.sum(probabilities * self._pi, axis=1)


def train_gaussian_mixture_model(data,
                                 num_of_clusters,
                                 number_of_runs=10,
                                 number_of_iterations=30,
                                 figure_name=None):
    """
    Train a gassian mixture model given a number of clusters and the data. Return the best
    classifier based on their learning objectives

    :param data:
    :param num_of_clusters:
    :param number_of_runs:
    :param number_of_iterations:
    :param figure_name:
    :return:
    """

    initial_pi = np.ones([1, num_of_clusters]) / num_of_clusters
    n, m = np.shape(data)
    empirical_mean = np.mean(data, axis=0)
    empirical_cov = (data - empirical_mean).T @ (data - empirical_mean) / n

    all_training_objectives = []
    all_classifiers = []
    for run in range(1, number_of_runs + 1):
        initial_sigma = np.asarray([empirical_cov] * num_of_clusters)
        initial_mu = np.random.multivariate_normal(mean=empirical_mean,
                                                   cov=empirical_cov,
                                                   size=num_of_clusters)
        gmm = GaussianMixtureModel(name=f'Run-{run}',
                                   initial_mu=initial_mu,
                                   initial_sigma=initial_sigma,
                                   initial_pi=initial_pi,
                                   k_cluster=num_of_clusters,
                                   num_of_iterations=number_of_iterations)
        gmm.train(data)

        all_classifiers.append((gmm.learning_objectives[-1][1], gmm))
        all_training_objectives.extend([(f'Run-{run}', i, l) for i, l in gmm.learning_objectives])

    _, best_classifier = sorted(all_classifiers, key=lambda t: t[0], reverse=True)[0]

    if figure_name:
        all_training_objectives_pd = pd.DataFrame(all_training_objectives,
                                                  columns=['run', 'iteration',
                                                           'learning_objective'])

        fig, ax = plt.subplots(figsize=(12, 9))

        sns.lineplot(data=all_training_objectives_pd[all_training_objectives_pd.iteration > 5],
                     x='iteration', y='learning_objective', hue='run', ax=ax)

        # Save the plot
        plt.savefig(path.join(INPUT_FOLDER, f'learning_objective_{figure_name}.png'))

        # Clear the figure
        plt.clf()

    return best_classifier


def gmm_bayes_classifier(x_train, y_train, x_test, y_test):
    """
    Train four Bayes classifiers whose data likelihood function is a k gaussian mixture model

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    x_train_spam = x_train[np.squeeze(y_train == 1)]
    x_train_non_spam = x_train[np.squeeze(y_train == 0)]
    prior_0 = np.sum(y_train == 0) / len(y_train)
    prior_1 = np.sum(y_train == 1) / len(y_train)
    train_gaussian_mixture_model(data=x_train_spam, num_of_clusters=3,
                                 figure_name='spam')
    train_gaussian_mixture_model(data=x_train_non_spam, num_of_clusters=3,
                                 figure_name='non_spam')
    for k in range(1, 5):
        spam_gmm = train_gaussian_mixture_model(data=x_train_spam, num_of_clusters=k)
        non_spam_gmm = train_gaussian_mixture_model(data=x_train_non_spam, num_of_clusters=k)
        y_hat = (prior_1 * spam_gmm.calculate_log_prob(
            x_test) > prior_0 * non_spam_gmm.calculate_log_prob(x_test)).astype(int)
        print(
            f'number_of_cluster: {k} accuracy: {accuracy_score(y_test, y_hat)} \n {confusion_matrix(y_test, y_hat)}')


def main():
    np.random.seed(100)

    # Load data for problem 1
    x = pd.read_csv(PROB_1_X, header=None, sep=',').values
    y = pd.read_csv(PROB_1_Y, header=None, sep=',').values
    adaboost_linear_classifier(x, y)

    # Problem 2
    plot_k_means_training_history()

    # Problem 3
    x_train = pd.read_csv(PROB_3_X_TRAIN, header=None, sep=',').values
    y_train = pd.read_csv(PROB_3_Y_TRAIN, header=None, sep=',').values

    x_test = pd.read_csv(PROB_3_X_TEST, header=None, sep=',').values
    y_test = pd.read_csv(PROB_3_Y_TEST, header=None, sep=',').values

    gmm_bayes_classifier(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    """
    Author: Chao Pang
    """
    main()
