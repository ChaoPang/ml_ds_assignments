import numpy as np
import pandas as pd
from os import path
from typing import NamedTuple

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

INPUT_DATA = 'hw2-data'

BAYES_CLASSIFIER_DATA = path.join(INPUT_DATA, 'Bayes_classifier')
BAYES_CLASSIFIER_X = path.join(BAYES_CLASSIFIER_DATA, 'X.csv')
BAYES_CLASSIFIER_Y = path.join(BAYES_CLASSIFIER_DATA, 'y.csv')
README = path.join(BAYES_CLASSIFIER_DATA, 'README')

gaussian_process_data = path.join(INPUT_DATA, 'Gaussian_process')


class NaiveBayesClassifier(NamedTuple):
    """
    A named tuple for storing the trained parameters for a naive bayes classifier
    """
    pi_0: float
    pi_1: float
    lambda_0_d: np.ndarray
    lambda_1_d: np.ndarray


def train_naive_bayes_classifier(x, y):
    """
    Compute the parameters for the bayes classifier
    :param x:
    :param y:
    :return:
    """
    pi_0 = np.sum(y != 1) / len(y)
    pi_1 = np.sum(y == 1) / len(y)

    lambda_0_d = (np.sum(x * (y != 1), axis=0) + 1) / (np.sum(y != 1) + 1)
    lambda_1_d = (np.sum(x * (y == 1), axis=0) + 1) / (np.sum(y == 1) + 1)

    return NaiveBayesClassifier(pi_0=pi_0,
                                pi_1=pi_1,
                                lambda_0_d=lambda_0_d,
                                lambda_1_d=lambda_1_d)


def naive_bayes_classifier_prediction(naive_bayes_classifier, x):
    """
    Predict the classes using naive_bayes_classifier for a given x
    :param naive_bayes_classifier:
    :param x:
    :return:
    """
    # Compute the log sum of the probabilities for each dimension for class 0 (non spam)
    log_prob_0 = np.log(naive_bayes_classifier.pi_0) + np.sum(
        np.log(naive_bayes_classifier.lambda_0_d) * x - naive_bayes_classifier.lambda_0_d, axis=1)
    # Compute the log sum of the probabilities for each dimension for class 1 (spam)
    log_prob_1 = np.log(naive_bayes_classifier.pi_1) + np.sum(
        np.log(naive_bayes_classifier.lambda_1_d) * x - naive_bayes_classifier.lambda_1_d, axis=1)
    return (log_prob_0 < log_prob_1).astype(int)


def naive_bayes_k_fold_validation(x, y):
    k_fold = KFold(n_splits=10, random_state=1, shuffle=True)

    y_true = []
    y_pred = []

    lambda_0 = []
    lambda_1 = []

    for train_index, test_index in k_fold.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        naive_bayes_classifier = train_naive_bayes_classifier(x_train, y_train)

        lambda_0.append(naive_bayes_classifier.lambda_0_d)
        lambda_1.append(naive_bayes_classifier.lambda_1_d)

        y_true.append(y_test)
        y_pred.append(naive_bayes_classifier_prediction(naive_bayes_classifier, x_test))

    conf_matrix = confusion_matrix(np.concatenate(y_true), np.concatenate(y_pred))
    accuracy = np.sum(np.diagonal(conf_matrix)) / np.sum(conf_matrix)

    print(f'Confusion matrix:\n {conf_matrix}')
    print(f'Accuracy: {accuracy}')

    lambda_0_avg = np.average(lambda_0, axis=0)
    lambda_1_avg = np.average(lambda_1, axis=0)

    lambda_0_pd = pd.DataFrame(lambda_0_avg, columns=['lambda']).reset_index()
    lambda_0_pd['index'] = lambda_0_pd.index + 1

    lambda_1_pd = pd.DataFrame(lambda_1_avg, columns=['lambda']).reset_index()
    lambda_1_pd['index'] = lambda_1_pd.index + 1

    plt.stem(lambda_0_pd['index'], lambda_0_pd['lambda'], linefmt='g-.', markerfmt='go',
             use_line_collection=True, label='non spam', basefmt='b')
    plt.stem(lambda_1_pd['index'], lambda_1_pd['lambda'], linefmt='r:', markerfmt='ro',
             use_line_collection=True, label='spam', basefmt='b')
    plt.legend()

    # Save the plot
    plt.savefig(path.join(BAYES_CLASSIFIER_DATA, 'lambda_stem.png'))

    # Clear the figure
    plt.clf()


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


def compute_objective_function(x, y, w):
    return np.sum(sigmoid(x @ w * y))


def train_logistic_regression(x, y, learning_rate=0.01 / 4600):
    n, m = np.shape(x)
    w = np.zeros([m, 1])
    learning_objectives = []
    for i in range(1, 1001):
        learning_objectives.append(compute_objective_function(x, y, w))
        d_w = ((1 - sigmoid(np.multiply(x @ w, y))) * y).T @ x
        w += (learning_rate * d_w).T
    return w, learning_objectives


def draw_learning_objective(x, y):
    k_fold = KFold(n_splits=10, random_state=1, shuffle=True)
    learning_objectives_all = []
    for train_index, test_index in k_fold.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, learning_objectives = train_logistic_regression(x_train, y_train)
        learning_objectives_all.append(learning_objectives)

    learning_objective_pd = pd.DataFrame(np.asarray(learning_objectives_all).T)
    learning_objective_pd = learning_objective_pd.reset_index()
    learning_objective_pd['index'] = learning_objective_pd.index + 1

    unpivoted_learning_objective_pd = pd.melt(learning_objective_pd, id_vars='index',
                                              value_vars=learning_objective_pd.columns[1:])
    unpivoted_learning_objective_pd['variable'] = unpivoted_learning_objective_pd['variable'] + 1
    unpivoted_learning_objective_pd['variable'] = unpivoted_learning_objective_pd[
        'variable'].astype(
        str).apply(lambda v: f'run {v}')

    unpivoted_learning_objective_pd.rename(
        columns={'index': 'iteration',
                 'value': 'learning_objective',
                 'variable': 'run_number'
                 },
        inplace=True)

    plt.subplots(figsize=(12, 9))
    sns.lineplot(data=unpivoted_learning_objective_pd,
                 x='iteration',
                 y='learning_objective',
                 hue='run_number')

    # Save the plot
    plt.savefig(path.join(BAYES_CLASSIFIER_DATA, 'learning_objective.png'))

    # Clear the figure
    plt.clf()


def main():
    # Load data for naive bayes classifier and logistic regression
    x = pd.read_csv(BAYES_CLASSIFIER_X, header=None, sep=',').values
    y = pd.read_csv(BAYES_CLASSIFIER_Y, header=None, sep=',').values

    naive_bayes_k_fold_validation(x, y)

    # add one extra dimension to x
    x = np.hstack([x, np.ones([np.shape(x)[0], 1]).astype(int)])
    # overwite the label 0 to 1 for logistic regression
    y[y == 0] = -1
    draw_learning_objective(x, y)


if __name__ == '__main__':
    """
    Author: Chao Pang 
    """
    main()