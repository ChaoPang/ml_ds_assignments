# import system module
from os import path

# import data related modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, accuracy_score

# import plot modules
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FOLDER = 'hw3-data'
PROB_1_X = path.join(INPUT_FOLDER, 'Prob1_X.csv')
PROB_1_Y = path.join(INPUT_FOLDER, 'Prob1_y.csv')

# Load data for problem 1
x = pd.read_csv(PROB_1_X, header=None, sep=',').values
y = pd.read_csv(PROB_1_Y, header=None, sep=',').values


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
        current_data_dist = np.ones(n) / n

        for t in range(0, self._iteration):

            self._data_dists.append(current_data_dist)
            x_t, y_t = sample_data(x_train, y_train, current_data_dist)
            w_ls_t = least_square_solution(x_t, y_t)

            # compute weight error
            y_hat = np.sign(x_train @ w_ls_t)
            epsilon_t = np.sum(current_data_dist @ (y_hat != y_train).astype(int))
            # if error is larger than 0.5, flip the sign of the prediction
            if epsilon_t > 0.5:
                w_ls_t *= -1
                # re-calculate the prediction and error
                y_hat = np.sign(x_train @ w_ls_t)
                epsilon_t = np.sum(current_data_dist @ (y_hat != y_train).astype(int))

            alpha_t = 1 / 2 * np.log((1 - epsilon_t) / epsilon_t)
            current_data_dist = current_data_dist * np.squeeze(np.exp(-alpha_t * y_hat * y_train))
            current_data_dist = current_data_dist / np.sum(current_data_dist)

            self._classifiers.append(w_ls_t)
            self._alphas.append(alpha_t)
            self._epsilons.append(epsilon_t)

            training_error = 1 - accuracy_score(y_train, self.predict(x_train))
            training_error_upper_bound = self._compute_training_error_upper_bound()
            print(f'Iteration {t} Training Error {training_error} Upper bound {training_error_upper_bound}')
            self._training_errors.append((t, training_error))
            self._training_error_upper_bounds.append((t, training_error_upper_bound))

        return self

    def predict(self, x_test):
        classifier_Ws = np.hstack(self._classifiers)
        classifier_alphas = np.vstack(self._alphas)
        boosting_y_hat = np.sign(np.sign(x_test @ classifier_Ws) @ classifier_alphas)
        return boosting_y_hat


def least_square_solution(x_train, y_train):
    w_ls = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train
    return w_ls


def sample_data(x_train, y_train, x_dist):
    n, _ = np.shape(x_train)
    all_indexes = list(range(0, n))
    sampled_indexes = np.random.choice(all_indexes, size=n, replace=True, p=x_dist)
    return x_train[sampled_indexes], y_train[sampled_indexes]


classifier = AdaBoostLinearClassifier(iteration=2500)
boosting_y_hat = classifier.train(x, y).predict(x)

confusion_matrix(y, boosting_y_hat)

accuracy_score(y, boosting_y_hat)

training_error_pd = pd.DataFrame(classifier.train_errors, columns=['iteration', 'error'])
training_error_upper_bound_pd = pd.DataFrame(classifier.training_error_upper_bounds, columns=['iteration', 'upper_bound'])                           

training_history_pd = pd.melt(training_error_pd.merge(training_error_upper_bound_pd, on='iteration'), 
        id_vars='iteration', var_name='type',
        value_vars=['error', 'upper_bound'])

sns.lineplot(data=training_history_pd,
             x='iteration', y='value', hue='type')

plt.stem(np.average(np.vstack(classifier.data_dists), axis=0),
         use_line_collection=True)

alpha_pd = pd.DataFrame(classifier.alphas, columns=['alpha']).reset_index()
epsilon_pd = pd.DataFrame(classifier.epsilons, columns=['epsilon']).reset_index()

sns.lineplot(data=alpha_pd, x='index', y='alpha')

sns.lineplot(data=epsilon_pd, x='index', y='epsilon')


