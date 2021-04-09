from os import path

import numpy as np
import pandas as pd
import scipy.sparse.linalg as sla

# import plot modules
import matplotlib.pyplot as plt
import seaborn as sns

# +
INPUT_FOLDER = 'hw4-data'
PROB_1_DATA = path.join(INPUT_FOLDER, 'CFB2019_scores.csv')
PROB_1_VOCAB = path.join(INPUT_FOLDER, 'TeamNames.txt')

PROB_2_DOCUMENT = path.join(INPUT_FOLDER, 'nyt_data.txt')
PROB_2_VOCAB = path.join(INPUT_FOLDER, 'nyt_vocab.dat')


# +
def load_cfb_data():
    """
    Load the CFB data and its team dictionary
    :return:
    """

    # Load the team_dict and assume the index is 0 based
    team_dict = dict()
    with open(PROB_1_VOCAB, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            team_dict[i] = line.strip()
    n_of_teams = len(team_dict)
    # +
    M_hat = np.zeros((n_of_teams, n_of_teams))
    with open(PROB_1_DATA, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line_parts = list(map(int, line.strip().split(',')))
            # convert to zero based index
            team_a_index = line_parts[0] - 1
            team_a_points = line_parts[1]
            # convert to zero based index
            team_b_index = line_parts[2] - 1
            team_b_points = line_parts[3]

            team_a_win = int(team_a_points > team_b_points)
            team_b_win = int(team_a_points < team_b_points)

            M_hat[team_a_index, team_a_index] += team_a_win + team_a_points / (
                    team_a_points + team_b_points)
            M_hat[team_b_index, team_a_index] += team_a_win + team_a_points / (
                    team_a_points + team_b_points)

            M_hat[team_b_index, team_b_index] += team_b_win + team_b_points / (
                    team_a_points + team_b_points)
            M_hat[team_a_index, team_b_index] += team_b_win + team_b_points / (
                    team_a_points + team_b_points)
    M = M_hat / np.expand_dims(np.sum(M_hat, axis=1), axis=-1)

    return M, team_dict


def rank_cfb_teams(M, team_dict):
    """
    Rank the cfb team based on their scores and game results
    :param M:
    :param team_dict:
    :return:
    """
    n_of_teams = len(team_dict)
    w_t = np.ones((1, n_of_teams)) / n_of_teams
    for t in range(1, 10001):
        w_t = w_t @ M
        if t in [10, 100, 1000, 10000]:
            print([(team_dict[i], np.squeeze(w_t)[i]) for i in
                   np.argsort(np.squeeze(w_t))[::-1][:25]])
            print()

    eigenvalues, eigenvectors = sla.eigs(M.T, k=1, which='LM',
                                         v0=np.random.uniform(0, 1, n_of_teams))

    # assert that all entries in the eigenvector have the same sign
    assert np.prod(np.sign(eigenvectors)).real > 0

    w_inf = np.squeeze((eigenvectors / np.sum(eigenvectors, axis=0)).real)
    distances = []
    w_t = np.ones((1, n_of_teams)) / n_of_teams
    for t in range(1, 10001):
        w_t = w_t @ M
        distances.append((t, np.linalg.norm(w_inf - np.squeeze(w_t))))

    # Plot the learning objective
    distance_pd = pd.DataFrame(distances, columns=['iteration', 'l1_norm'])

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.lineplot(data=distance_pd, x='iteration', y='l1_norm', ax=ax)

    # Save the plot
    plt.savefig(path.join(INPUT_FOLDER, f'l1_norm.png'))

    # Clear the figure
    plt.clf()


def convert_to_zero_based_indexes(indexes):
    """
    Shift the index by one
    :param indexes:
    :return:
    """
    return list(map(lambda i: i - 1, indexes))


def load_nyt_data():
    """
    Load the new york times document and vocab
    :return:
    """
    # Load the vocab and assume the index is 0 based
    vocab = dict()
    with open(PROB_2_VOCAB, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            vocab[i] = line.strip()
    vocab_size = len(vocab)

    # Load the data
    documents = []
    with open(PROB_2_DOCUMENT, 'r', encoding='utf-8-sig') as reader:
        # Read and print the entire file line by line
        while True:
            line = reader.readline()
            if not line:
                break
            indexes, counts = zip(*[map(int, pair.split(':')) for pair in line.strip().split(',')])
            zero_based_indexes = convert_to_zero_based_indexes(indexes)
            d = np.zeros(vocab_size)
            d[zero_based_indexes] = counts
            documents.append(d)
    return np.asarray(documents).T, vocab


def run_non_negative_matrix_factorization(X, vocab, n_topics, n_iter):
    """
    Run the NMF algorithm

    :param X:
    :param vocab:
    :param n_topics:
    :param n_iter:
    :return:
    """
    vocab_size, number_of_docs = np.shape(X)

    # Initialize W and H matrices using a uniform sampling strategy
    W = np.random.uniform(1, 2, size=(vocab_size, n_topics))
    H = np.random.uniform(1, 2, size=(n_topics, number_of_docs))

    # Run the algorithm for n_iter times
    losses = []
    for i in range(n_iter):
        loss = np.sum(X * np.log(1 / (W @ H + 1.0E-16)) + W @ H)
        H = H * (W.T / np.expand_dims(np.sum(W.T, axis=1), axis=-1) @ (
                X / (W @ H + 1.0E-16)))
        W = W * (X / (W @ H + 1.0E-16) @ H.T) / np.expand_dims(
            np.sum(H.T, axis=0), axis=0)
        losses.append((i + 1, loss))
        print(f'iteration {i + 1} Loss {loss}')

    # Plot the learning objective
    loss_pd = pd.DataFrame(losses, columns=['iteration', 'learning_objective'])

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.lineplot(data=loss_pd[loss_pd['iteration'] > 1], x='iteration', y='learning_objective',
                 ax=ax)

    # Save the plot
    plt.savefig(path.join(INPUT_FOLDER, f'nmf_learning_objective.png'))

    # Clear the figure
    plt.clf()

    normalized_W = W / np.sum(W, axis=0)

    topics = []
    for i in range(np.shape(normalized_W)[1]):
        print(f'Topic {i + 1}')
        print('\n'.join([f'{vocab[j]} : {normalized_W[j, i]}' for j in
                         np.argsort(normalized_W[:, i])[::-1][:10]]))
        print()
        topics.extend([(i + 1, vocab[j], normalized_W[j, i]) for j in
                       np.argsort(normalized_W[:, i])[::-1][:10]])


def main():
    # Set the seed
    np.random.seed(100)

    # Problem 1
    rank_cfb_teams(*load_cfb_data())

    # Problem 2
    run_non_negative_matrix_factorization(*load_nyt_data(), n_topics=25, n_iter=100)


# -

if __name__ == '__main__':
    """
    Author: Chao Pang
    """
    main()
#
# # +
# M, vocab = load_cfb_data()
# n_of_teams = len(vocab)
# w_t = np.ones((1, n_of_teams)) / n_of_teams
# for t in range(1, 10001):
#     w_t = w_t @ M
#     if t in [10, 100, 1000, 10000]:
#         print([(vocab[i], np.squeeze(w_t)[i]) for i in np.argsort(np.squeeze(w_t))[::-1][:25]])
#         print()
# np.random.seed(100)
#
# eigenvalues, eigenvectors = sla.eigs(M.T, k=1, which='LM', v0=np.random.uniform(0, 1, len(vocab)))
#
# # assert that all entries in the eigenvector have the same sign
# assert np.prod(np.sign(eigenvectors)).real > 0
#
# w_inf = np.squeeze((eigenvectors / np.sum(eigenvectors, axis=0)).real)
#
# distances = []
# w_t = np.ones((1, n_of_teams)) / n_of_teams
# for t in range(1, 10001):
#     w_t = w_t @ M
#     distances.append((t, np.linalg.norm(w_inf - np.squeeze(w_t))))
#
# # Plot the learning objective
# distance_pd = pd.DataFrame(distances, columns=['iteration', 'l1_norm'])
#
# sns.lineplot(data=distance_pd, x='iteration', y='l1_norm')
# # -
