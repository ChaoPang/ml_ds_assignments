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


def print_team_ranking(w_t, team_dict, name):
    print('Rank\tTeam\tprobability')
    rows = []
    for rank, index in enumerate(np.argsort(np.squeeze(w_t))[::-1][:25]):
        print(f'{rank + 1}\t{team_dict[index]}\t{"{:.4f}".format(np.squeeze(w_t)[index])}')
        rows.append((rank + 1, team_dict[index], float("{:.4f}".format(np.squeeze(w_t)[index]))))
    print()
    results_pd = pd.DataFrame(rows, columns=['Rank', 'Team', 'Probability'])

    with open(path.join(INPUT_FOLDER, name), 'w') as opened_file:
        opened_file.write(results_pd.to_latex(caption=name,
                                              index=False))


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
            print_team_ranking(w_t, team_dict, f'rankings_w_{t}')

    eigenvalues, eigenvectors = sla.eigs(M.T, k=1, which='LM',
                                         v0=np.random.uniform(0, 1, n_of_teams))

    # assert that all entries in the eigenvector have the same sign
    assert np.prod(np.sign(eigenvectors)).real > 0

    w_inf = np.squeeze((eigenvectors / np.sum(eigenvectors, axis=0)).real)
    print_team_ranking(w_inf, team_dict, f'rankings_w_infinity')

    distances = []
    w_t = np.ones((1, n_of_teams)) / n_of_teams
    for t in range(1, 10001):
        w_t = w_t @ M
        distances.append((t, np.linalg.norm(w_inf - np.squeeze(w_t), ord=1)))

    print(f'The last l1 norm is {distances[-1][-1]}')
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

    write_top_words_to_files(normalized_W, vocab)


def write_top_words_to_files(normalized_W, vocab):
    """
    Write the raw data to a csv file and also write to a latex table

    :param normalized_W:
    :param vocab:
    :return:
    """
    topics = []
    for i in range(np.shape(normalized_W)[1]):
        topics.extend([(i + 1, vocab[j], float("{:.4f}".format(normalized_W[j, i]))) for j in
                       np.argsort(normalized_W[:, i])[::-1][:10]])
    topics_pd = pd.DataFrame(topics, columns=['topic', 'word', 'weight'])
    topics_pd['rank'] = topics_pd.groupby("topic")["weight"].rank('first', ascending=False)
    topics_pd['rank'] = topics_pd['rank'].astype(int)
    # Write the raw data to a CSV file
    topics_pd.to_csv(path.join(INPUT_FOLDER, 'words_top_25_raw'), index=False)

    # Combine the word and its corresponding probability
    group_results = {}
    for i, (name, group) in enumerate(topics_pd.groupby('topic')):
        group_id = i // 5
        if group_id not in group_results:
            group_results[group_id] = []
        group_results[group_id].append(
            (name, group.apply(lambda r: f'{r[1]} {r[2]}', axis=1).to_list()))

    # create each row in the latex table
    latex_table_rows = []
    for i, (group_id, group_result_tuple) in enumerate(group_results.items()):
        if i != 0:
            latex_table_rows.append(' & '.join([''] * 5))
        topics, group_result = zip(*group_result_tuple)
        _, n_words = np.shape(np.asarray(group_result))
        header = ' & '.join([f'\textbf{{Topic {topic}}}' for topic in topics])
        latex_table_rows.append(f'\toprule {header}')
        body = [' & '.join(np.asarray(group_result)[:, i]) for i in range(n_words)]
        body[0] = f'\midrule {body[0]}'
        latex_table_rows.extend(body)

    # Write the table content to a latex table
    with pd.option_context("max_colwidth", 10000):
        with open(path.join(INPUT_FOLDER, 'words_top_25_latex'), 'w') as opened_file:
            opened_file.write(
                pd.DataFrame(latex_table_rows).to_latex(index=False, escape=False).replace(
                    '\hline \\\\', '\hline'))


def main():
    # Set the seed
    np.random.seed(100)

    # Problem 1
    rank_cfb_teams(*load_cfb_data())

    # Problem 2
    run_non_negative_matrix_factorization(*load_nyt_data(), n_topics=25, n_iter=100)


if __name__ == '__main__':
    """
    Author: Chao Pang
    """
    main()
