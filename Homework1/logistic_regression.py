import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from naive_bayes import better_tokenize


def sigmoid(dot_prod):
    return 1 / (1+np.exp(-dot_prod))


def log_likelihood(X, y, b):
    return np.sum(y*(b.dot(X.T)) - np.log(1+np.exp(b.dot(X.T))))


def compute_gradient(x_i, y_i_true, dot_prod):
    return (y_i_true-sigmoid(dot_prod)) * x_i


def logistic_regression(X, y, learning_rate, num_steps=None):
    b = np.zeros(X.shape[1])
    step_count = 0
    log_likelihoods = []
    not_converge = True
    while not_converge:
        for i in range(len(X)):
            dot_prod = b.dot(X[i])
            b += learning_rate*compute_gradient(X[i], y[i], dot_prod)
            step_count += 1
            if (num_steps is not None) and (step_count > num_steps):
                not_converge = False
                break
            if step_count % 100 == 0:
                # every 100 steps
                log_likelihood_this = log_likelihood(X, y, b)
                if (len(log_likelihoods) > 0) and (abs(log_likelihood_this-log_likelihoods[len(log_likelihoods)-1] < 10**(-5))):
                    not_converge = False
                    break
                log_likelihoods.append(log_likelihood_this)
    ll_df = pd.DataFrame({'step': np.arange(len(log_likelihoods)), 'log_likelihood': log_likelihoods})
    return b, ll_df


def predict(b, x):
    if sigmoid(b.dot(x)) > 0.5:
        return 1.0
    return 0.0


def extract_feature(training):
    vocabulary = set()
    for each in training:
        tokens = better_tokenize(each)
        for token in tokens:
            if token not in vocabulary:
                vocabulary.add(token)
    return list(vocabulary)


def preprocess(vocabulary, comments):
    X = np.zeros((len(comments), len(vocabulary)))
    for i in range(len(comments)):
        tokens = better_tokenize(comments[i])
        for token in tokens:
            if token in vocabulary:
                X[i][vocabulary.index(token)] += 1
    return np.hstack([np.ones((len(X), 1)), X])


def main():
    sns.set_theme(style="darkgrid")
    train_df = pd.read_csv('train.csv')
    vocabulary = extract_feature(train_df['comment_text'])
    train_X = preprocess(vocabulary, train_df['comment_text'])
    train_y = train_df['is_constructive']

    b1, ll_normal = logistic_regression(train_X, train_y, 5*10**(-5), 1000)
    b2, ll_large = logistic_regression(train_X, train_y, 5*10**(-3), 1000)
    b3, ll_small = logistic_regression(train_X, train_y, 5*10**(-7), 1000)
    fig, axes = plt.subplots(ncols=3)
    sns.scatterplot(x='step', y='log_likelihood', data=ll_normal, ax=axes[0])
    sns.scatterplot(x='step', y='log_likelihood', data=ll_large, ax=axes[1])
    sns.scatterplot(x='step', y='log_likelihood', data=ll_small, ax=axes[2])
    fig.savefig('plot_2.png', dpi=400)
    plt.close()

    test_df = pd.read_csv('test.csv')
    test_X = preprocess(vocabulary, test_df['comment_text'])
    b, ll_df = logistic_regression(train_X, train_y, 5*10**(-3))
    prediction = [predict(b, test_X[i]) for i in range(len(test_X))]
    pred_df = pd.DataFrame(test_df['commend_id'])
    pred_df['is_constructive'] = prediction
    pred_df.to_csv('result2.csv', index=False)


if __name__ == '__main__':
    main()
