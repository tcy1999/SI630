import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.metrics import f1_score


def setup():
    sns.set_theme(style="darkgrid")
    train_df = pd.read_csv('train.csv')
    train2_df, dev_df = np.split(train_df.sample(frac=1, random_state=1234), [int(.8*len(train_df)),])
    return train_df, train2_df, dev_df


def tokenize(string):
    return string.split()


def better_tokenize(string):
    # stop words?
    str_split = string.split()
    tokens = []
    for each in str_split:
        if re.search(r'(http[s]?://.*)|(.*www\..*)', each):
            continue
        each = re.sub(r'(er)|(est)|([^\w])$', '', each)
        each = re.sub(r'^[^\w]', '', each)
        tokens.append(each.lower())
    return tokens


def train(training, func_token, alpha=0):
    if alpha < 0:
        print('Please use non-negative smoothing alpha')
        exit()
    count_by_class = {}  # count of word i in two classes, [n_{i, 0}, n_{i, 1}]
    word_count = [0, 0]  # number of words in each class
    likelihood = {}
    prior = training.is_constructive.value_counts() / training.is_constructive.size
    for i, row in training.iterrows():
        tokens = func_token(row['comment_text'])
        is_constructive = int(row['is_constructive'])
        for token in tokens:
            if token not in count_by_class:
                count_by_class[token] = [0, 0]
            count_by_class[token][is_constructive] += 1
            word_count[is_constructive] += 1
    vocabulary_size = len(count_by_class)
    for word in count_by_class:
        likelihood[word] = [(count_by_class[word][0]+alpha) / (word_count[0]+vocabulary_size*alpha),
                            (count_by_class[word][1]+alpha) / (word_count[1]+vocabulary_size*alpha)]
    likelihood['For none'] = [alpha / (word_count[0]+vocabulary_size*alpha),
                              alpha / (word_count[1]+vocabulary_size*alpha)]
    return prior, likelihood


def classify(prior, likelihood, tokens):
    prob_bad = np.log(prior[0])
    prob_good = np.log(prior[1])
    for token in tokens:
        if token in likelihood:
            prob_bad += np.log(likelihood[token][0])
            prob_good += np.log(likelihood[token][1])
        else:
            prob_bad += np.log(likelihood['For none'][0])
            prob_good += np.log(likelihood['For none'][1])
    if prob_good > prob_bad:
        return 1.0
    return 0.0


def main(func_token, plot_name, result_file):
    train_df, train2_df, dev_df = setup()
    default_prior, default_likelihood = train(train2_df, func_token)
    dev_df['pred'] = [classify(default_prior, default_likelihood, func_token(comment)) for comment in dev_df['comment_text']]
    print('Performance with no smoothing: {}.'
          .format(f1_score(dev_df['is_constructive'], dev_df['pred'])))

    data_plot = pd.DataFrame({'alpha': [], 'score': []})
    for power in np.arange(-4, 2.1, 0.1):
        train_prior, train_likelihood = train(train2_df, func_token, 10**power)
        dev_df['pred'] = [classify(train_prior, train_likelihood, func_token(comment)) for comment in dev_df['comment_text']]
        new_row = pd.DataFrame({'alpha': 10**power,
                                'score': f1_score(dev_df['is_constructive'], dev_df['pred'])}, index=[0])
        data_plot = data_plot.append(new_row, ignore_index=True)
    fig = sns.scatterplot(x='alpha', y='score', data=data_plot).get_figure()
    fig.savefig(plot_name, dpi=400)
    plt.close()

    best_alpha = data_plot.iloc[np.argmax(data_plot.score, axis=0)]['alpha']
    print('Best alpha: {}.'.format(best_alpha))
    best_prior, best_likelihood = train(train_df, func_token, best_alpha)
    test_df = pd.read_csv('test.csv')
    prediction = [classify(best_prior, best_likelihood, func_token(comment)) for comment in test_df['comment_text']]
    pred_df = pd.DataFrame(test_df['commend_id'])
    pred_df['is_constructive'] = prediction
    pred_df.to_csv(result_file, index=False)


main(tokenize, 'plot_1_1.png', 'result_1_1.csv')
main(better_tokenize, 'plot_1_2.png', 'result_1_2.csv')
