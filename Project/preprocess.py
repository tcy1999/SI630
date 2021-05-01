import pandas as pd
import re
from ast import literal_eval
from toxic_spans.evaluation.fix_spans import fix_spans
from string import punctuation
# import spacy
# nlp = spacy.load("en_core_web_sm")

'''
Failed approach with spacy tokenization
def convert_data(df):
    data = []
    
    for i, row in df.iterrows():
        row.spans = fix_spans(row.spans, row.text)
        doc = nlp(row.text)
        spans = set(row.spans)
        for token in doc:
            label = 'N'
            if spans.intersection(set(range(token.idx, token.idx + len(token.text)))):
                label = 'T'
            data.append([i, token.text, label])

    return data
'''

def offset2word(text, spans):
    gaps = [[s, e] for s, e in zip(spans, spans[1:]) if s+1 < e]
    edges = iter(spans[:1] + sum(gaps, []) + spans[-1:])
    lis = list(zip(edges, edges))
    toxic_words = []
    for each in lis:
        temp = text[each[0]:each[1]+1].split()
        for item in temp:
            toxic_words.append(item)
    return toxic_words

'''
# Replaced with a better punctuation strip
def convert_data(df):
    data = []
    
    for i, row in df.iterrows():
        row.spans = fix_spans(row.spans, row.text)
        row.text = re.sub(r'[^\w\s]+', ' ', row.text)
        words = row.text.split()
        toxic_words = offset2word(row.text, row.spans)
        for word in words:
            label = 'N'
            if word in toxic_words:
                label = 'T'
            data.append([i, word, label])
    return data
'''
def convert_data(df):
    data = []
    
    for i, row in df.iterrows():
        row.spans = fix_spans(row.spans, row.text)
        words = row.text.split()
        toxic_words = offset2word(row.text, row.spans)
        for word in words:
            word = word.strip(punctuation)
            label = 'N'
            if word in toxic_words:
                label = 'T'
            data.append([i, word, label])
    return data


def load_raw():
    train = pd.read_csv("toxic_spans/data/tsd_train.csv") 
    # spans was string and needs clean
    train.spans = train.spans.apply(literal_eval)
    trial = pd.read_csv("toxic_spans/data/tsd_trial.csv") 
    trial.spans = trial.spans.apply(literal_eval)
    test = pd.read_csv("toxic_spans/data/tsd_test.csv") 
    test.spans = test.spans.apply(literal_eval)
    return train, trial, test


def load_data():
    train, trial, test = load_raw()
    train_data = convert_data(train)
    train_df = pd.DataFrame(train_data, columns=["sentence_id", "words", "labels"])
    
    trial_data = convert_data(trial)
    trial_df = pd.DataFrame(trial_data, columns=["sentence_id", "words", "labels"])
    
    text_punc_strip = []
    for i, row in test.iterrows():
        row.spans = fix_spans(row.spans, row.text)
        # row.text = re.sub(r'[^\w\s]+', ' ', row.text)
        words = row.text.split()
        string = ''
        for word in words:
            word = word.strip(punctuation)
            string += word + ' '
        text_punc_strip.append(string)
    test['text_punc_strip'] = text_punc_strip

    return train_df, trial_df, test
