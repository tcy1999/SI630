# Lint as: python3
"""Example tagging for Toxic Spans based on Spacy.

Baseline provided by https://github.com/ipavlopoulos/toxic_spans/blob/master/baselines/spacy_tagging.py
Note: After this baseline is provided, spacy is upgraded to v3.0, which has quite a few changes. 
Therefore, running this baseline requires older versions of spacy.

Requires:
  pip install spacy==2.3.5
  pip install sklearn

Install models:
  en_core_sm: [2.3.1, 2.3.0]
  pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
"""

import ast
import csv
import pandas as pd
from scipy.stats import sem
import sys
import random
random.seed(2021)

import sklearn
import spacy

from evaluation.semeval2021 import f1
from evaluation import fix_spans

def spans_to_ents(doc, spans, label):
  """Converts span indicies into spacy entity labels."""
  started = False
  left, right, ents = 0, 0, []
  for x in doc:
    if x.pos_ == 'SPACE':
      continue
    if spans.intersection(set(range(x.idx, x.idx + len(x.text)))):
      if not started:
        left, started = x.idx, True
      right = x.idx + len(x.text)
    elif started:
      ents.append((left, right, label))
      started = False
  if started:
    ents.append((left, right, label))
  return ents


def read_datafile(filename):
  """Reads csv file with python span list and text."""
  data = []
  with open(filename) as csvfile:
    reader = csv.DictReader(csvfile)
    count = 0
    for row in reader:
      fixed = fix_spans.fix_spans(
          ast.literal_eval(row['spans']), row['text'])
      data.append((fixed, row['text']))
  return data


def main():
  """Train and eval a spacy named entity tagger for toxic spans."""
  # Read training data
  print('loading training data')
  train = read_datafile('data/tsd_train.csv')

  # Read test data
  print('loading test data')
  test = read_datafile('data/tsd_test.csv')

  # Convert training data to Spacy Entities
  nlp = spacy.load("en_core_web_sm")

  print('preparing training data')
  training_data = []
  for n, (spans, text) in enumerate(train):
    doc = nlp(text)
    ents = spans_to_ents(doc, set(spans), 'TOXIC')
    training_data.append((doc.text, {'entities': ents}))

  toxic_tagging = spacy.blank('en')
  toxic_tagging.vocab.strings.add('TOXIC')
  ner = nlp.create_pipe("ner")
  toxic_tagging.add_pipe(ner, last=True)
  ner.add_label('TOXIC')
  
  pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
  unaffected_pipes = [
      pipe for pipe in toxic_tagging.pipe_names
      if pipe not in pipe_exceptions]

  print('training')
  with toxic_tagging.disable_pipes(*unaffected_pipes):
    toxic_tagging.begin_training()
    for iteration in range(30):
      random.shuffle(training_data)
      losses = {}
      batches = spacy.util.minibatch(
          training_data, size=spacy.util.compounding(
              4.0, 32.0, 1.001))
      for batch in batches:
        texts, annotations = zip(*batch)
        toxic_tagging.update(texts, annotations, drop=0.5, losses=losses)
      print("Losses", losses)

  # Score on test data
  print('evaluation')
  scores = []
  for spans, text in test:
    pred_spans = []
    doc = toxic_tagging(text)
    for ent in doc.ents:
      pred_spans.extend(range(ent.start_char, ent.start_char + len(ent.text)))
    score = f1(pred_spans, spans)
    scores.append(score)
  
  test_f1 = pd.DataFrame({'spacy_f1': scores})
  print (f"Spacy tagging baseline F1 = {test_f1.spacy_f1.mean():.2f} ± {sem(test_f1.spacy_f1):.2f}")
  test_f1.to_csv('spacy_f1.csv')


if __name__ == '__main__':
  main()
