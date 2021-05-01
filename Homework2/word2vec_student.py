#
# SI630 Homework 2: word2vec
#
# GENERAL NOTES:
#
# We've provided skeleton code for the basic functions of word2vec, which will
# let you get started. The code tasks are
#
#  (1) preprocessing the text input to covert it into a sequence of token IDs
#  (2) writing the core training loop that uses those IDs as training examples
#  (3) writing the gradient descent part that update the weights based on how
#      correct the model's predictions were
#
# Of the three parts, Part 3 is the trickiest to get right and where you're
# likely to spend most of your time. You can do a simple job on Part 1 initially
# to get the model up and running so you can start on Part 2 and then begin
# implementing Part 3
#
# Word2vec itself is a complex set of software and we're only implementing (1)
# negative sampling, (2) rare word deletion, and (3) frequent word subsampling
# in this. You may see more advanced/complex methods described elsewhere.
#
# Note that the model implemented in this homework will take time to fully
# train. However, you can definitely still get intermediate results that confirm
# your model is working by running for fewer steps (or epochs). The easiest way
# to test is to look at the neighest neighbors for common words. We've built
# this in for you already so that after training, the code will print out the
# nearest neighbors of a few works. Usually after ~100K steps, the neighest
# neighbors for at least some of the words will start to make sense.
#
# Other helpful debugging notes:
#
# 1. While you get started, you can turn down the embedding size to print the
# embeddings and see what they look like. Printing and checking is a great way
# to see what's going on and confirm your intuitions.
#
# 2. There are many wrong implementations of word2vec out there. If you go
# looking for inspiration, be aware that some may be wrong.
#
# 3. Familiarizing yourself with numpy will help.
#
# 4. The steps that are listed here are general guides and may correspond to one
# or more lines of code, depending on your programming style. Make sure you
# understand what the word2vec technique is supposed to be doing at that point
# to help guide your development.
#

import csv
import math
import pickle
import random
import sys
from collections import Counter
import numpy as np

# Hhelpful for computing cosine similarity
from scipy.spatial.distance import cosine

# This will make things go fast when we finally use it
from numba import jit

# Handy command-line argument parsing
import argparse

# Progress bar tracker
from tqdm import tqdm

# Sort of smart tokenization
from nltk.tokenize import RegexpTokenizer

# We'll use this to save our models
from gensim.models import KeyedVectors

#
# IMPORTANT NOTE: Always set your random seeds when dealing with stochastic
# algorithms as it lets your bugs be reproducible and (more importantly) it lets
# your results be reproducible by others.
#
random.seed(1234)
np.random.seed(1234)


class word2vec:
    def __init__(self, hidden_layer_size=50):
        self.hidden_layer_size = hidden_layer_size
        self.tokenizer = RegexpTokenizer(r'\w+')
        
        # These state variables become populated as the main() function calls
        #
        # 1. load_data()
        # 2. generate_negative_sampling_table()
        # 3. init_weights()
        #
        # See those functions for how the various values get filled in

        self.word_to_index = {} # word to unique-id
        self.index_to_word = [] # unique-id to word

        # How many times each word occurs in our data after filtering
        self.word_counts = Counter()

        # A utility data structure that lets us quickly sample "negative"
        # instances in a context. This table contains unique-ids
        self.negative_sampling_table = []
        
        # The dataset we'll use for training, as a sequence of unique word
        # ids. This is the sequence across all documents after tokens have been
        # randomly subsampled by the word2vec preprocessing step
        self.full_token_sequence_as_ids = []

        # These will contain the two weight matrices. W is the embeddings for
        # the center/target word and C are the embeddings for the context
        # words. You might see these called (W, V) or (W1, W2) in various
        # documentation too. These get initialized later in init_weights() once
        # we know the vocabulary size
        self.W = None
        self.C = None

        # Store the synonyms
        self.synonyms = {}


    def tokenize(self, text):
        '''
        Tokenize the document and returns a list of the tokens
        '''
        return self.tokenizer.tokenize(text)


    def load_data(self, file_name, min_token_freq):
        '''
        Reads the data from the specified file as long long sequence of text
        (ignoring line breaks) and populates the data structures of this
        word2vec object.
        '''

        # Step 1: Read in the file and create a long sequence of tokens
        with open(file_name, "r", encoding='UTF-8') as f:
            text = f.read()
        tokens = self.tokenize(text)
        tokens = [x.lower() for x in tokens]
            
        # Step 2: Count how many tokens we have of each type
        print('Counting token frequencies')
        token_count = Counter(tokens)

        # Step 3: Replace all tokens below the specified frequency with an <UNK>
        # token
        print("Performing minimum thresholding")
        for i in range(len(tokens)):
            if token_count[tokens[i]] < min_token_freq:
                tokens[i] = '<UNK>'

        # Step 4: update self.word_counts to be the number of times each word
        # occurs (including <UNK>)
        self.word_counts = Counter(tokens)

        # Step 5: Create the mappings from word to unique integer ID and the
        # reverse mapping.
        #
        # HINT: the id-to-word mapping is easily represented as a list data
        # structure
        self.index_to_word = sorted(set(self.word_counts.elements()))
        for i in range(len(self.index_to_word)):
            self.word_to_index[self.index_to_word[i]] = i

        # Step 6: Compute the probability of keeping any particular token of a
        # word in the training sequence, which we'll use to subsample. This
        # avoids having the training data be filled with many overly common words
        total = sum(self.word_counts.values())
        p_subsample = {}
        for word in self.index_to_word:
            p = self.word_counts[word] / total
            p_subsample[word] = (np.sqrt(p/0.001)+1) * 0.001/p

        # Step 7: process the list of tokens (after min-freq filtering) to fill
        # a new list self.full_token_sequence_as_ids where (1) we
        # probabilistically choose whether to keep each token based on the
        # subsampling probabilities and (2) all tokens are converted to their
        # unique ids for faster training.
        for token in tokens:
            if p_subsample[token] < np.random.random():
                continue
            self.full_token_sequence_as_ids.append(self.word_to_index[token])

        # self.negative_sampling_table = generate_negative_sampling_table()
        print('Loaded all data from %s; saw %d tokens (%d unique)' \
              % (file_name, len(self.full_token_sequence_as_ids),
                 len(self.word_to_index)))


    def load_synonym(self, synonym_file):
        with open(synonym_file, "r", encoding='UTF-8') as f:
            synonyms = f.read().splitlines()

        # As I'm using a dict, but a word might have multiple meanings,
        # only the synonyms of one of the meanings will be kept
        for line in synonyms:
            syn_list = line.split(' ')
            syn_list_index = []
            for word in syn_list:
                if word in self.index_to_word:
                    syn_list_index.append(self.word_to_index[word])
            for index in syn_list_index:
                self.synonyms[index] = syn_list_index


    def generate_negative_sampling_table(self, exp_power=0.75, table_size=1e6):
        '''
        Generates a big list data structure that we can quickly randomly index into
        in order to select a negative training example (i.e., a word that was
        *not* present in the context). 
        '''       
        
        # Step 1: Figure out how many instances of each word need to go into the
        # negative sampling table. 
        #
        # HINT: np.power and np.fill might be useful here        
        print("Generating sampling table")
        powers = np.full(len(self.word_to_index), exp_power, dtype=float)
        counts = []
        for word in self.index_to_word:
            counts.append(self.word_counts[word])
        probs = np.power(counts, powers)
        probs /= sum(probs)

        # Step 2: Create the table to the correct size. You'll want this to be a
        # numpy array of type int

        # Step 3: Fill the table so that each word has a number of IDs
        # proportionate to its probability of being sampled.
        #
        # Example: if we have 3 words "a" "b" and "c" with probabilites 0.5,
        # 0.33, 0.16 and a table size of 6 then our table would look like this
        # (before converting the words to IDs):
        #
        # [ "a", "a", "a", "b", "b", "c" ]
        #
        self.negative_sampling_table = np.random.choice(len(self.index_to_word), int(table_size), p=probs)


    def generate_negative_samples(self, cur_context_word_id, num_samples):
        '''
        Randomly samples the specified number of negative samples from the lookup
        table and returns this list of IDs as a numpy array. As a performance
        improvement, avoid sampling a negative example that has the same ID as
        the current positive context word.
        '''

        # Step 1: Create a list and sample from the negative_sampling_table to
        # grow the list to num_samples, avoiding adding a negative example that
        # has the same ID as the current context_word
        results = []
        while len(results) < num_samples:
            sample = np.random.choice(self.negative_sampling_table)
            if sample != cur_context_word_id:
                results.append(sample)

        # Step 2: Convert the list of samples to numpy array and return it            
        return np.array(results)


    def save(self, filename):
        '''
        Saves the model to the specified filename as a gensim KeyedVectors in the
        text format so you can load it separately.
        '''

        # Creates an empty KeyedVectors with our embedding size
        kv = KeyedVectors(vector_size=self.hidden_layer_size)        
        vectors = []
        words = []
        # Get the list of words/vectors in a consistent order
        for index, word in enumerate(self.index_to_word): 
            vectors.append(self.W[index].copy())
            words.append(word)
            
        # Fills the KV object with our data in the right order
        kv.add(words, vectors) 
        kv.save_word2vec_format(filename, binary=False)


    def init_weights(self, init_range=0.1):
        '''
        Initializes the weight matrices W (input->hidden) and C (hidden->output)
        by sampling uniformly within a small range around zero.
        '''

        # Step 1: Initialize two numpy arrays (matrices) for W and C by filling
        # their values with a random sample within the speified range.
        #
        # Hint: numpy.random has lots of ways to create matrices for this task
        self.W = np.random.uniform(-init_range, init_range, [len(self.word_to_index), self.hidden_layer_size])
        self.C = np.random.uniform(-init_range, init_range, [len(self.word_to_index), self.hidden_layer_size])


    def train(self, num_epochs=2, window_size=2, num_negative_samples=2,
              learning_rate=0.05, nll_update_iter=10000, max_steps=-1):
        '''
        Trains the word2vec model on the data loaded from load_data for the
        specified number of epochs.
        '''

        # Rather than compute the full negative log-likelihood (NLL), we'll keep
        # a running tally of the nll values for each step and periodically report them
        nll_results = []
        
        # This value keeps track of which step we're on. Since we don't update
        # when the center token is "<UNK>", we may skip over some ids in the
        # inner loop, so we need a separate step count to keep track of how many
        # updates we've done.
        step = 0

        # Iterate for the specified number of epochs
        for epoch in range(1, num_epochs+1):
            print("Beginning epoch %d of %d" % (epoch, num_epochs))
            # Step 1: Iterate over each ID in full_token_sequence_as_ids as a center
            # token (skipping those that are <UNK>) and predicting the context
            # word and negative samples
            #
            # Hint: this is a great loop to wrap with a tqdm() call so you can
            # see how long each epoch will take with a progress bar
            for center_index in range(0, len(self.full_token_sequence_as_ids)):
                center_id = self.full_token_sequence_as_ids[center_index]
                if center_id == self.word_to_index['<UNK>']:
                    continue

                # Tokens at the beginning and the last have fewer contexts
                if center_index < window_size:
                    window = list(range(-center_index, 0)) + list(range(1, window_size+1))
                elif center_index >= len(self.full_token_sequence_as_ids)-window_size:
                    window = list(range(-window_size, 0)) + list(range(1, len(self.full_token_sequence_as_ids)-center_index))
                else:
                    window = list(range(-window_size, 0)) + list(range(1, window_size + 1))

                # Synonyms
                if center_id in self.synonyms:
                    center_id = np.random.choice(self.synonyms[center_id])

                # Periodically print the NLL so we can see how the model is converging
                if nll_update_iter > 0 and step % nll_update_iter == 0 and step > 0 and len(nll_results) > 0:
                    print("Negative log-likelihood (step: %d): %f " % (step, sum(nll_results)))
                    nll_results = []
                                    
                # Step 2: For each word in the window range (before and after)
                # perform an update where we (1) use the current parameters of
                # the model to predict it using the skip-gram task and (2)
                # sample negative instances and predict those. We'll use the
                # values of those predictions (i.e., the output of the sigmoid)
                # to update the W and C matrices using backpropagation.
                #
                # NOTE: this inner loop should call predict_and_backprop() which is
                # defined outside of the class. See note there for why.
                for i in window:
                    # Step 3: Pick the context word ID
                    context_id = self.full_token_sequence_as_ids[center_index + i]
                    # Step 4: Sample negative instances
                    negative_samples = self.generate_negative_samples(context_id, num_negative_samples)
                    # Step 5: call predict_and_backprop. Don't forget to add the
                    # nll return value to nll_results to keep track of how the
                    # model is learning
                    nll = predict_and_backprop(self.W, self.C, learning_rate, center_id, context_id,
                                               negative_samples)
                    nll_results.append(nll)

                step += 1
                if max_steps > 0 and step >= max_steps:
                    break

            if max_steps > 0 and step >= max_steps:
                print('Maximum number of steps reached: %d' % max_steps)
                break


    def get_neighbors(self, target_word):
        """ 
        Finds the top 10 most similar words to a target word
        """
        outputs = []
        for index, word in tqdm(enumerate(self.index_to_word), total=len(self.index_to_word)):
            similarity = self.compute_cosine_similarity(target_word, word)
            result = {"word": word, "score": similarity}
            outputs.append(result)
    
        # Sort by highest scores
        neighbors = sorted(outputs, key=lambda o: o['score'], reverse=True)
        return neighbors[1:11]


    def compute_cosine_similarity(self, word_one, word_two):
        '''
        Computes the cosine similarity between the two words
        '''
        try:
            word_one_index = self.word_to_index[word_one]
            word_two_index = self.word_to_index[word_two]
        except KeyError:
            return 0
    
        embedding_one = self.W[word_one_index]
        embedding_two = self.W[word_two_index]
        similarity = 1 - abs(float(cosine(embedding_one, embedding_two)))
        return similarity


#
# IMPORTANT NOTE:
# 
# These functions are specified *outside* of the word2vec class so that they can
# be compiled down into very effecient C code by the Numba library. Normally,
# we'd put them in the word2vec class itself but Numba doesn't know what to do
# with the self parameter, so we've pulled them out as separate functions which
# are easily compiled down.
#
# When you've gotten your implementation fully correct, uncomment the line above
# the function that reads:
#
#   @jit(nopython=True)
#
# Which will turn on the just-in-time (jit) Numba compiler for making this
# function very fast. From what we've seen, this makes our implemetnation around
# 300% faster which is a huge and easy speed up to run on the dataset.
#
# The gradient descent part requires the most "math" for your program and
# represents the hottest part of the code (since this is called multiple times
# for each context!). Speeding up this one piece can result in huge performance
# gains, so if you're feeling adventurous, try copying the code and then
# modifying it to see how you can make it faster. On a 2.7GHz i7 processor, tdqm
# reports about ~10k iterations/sec in our reference implementation.
#

@jit(nopython=True)
def predict_and_backprop(W, C, learning_rate, center_id, context_id,
                         negative_sample_ids):
    '''
    Using the center token (specified by center_id), makes a forward pass through
    the network to predict the context token (context_id) and negative samples,
    then backprops the error of those predictions to update the network and
    returns the negative log likelihood (Equation 1 in your homework) from the
    current predictions. W and C are the weight matrices of the network and IDs
    refer to particular rows of the matrices (i.e., the word embeddings of the
    target word and the context words!)

    '''

    #
    # GENERAL NOTE: There are many ways to implement this function, depending on
    # how fancy you want to get with numpy. The instructions/steps here are
    # intended as guides for the main tasks on what you have to do and may be
    # implemented as one line or more lines, depending on which methods you use
    # and how you want to write it. The important thing is that it works, not
    # how fast it is, so feel free to write it in a way that is understandable
    # to you. Often when you get to that point, you'll see ways to improve (but
    # first save a copy of your working code!).
    #

    # Step 1: Look up the two vectors in W and C. Note that the row for the
    # center_id is effectively the hidden layer activation, h.
    h = W[center_id].copy()
    v_c = C[context_id].copy()

    # Step 2: Look up the vectors for the negative sample IDs.
    #
    # NOTE: numpy supports multiple indexing (getting multiple rows at once) if
    # you want to use it
    v_negs = C[negative_sample_ids].copy()

    # Step 3: Compute the predictions for the context word and the negative
    # examples. We want the predictions of the context word to be near 1 and
    # those for the negative examples to be near 0.
    pred_c = sigmoid(v_c.dot(h))
    pred_neg = []
    for i in range(len(negative_sample_ids)):
        pred_neg.append(sigmoid(v_negs[i].dot(h)))

    # Step 4: Compute the negative log likelihood
    nll = -np.log(pred_c)
    # seems that np.log for list is not supported here? unfamiliar with np in numba
    for i in range(len(negative_sample_ids)):
        nll -= np.log(sigmoid(-v_negs[i].dot(h)))

    # Step 5: Update the negative sample vectors to push their dot product with the
    # center word's vecter closer to zero.
    C[context_id] -= learning_rate * (pred_c-1) * h
    for i in range(len(negative_sample_ids)):
        C[negative_sample_ids[i]] -= learning_rate * pred_neg[i] * h
        
    # Step 6: Now backprop all the way back to the center word's vector. Be sure to
    # update it based on the *old* values of the context vectors, not the
    # new values of the context vectors that you just updated!
    total = (pred_c-1)*v_c
    for i in range(len(negative_sample_ids)):
        total += pred_neg[i]*v_negs[i]
    W[center_id] -= learning_rate * total
    return nll

    
@jit(nopython=True)
def sigmoid(x):
    '''
    Returns the sigmoid of the provided value
    '''
    return 1.0 / (1 + np.exp(-x))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--corpus_file', type=str, 
                        help="The file name for the text file to use in training")

    parser.add_argument('--synonym_file', type=str,
                        help="The file name for the synonyms text file")
    
    parser.add_argument('--window_size', type=int, default=2,
                        help="The number of tokens before or after the center token to include as context when training")

    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help="The learning rate to use when updating embeddings during SGD")

    parser.add_argument('--embedding_size', type=int, default=50,
                        help="The embedding dimension side (i.e., hidden layer size)")    

    parser.add_argument('--min_token_frequency', type=int, default=5,
                        help="The minimum number of times a token must occur to be considered in the vocabulary")

    parser.add_argument('--num_epochs', type=int, default=2,
                        help="The number of epochs of training to complete, where an epoch is a full pass through the entire data.")    
    
    parser.add_argument('--max_steps', type=int, default=-1,
                        help="The maximum number of steps to take. Positive values will cause the training to end early if the maximum number is reached before all epochs have finished.")    

    parser.add_argument('--nll_update_iter', type=int, default=100000,
                        help="How many steps to take between printing the negative log-likelihood when testing for convergence. Negative values cause the NLL to not be printed")

    parser.add_argument('--do_quick_nn_test', type=bool, default=True,
                        help="Run a quick nearest-neighbor test")
    
    parser.add_argument('--output_file', type=str, 
                        help="Where to save the word vectors as a gensim.KeyedVectors")

    args = parser.parse_args()

    if not args.corpus_file:
        print('No file specified for training! See help message:\n')
        parser.print_help(sys.stderr)
        exit(1)

    if not args.output_file:
        print('REMINDER: this run is not saving any output. If you meant to, restart and add --output_file')

    corpus_file_name = args.corpus_file
    model = word2vec()

    model.load_data(corpus_file_name, args.min_token_frequency)
    model.generate_negative_sampling_table()
    model.init_weights()

    # If synonym_file is provided, preprocess the file so that the synonyms could be used in train
    if args.synonym_file:
        model.load_synonym(args.synonym_file)

    # Could pass other args here if needed
    model.train(nll_update_iter=args.nll_update_iter, max_steps=args.max_steps,
                num_epochs=args.num_epochs)

    if args.output_file:
        print("Saving model")
        model.save(args.output_file)
    
    if args.do_quick_nn_test:
        targets = ["january", "good", "the", "food", "engineering"]

        for targ in targets:
            print("Target: ", targ)
            bestpreds = (model.get_neighbors(targ))
            for pred in bestpreds:
                print(pred["word"], ":", pred["score"])
            print("\n")

