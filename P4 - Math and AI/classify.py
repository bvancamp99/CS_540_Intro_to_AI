# Author: Bryce Van Camp
# Project: P4
# Files: envelope_sim.py, classify.py
# File: classify.py

import os
import pathlib
import math

# Create and return a vocabulary as a list of word types with counts >= cutoff
# in the training directory.
#
# training_directory - name of directory to traverse
# cutoff - minimum count for a specific word type
def create_vocabulary(training_directory, cutoff):
    word_types = {}
    
    # retrieve word types
    for dirpath, dirnames, filenames in os.walk(training_directory):
        for filename in filenames:
            with open(dirpath + '/' + filename, 'r', encoding='utf8') as f:
                for line in f:
                    # remove newline character
                    line = line.rstrip('\n')
                    
                    if line in word_types:
                        word_types[line] += 1
                    else:
                        word_types[line] = 1
    
    # add all word_types with count >= cutoff to vocab
    vocab = []
    for key in word_types:
        if word_types[key] >= cutoff:
            vocab.append(key)
    
    return sorted(vocab)


# Create and return a bag of words Python dictionary from a single document.
#
# vocab - list of all word types
# filepath - directory of a specific file
def create_bow(vocab, filepath):
    vocab_dict = {}
    for word in vocab:
        vocab_dict[word] = None
    
    # get bag of words
    bow = {}
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            # remove newline character
            line = line.rstrip('\n')
            
            if line in vocab_dict:
                if line in bow:
                    bow[line] += 1
                else:
                    bow[line] = 1
            else:
                if None in bow:
                    bow[None] += 1
                else:
                    bow[None] = 1
                    
    return bow


# Create and return training set (bag of words Python dictionary + label) 
# from the files in a training directory.
#
# vocab - list of all word types
# directory - name of directory to traverse
def load_training_data(vocab, directory):
    training_data = []
    
    # retrieve training data
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filedir = dirpath + '/' + filename
            path = pathlib.Path(filedir)
            cur_data = {'label': path.parent.name, 'bow': create_bow(vocab, filedir)}
            
            training_data.append(cur_data)
    
    return training_data
    

# Given a training set, estimate and return the prior probability p(label)
# of each label.
#
# training_data - the training set
# label_list - list of label names
def prior(training_data, label_list):
    probs = {}
    for cur_label in label_list:
        probs[cur_label] = 0
    
    # retrieve number of documents with each label
    for cur_data in training_data:
        if cur_data['label'] in probs:
            probs[cur_data['label']] += 1
    
    # log probability of each label
    for cur_label in probs:
        probs[cur_label] = math.log(probs[cur_label] / len(training_data))
    
    return probs


# Given a training set and a vocabulary, estimate and return the class 
# conditional distribution P(word | label) over all words for the given
# label using smoothing.
#
# vocab - list of all word types
# training_data - the training set
# label - given label
def p_word_given_label(vocab, training_data, label):
    # set up return val
    probs = {}
    for x in vocab:
        probs[x] = 0
    probs[None] = 0
    
    # increment occurrences of word types with given label
    num_tokens = 0
    for cur_data in training_data:
        if cur_data['label'] == label:
            for x in cur_data['bow']:
                probs[x] += cur_data['bow'][x]
                num_tokens += cur_data['bow'][x]
    
    num_types = len(probs)
    
    # get probability for each word type x
    for x in probs:
        # x_prob = ((num occurrences of x) + 1) / ((num_tokens) + (num_types))
        probs[x] = math.log((probs[x] + 1) / (num_tokens + num_types))
    
    return probs


# Loads the training data, estimates the prior distribution P(label) and class
# conditional distributions P(word | label), and return the trained model.
#
# training_directory - name of directory to traverse
# cutoff - minimum count for a specific word type
def train(training_directory, cutoff):
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    log_prior = prior(training_data, ['2020', '2016'])
    p_given_2020 = p_word_given_label(vocab, training_data, '2020')
    p_given_2016 = p_word_given_label(vocab, training_data, '2016')
    
    return { 'vocabulary': vocab, 'log prior': log_prior, 'log p(w|y=2016)': p_given_2016, 'log p(w|y=2020)': p_given_2020 }


# Given a trained model, predict the label for the test document.
#
# model - trained model
# filepath - filepath of the test document
def classify(model, filepath):
    bow = create_bow(model['vocabulary'], filepath)
    
    # log property: log(x) + log(y) = log(xy)
    
    # 2020
    p_2020 = model['log prior']['2020']
    for x in bow:
        if x in model['log p(w|y=2020)']:
            p_2020 += model['log p(w|y=2020)'][x] * bow[x]
    
    # 2016
    p_2016 = model['log prior']['2016']
    for x in bow:
        if x in model['log p(w|y=2016)']:
            p_2016 += model['log p(w|y=2016)'][x] * bow[x]
            
    return { 'log p(y=2020|x)': p_2020, 'log p(y=2016|x)': p_2016, 'predicted y': '2020' if max(p_2020, p_2016) is p_2020 else '2016' }