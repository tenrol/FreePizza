#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:31:21 2018

"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk import pos_tag
from string import punctuation



DATASET_FILE = 'data.json'
STOPWORDS = 'stopwords.json'

lmt = WordNetLemmatizer()
nb = MultinomialNB()
lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

# Opening Dataset (data.json)
with open(DATASET_FILE) as fin: 
    data = json.load(fin)

# Opening stopwords.json file
with open(STOPWORDS) as fin: 
    stopwords01 = json.load(fin)

# Creating Stopwords set
stopwords01_ = set(stopwords01['en'])
stopwords02_ = set(stopwords.words('english'))
stopwords03_ = set(punctuation)
stoplist_combined = set.union(stopwords01_, stopwords02_, stopwords03_)

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    # Input: str - peen tag
    # Output: str - wordNet
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' 


def lemmatize(text): 
    """ Finding the root words """
    # Input: str - text
    # Output: (lowercase) str - text.
    return [lmt.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(word_tokenize(text))]

def preprocess(text):
    """ Removing stopwords and digits """
    # Input: str - document/sentence
    # Output: list(str) - list of lemmas
    return [word for word in lemmatize(text) 
            if word not in stoplist_combined
            and not word.isdigit()]

# Creating Dataframe from json formated data
df = pd.io.json.json_normalize(data)

# Keeping only relevant information (based on the assumptions made)
df_train = df[['request_id', 'request_title', 
               'request_text_edit_aware', 
               'requester_received_pizza']]

# Spliting data into two subsets : training dataset and test dataset
train, test = train_test_split(df_train, test_size=0.3)

# Vectorizing Datasets
count_vect = CountVectorizer(analyzer=preprocess)
train_set = count_vect.fit_transform(train['request_text_edit_aware'])
train_tags = train['requester_received_pizza']
test_set = count_vect.transform(test['request_text_edit_aware'])
test_tags = test['requester_received_pizza']

# Training NB and LR model
nb_clf = nb.fit(train_set, train_tags) 
lr_clf = lr.fit(train_set, train_tags)

# Testing the trained model
predictions_test_nb = nb_clf.predict(test_set)
predictions_test_lr = lr_clf.predict(test_set)

# Printing accuracy
print('\n-- Accuracy --')
print('MultinomialNB = {}'.format(
        accuracy_score(predictions_test_nb, test_tags) * 100)
     )
print('LogisticRegression = {}'.format(
        accuracy_score(predictions_test_lr, test_tags) * 100)
     )

print('\n-- Cross Validation Accuracy --')
scores_nb = cross_val_score(nb, train_set, train_tags, cv=3)
print('MultinomialNB = ', scores_nb)
scores_lr = cross_val_score(lr, train_set, train_tags, cv=3)
print('LogisticRegression = ', scores_lr)


# Ploting ROC: Receiver Operating Characteristic
print('\n-- MultinomialNB ROC --')
fpr_nb, tpr_nb, thresh_nb = metrics.roc_curve(test_tags, nb_clf.predict_proba(test_set)[:, 1])
plt.plot(fpr_nb, tpr_nb)
plt.show()
print('\n-- LogisticRegression ROC --')
fpr_lr, tpr_lr, thresh_lr = metrics.roc_curve(test_tags, lr_clf.predict_proba(test_set)[:, 1])
plt.plot(fpr_lr, tpr_lr)
plt.show()