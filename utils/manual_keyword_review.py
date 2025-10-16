import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

import string

import numpy as np

from utils.data_loading import ReviewLoader
from utils.preprocessing import ReviewProcessor

loader = ReviewLoader()
train_reviews, train_labels = loader.load_train_reviews()

for i in range(len(train_labels)):
    if 'chicago' in train_reviews[i].lower().split():
        sentences = train_reviews[i].replace('!','.').replace('?','.').split('.')
        for sentence in sentences:
            if ('chicago' in sentence.lower().split()) and ('hilton' in sentence.lower().split()):
                print("-"*50)
                if train_labels[i]:
                    print('TRUTHFUL REVIEW')
                else:
                    print('DECEPTIVE REVIEW')
                print("-"*50)
                input(sentence)
                print('\n\n')
                break