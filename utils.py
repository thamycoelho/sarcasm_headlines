from nltk import download
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer # Lemmatizer of coice
from nltk.stem import SnowballStemmer # Stemmer of choice

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
#from wordcloud import WordCloud
#from textblob import TextBlob
from os.path import join
import pandas as pd
import re

import numpy as np

# Getting what we need from NLTK
download('punkt')
download('stopwords')
download('wordnet')
download('averaged_perceptron_tagger')


###### TESTING #######################################
def testing(headlines, labels, X, Y):
    # Recreate headlines from encoding
    encode_check = True
    for i in range(len(headlines)):
        head = set(headlines[i])
        code = set(labels.classes_[X[i,:]==1])
        if (head-code)!=set():
            encode_check = False
            break

    dim_check = (Y.shape[0]==X.shape[0]) and Y.shape[1]==1 
    dim_check = dim_check and X.shape[1]==len(labels.classes_)

    print("Amount of samples:", Y.shape[0])
    print("Amount of features:", X.shape[1])
    print("Dimensions match?", dim_check)
    print("Properly encoded?", encode_check)

    return dim_check and encode_check
#######################################################

def clean_up(string):
    '''Removes punctuation from a string'''
    pattern="[.,:%-?()&$'\"!“”¯°–―—_\/|#\[\]…@ツ¡©\d]"
    return re.sub(pattern, '', string)

# Removing stopwords: common words that are less useful for detection (example:"the")
# Should be done before stemmeing, since some stop words might not be recognized
# after the stemmezation (doesnt happen if used lemmatization)
def stop_words(tokens):
    '''Removes stop words from a list of tokens preserving their order'''
    stop = set(stopwords.words('english'))
    tok_set = set()
    for t in tokens : tok_set.add(t)
    return list(tok_set-stop)

# Stemming words (brute reduction of words)
def stemmezation(tokens):
    '''Applies stemmeing to a list of tokens'''
    stemmer = SnowballStemmer('english')
    return list(map(stemmer.stem, tokens))

# Counting words (gives a matrix with the count of each word)
def count_words(tokens_list):
    '''Count number words in a list of tokens lists'''
    # Merge all tokens into a single string
    join = lambda words : " ".join(words)
    big_line =  " ".join(list(map(join, tokens_list)))
    counter = CountVectorizer()
    matrix = counter.fit_transform([big_line])
    words = counter.get_feature_names()
    counts = matrix.toarray()
    return words, counts

def onehotencoding(ids, size):
    '''One-hot encodes a list of labels (ids)'''
    hotencoded = np.zeros(shape=(size,))
    for x in ids : hotencoded[x] = 1
    return hotencoded