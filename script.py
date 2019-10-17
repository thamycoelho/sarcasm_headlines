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
    '''Removes stop words from a list of tokens'''
    stop = set(stopwords.words('english'))
    return list(set(tokens)-stop)

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

# Reading dataset
folder = 'Dataset'
dataset = pd.read_json(join(folder, 'Sarcasm_Headlines_Dataset_v2.json'), lines=True)

# Tokenize headlines (nltk tokenizer is more robust with punctuation)
dataset.headline = dataset.headline.apply(clean_up)
dataset.headline = dataset.headline.apply(word_tokenize)
dataset.headline = dataset.headline.apply(stop_words)
dataset.headline = dataset.headline.apply(stemmezation)

words, counts = count_words(dataset.headline)
# Create a list of insignificant words (Words with low frequency)
MIN_FREQ = 3
discard = []
for (w,c) in zip(words, counts[0]):
    if c < MIN_FREQ : discard.append(w)

# Create a set with the selected words by the countvectorizer
# and the discarded ones due to low frequency, and removes any
# words from the headlines which dont belong to the such set
select = set(words)-set(discard)
dataset.headline = dataset.headline.apply(lambda x : list(set(x) & select))

# One Hot Encoding 
# First we need to label the words
labels = LabelEncoder().fit(list(select))
encoded = dataset.headline.apply(labels.transform) # Transforming words to labels
encoded = encoded.apply(lambda x : onehotencoding(x, len(select)))

#### VARIBLES ####
X = np.array(encoded.tolist())
Y = np.array([[x] for x in dataset.is_sarcastic.tolist()]) 
testing(dataset.headline, labels, X, Y)

# Splitting dataset.
X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.1)











##### SOME TESTED STUFF WHICH WAS DISCARDED #####

#### COUNT VECTORIZER ALREADY DOES THE CLEANUP ####
# dataset.headline = dataset.headline.apply(clean_up)

#### DIFFERENT METHOD FOR REMOVING STOPWORDS ####
# # Removing stopwords: common words that are less useful for detection (example:"the")
# stop = set(stopwords.words('english'))
# filt = dataset.headline.apply(lambda row: list(filter(lambda w: w not in stop, row)))
# dataset.headline = filt

#### USING PART-OF-SPEECH TAGGING ####s
# # POS tagging (somewhat limited by the flexibility of words)
# tagged_data = pos_tag(words)

#### LEMMATIZATION INSTEAD OF STEMMING ####
# # Lemmatizing words (preserves more meaning)
# wnlt = WordNetLemmatizer()
# lemmatization = lambda phrase : [wnlt.lemmatize(w) for w in phrase]
# dataset.headline = dataset.headline.apply(lemmatization)

#### CHECKING DATASET DISTRIBUTION ####
# # Perfectly balanced, as all things should be...
# sarcastic = dataset[dataset['is_sarcastic']==1]
# legit = dataset[dataset['is_sarcastic']==0]
# print(f'Sarcastic: {len(sarcastic)*100/(len(sarcastic)+len(legit))}%')
# print(f'Legitimate: {len(legit)*100/(len(sarcastic)+len(legit))}%')

#### TESTING SUBJECTIVITY ####
# # Subjectivity in sarcasm (not very usefull)
# sarcastic = dataset[dataset['is_sarcastic']==1]
# legit = dataset[dataset['is_sarcastic']==0]
# subjectivity = lambda x: TextBlob(x).sentiment.subjectivity
# sum(sarcastic.headline.apply(subjectivity))/len(sarcastic)
# sum(legit.headline.apply(subjectivity))/len(legit)

#### WORD CLOUDS ####
# # Sarcastic word clouds
# wordcloud = WordCloud(max_words=50, background_color="white").generate(big_line)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()
# # Legitimate word cloud
# wordcloud = WordCloud(max_words=50, background_color="white").generate(big_line)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()
