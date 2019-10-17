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
from wordcloud import WordCloud
from textblob import TextBlob
from os.path import join
import pandas as pd
import re

import numpy as np

# Getting what we need from NLTK
download('punkt')
download('stopwords')
download('wordnet')
download('averaged_perceptron_tagger')

def big_line(text):
    join = lambda words : " ".join(words)
    return " ".join(list(map(join, text)))

def clean_up(text, pattern="[.,:%-?()&$'\"!\“\”¯°–―—_\/|#\[\]…@ツ¡\d]"):
    # Cleanning up the data 
    clean_up = lambda txt : re.sub(pattern, '', txt)
    text = text.apply(clean_up)

def onehotencoding(ids, size):
    hotencoded = np.zeros(shape=(size,))
    for x in ids : hotencoded[x] = 1
    return hotencoded

def remove_unlabeled(sentence):
    hit_list = ['of',',', 'is', 'on', 'rep.','9', ':', 'your']
    for x in hit_list:
        if x in sentence : sentence.remove(x)
    return sentence

# Reading dataset
folder = 'Dataset'
dataset = pd.read_json(join(folder, 'Sarcasm_Headlines_Dataset_v2.json'), lines=True)

# Tokenize headlines (nltk tokenizer is more robust with punctuation)
dataset.headline = dataset['headline'].apply(word_tokenize)

# Stemming words (good reduction of words)
stemmer = SnowballStemmer('english')
stemmezation = lambda words : [stemmer.stem(w) for w in words]
dataset.headline = dataset.headline.apply(stemmezation)

# Removing stopwords: common words that are less useful for detection (example:"the")
stop = set(stopwords.words('english'))
filt = dataset.headline.apply(lambda row: list(filter(lambda w: w not in stop, row)))
dataset['headline'] = filt

# Bagging words (gives a matrix with the count of each word)
counter = CountVectorizer()
matrix = counter.fit_transform([big_line(dataset.headline)])
words = counter.get_feature_names()
counts = matrix.toarray()]

# Create a list of insignificant words
# (Words which appear no more than twice)
discard = []
for (w,c) in zip(words, counts[0]):
    if c <= 2 : discard.append(w)

# One Hot Encoding 
# First we need to label the words
words = list(set(words)-set(discard))
labels = LabelEncoder().fit(words)

encoded = dataset.headline.apply(labels.transform) # Transforming words to labels
print("OK")
exit()
# exit()
encoded = encoded.apply(lambda x : onehotencoding(x, len(words)))

#### VARIBLES ####
from numpy import array
X = array(encoded.tolist())
Y = array([[x] for x in dataset.is_sarcastic.tolist()]) 
# Splitting dataset.
X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.1)

print(X)
print(Y)
    





##### SOME TESTED STUFF WHICH WAS DISCARDED #####

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
