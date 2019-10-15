from nltk import download
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer # Lemmatizer of coice
from nltk.stem import SnowballStemmer # Stemmer of choice

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from os.path import join
import pandas as pd
import re

# Getting what we need from NLTK
download('punkt')
download('stopwords')
download('wordnet')
download('averaged_perceptron_tagger')

def big_line(text, tokenized=True, line=""):
    for sentence in text : 
        if tokenized : 
            for word in sentence : 
                line += " "+word 
        else:
            line += " "+sentence  
    
    return [line]

def clean_up(text, pattern="[.,:%-?()&$'\"!\“\”¯°–―—_\/|#\[\]…@ツ¡\d]"):
    # Cleanning up the data 
    clean_up = lambda txt : re.sub(pattern, '', txt)
    text = text.apply(clean_up)

# Reading dataset
folder = 'Dataset'
dataset = pd.read_json(join(folder, 'Sarcasm_Headlines_Dataset_v2.json'), lines=True)

# Tokenize headlines (nltk tokenizer is more robust with punctuation)
dataset.headline = dataset['headline'].apply(word_tokenize)

# Stemming words (good reduction of words)
stemmer = SnowballStemmer('english')
stemmezation = lambda words : [stemmer.stem(w) for w in words]
dataset.headline = dataset.headline.apply(stemmezation)

# # Lemmatizing words (preserves more meaning)
# wnlt = WordNetLemmatizer()
# lemmatization = lambda phrase : [wnlt.lemmatize(w) for w in phrase]
# dataset.headline = dataset.headline.apply(lemmatization)

# Bagging words (gives a matrix with the count of each word)
counter = CountVectorizer(stop_words='english')
matrix = counter.fit_transform(big_line(dataset.headline))
words = counter.get_feature_names()
counts = matrix.toarray()
bag = pd.DataFrame(counts, columns=words)

# POS tagging (somewhat limited by the flexibility of words)
tagged_data = pos_tag(bag.head())

# Removing stopwords: common words that are less useful for detection (example:"the")
stop = set(stopwords.words('english'))
filt = token_head.apply(lambda row: list(filter(lambda w: w not in stop, row)))
dataset['headline'] = filt
print(filt)

# Splitting dataset.
X, X_test, Y, Y_test = train_test_split(dataset['headline'], dataset['is_sarcastic'], test_size=0.1)

print(X)
print(Y)
    





##### SOME TESTED STUFF WHICH WAS DISCARDED #####

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
