from nltk import download
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from textblob import TextBlob
from os.path import join
import pandas as pd
import re

# Getting what we need from NLTK
download('punkt')
download('stopwords')
download('wordnet')

# Reading dataset
folder = 'Dataset'
dataset = pd.read_json(join(folder, 'Sarcasm_Headlines_Dataset_v2.json'), lines=True)

# Perfectly balanced, as all things should be...
sarcastic = dataset[dataset['is_sarcastic']==1]
legit = dataset[dataset['is_sarcastic']==0]
print(f'Sarcastic: {len(sarcastic)*100/(len(sarcastic)+len(legit))}%')
print(f'Legitimate: {len(legit)*100/(len(sarcastic)+len(legit))}%')

# Cleanning up the data 
pattern = "[.,:%-?()&$'\"!\“\”¯°–―—_\/|#\[\]…@ツ¡\d]" 
clean_up = lambda txt : re.sub(pattern, '', txt)
dataset.headline = dataset.headline.apply(clean_up)

# Tokenize headlines
token_head = dataset['headline'].apply(word_tokenize)
print(token_head)

# Lemmatize all words
lemmatizer = WordNetLemmatizer()
lemmatization = lambda word : lemmatizer.lemmatize(word)
dataset.headline = dataset.headline.apply(lemmatization)

# Subjectivity in sarcasm (not very usefull)
sarcastic = dataset[dataset['is_sarcastic']==1]
legit = dataset[dataset['is_sarcastic']==0]
subjectivity = lambda x: TextBlob(x).sentiment.subjectivity
sum(sarcastic.headline.apply(subjectivity))/len(sarcastic)
sum(legit.headline.apply(subjectivity))/len(legit)

# Create bag of words
words_bags = CountVectorizer(stop_words='english')
words_matrix = words_bags.fit_transform(dataset.headline)
words = words_bags.get_feature_names()
content = words_matrix.toarray()
words_bags = pd.DataFrame(content, columns=words)

# Removing stopwords: common words that are less useful for detection (example:"the")
stop = set(stopwords.words('english'))
filt = token_head.apply(lambda row: list(filter(lambda w: w not in stop, row)))
dataset['headline'] = filt
print(filt)

# Splitting dataset.
X, X_test, Y, Y_test = train_test_split(dataset['headline'], dataset['is_sarcastic'], test_size=0.1)

print(X)
print(Y)
    