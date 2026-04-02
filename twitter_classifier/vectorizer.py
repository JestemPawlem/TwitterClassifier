from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle


cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'stopwords.pkl'), 'rb'))

def tokenizer(text):
    text = text.lower()
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\W+', ' ', text)

    tokens = [word for word in text.split() if word not in stop]
    return tokens

vect = HashingVectorizer(decode_error='ignore',
                         ngram_range=(1, 1),
                         n_features=2**21,
                         tokenizer=tokenizer)