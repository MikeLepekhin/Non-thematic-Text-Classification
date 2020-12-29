from os.path import join as pathjoin
import pickle
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk import sent_tokenize, word_tokenize, regexp_tokenize
from nltk.corpus import stopwords
import numpy as np


def save_model(predictor, vectorizer, model_dir):
    with open(pathjoin(model_dir, 'predictor'), 'wb') as fout:
        fout.write(pickle.dumps(predictor))
    with open(pathjoin(model_dir, 'vectorizer'), 'wb') as fout:
        fout.write(pickle.dumps(vectorizer))
        
def load_model(model_dir):
    return pickle.loads(open(pathjoin(model_dir, 'predictor'), 'rb').read()),\
           pickle.loads(open(pathjoin(model_dir, 'vectorizer'), 'rb').read())


class BigVectorizer:
    def __init__(self, max_word_features=5000, max_char_features=10000):
        self.vect_word = TfidfVectorizer(
            max_features=max_word_features, lowercase=True, analyzer='word',
            stop_words=stopwords.words(language), ngram_range=(1,3),dtype=np.float32
        )
        self.vect_char = TfidfVectorizer(
            max_features=max_char_features, lowercase=True, analyzer='char',
            stop_words=stopwords.words(language), ngram_range=(3,6),dtype=np.float32
        )

    def fit_transform(self, X):
        vect_word = self.vect_word.fit_transform(X)
        vect_char = self.vect_char.fit_transform(X)
        return sparse.hstack([vect_word, vect_char])
       
    def transform(self, X):
        vect_word = self.vect_word.transform(X)
        vect_char = self.vect_char.transform(X)
        return sparse.hstack([vect_word, vect_char])
    
    
class CharVectorizer:
    def __init__(self, max_char_features=10000, language='russian'):
        self.vect_char = TfidfVectorizer(
            max_features=max_char_features, lowercase=True, analyzer='char',
            stop_words=stopwords.words(language), ngram_range=(4,4),dtype=np.float32
        )

    def fit_transform(self, X):
        return self.vect_char.fit_transform(X)
       
    def transform(self, X):
        return self.vect_char.transform(X)