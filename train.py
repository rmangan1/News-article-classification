import nltk
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
import pickle

# download stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# load X_train, y_train from file
X_train = pd.read_hdf('train_store.h5', 'X_train')
y_train = pd.read_hdf('train_store.h5', 'y_train')

# define pipeline consisting of feature extractor and multi-label classifier
RF_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(RandomForestClassifier(random_state=0))),
            ])

categories = y_train.columns.values

# suppress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# train model and save to file
RF_pipeline.fit(X_train, y_train[categories])
pickle.dump((RF_pipeline, categories), open('pipeline.pickle', 'wb'))
