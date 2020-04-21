import re
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup

def clean_text(text):
    # note: copied function from https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

stemmer = PorterStemmer()
def process_text(df):
    # concat title and main content and strip html
    df['content.fullTextHtml'] = df['content.title'] + " " + df['content.fullTextHtml'].map(lambda text: BeautifulSoup(text, 'html.parser').get_text())
    # clean text
    df['content.fullTextHtml'] = df['content.fullTextHtml'].map(lambda text : clean_text(text))
    # stemming
    df['content.fullTextHtml'].apply(lambda x : stemmer.stem(x)) == df['content.fullTextHtml']
    return df
