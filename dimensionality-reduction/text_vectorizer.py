import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

class TextVectorizer:
    def __init__(self, documents : list):
        self.documents = documents
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('omw-1.4')
        nltk.download('punkt_tab')
        self.lemmatizeDocs()

    def lemmatizeDocs(self):
        self.lemmDocs = []
        lemmatizer = WordNetLemmatizer()
        for i in range(len(self.documents)):
            word_list = word_tokenize(self.documents[i])
            lemmatized_doc = ""
            for word in word_list:
                lemmatized_doc = lemmatized_doc + " " + lemmatizer.lemmatize(word)
            self.lemmDocs.append(lemmatized_doc)
        self.vectorizeDocs()

    def vectorizeDocs(self):
        stop_words = list(set(stopwords.words("english")))
        stop_words += ['wa', 'just', 've', 'ha', 'would', 'one', 'like', 'get', 
                       'could', 'also', 'much', 'know']
        print(stop_words)

        vectorizer = TfidfVectorizer(lowercase=True, stop_words=stop_words) ## Corpus is in English
        self.X: np.array = vectorizer.fit_transform(self.lemmDocs).toarray()
        print("X: documents (rows) x terms (columns):", self.X.shape)
        self.terms: np.array = vectorizer.get_feature_names_out()
        print("terms:", self.terms, self.terms.shape) 