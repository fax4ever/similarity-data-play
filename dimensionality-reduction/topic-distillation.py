import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
from time import time

CATEGORIES : list = ['comp.graphics', 'rec.motorcycles', 'rec.sport.baseball', 'sci.space', 'talk.religion.misc']
K : int = len(CATEGORIES) # The number of clusters will be equal to the number of categories

def printMetrics(labels: np.ndarray, X: np.ndarray, km: KMeans):
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f"
        % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, km.labels_, sample_size=1000))

def main():
    print("Loading 20 newsgroups dataset for categories: ", CATEGORIES)
    dataset = fetch_20newsgroups(subset='all', categories=CATEGORIES, shuffle=False, 
                                 remove=('headers', 'footers', 'quotes'))
    data : list = dataset.data
    print("Imported data: ", type(data), len(data))
    labels : np.array = dataset.target
    print("Imported labels: ", labels.shape, labels)
    true_k = len(np.unique(labels)) ## This should be 5 in this example
    print("True K: ", true_k)

    # First, we download the necessary NLTK resources
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')

     # We next perform lemmatization
    lemmatizer = WordNetLemmatizer()
    for i in range(len(data)):
        word_list = word_tokenize(data[i])
        lemmatized_doc = ""
        for word in word_list:
            lemmatized_doc = lemmatized_doc + " " + lemmatizer.lemmatize(word)
        data[i] = lemmatized_doc

    vectorizer = TfidfVectorizer(stop_words='english') ## Corpus is in English
    X: np.array = vectorizer.fit_transform(dataset.data).toarray()
    print("X: documents (rows) x terms (columns):", X.shape)

    km = KMeans(n_clusters=true_k, init='k-means++', n_init=20, max_iter=100)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    printMetrics(labels, X, km)

if __name__ == "__main__":
    main()