import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from time import time
import matplotlib.pyplot as plt
import pandas as pd
from text_vectorizer import TextVectorizer
from lsa import LatentSemanticAnalysis

CATEGORIES : list = ['comp.graphics', 'rec.motorcycles', 'rec.sport.baseball', 'sci.space', 'talk.religion.misc']

def printMetrics(labels: np.ndarray, X: np.ndarray, km: KMeans):
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f"
        % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, km.labels_, sample_size=1000))
    
def showResults(centroids: np.ndarray, terms: np.array, true_k: 3):
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in centroids[i, :20]:
            print(' %s' % terms[ind], end='')
        print()    

def kMean(true_k: int, labels, X, terms: np.array):
    km = KMeans(n_clusters=true_k, init='k-means++', n_init=20, max_iter=100, random_state=42)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    printMetrics(labels, X, km)

    centroids = km.cluster_centers_.argsort()[:, ::-1] ## Indices of largest centroids' entries in descending order
    showResults(centroids, terms, true_k)

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

    textVectorizer = TextVectorizer(data)
    X: np.array = textVectorizer.X
    terms: np.array = textVectorizer.terms

    # Part I
    kMean(true_k, labels, X, terms)

    # Part II
    lsa = LatentSemanticAnalysis(100, 5, X, terms)
    X = lsa.X

    # Part III
    kMean(true_k, labels, X, terms)

if __name__ == "__main__":
    main()