import numpy as np
from time import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from evaluation import Evaluation
from lsa import LatentSemanticAnalysis

categories = ['comp.graphics', 
              'rec.motorcycles', 
              'rec.sport.baseball', 
              'sci.space', 
              'talk.religion.misc']

def showResults(centroids: np.ndarray, terms: np.array, true_k: 3):
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in centroids[i, :20]:
            print(' %s' % terms[ind], end='')
        print()

def showCentroids(km: KMeans, true_k: int, terms: np.array):
    centroids = km.cluster_centers_.argsort()[:, ::-1] ## Indices of largest centroids' entries in descending order
    showResults(centroids, terms, true_k)

def main():
    # 1. Import the data
    dataset = fetch_20newsgroups(
        remove=("headers", "footers", "quotes"),
        subset="all",
        categories=categories,
        shuffle=True,
        random_state=42,
    )
    labels = dataset.target
    unique_labels, category_sizes = np.unique(labels, return_counts=True)
    true_k = unique_labels.shape[0]
    print(f"{len(dataset.data)} documents - {true_k} categories")

    # 2. Vectorize the data
    vectorizer = TfidfVectorizer(
        max_df=0.5, # ignoring terms that appear in more than 50% of the documents
        min_df=5,   # ignoring terms that are not present in at least 5 documents
        stop_words="english",
    )
    t0 = time()
    X_tfidf = vectorizer.fit_transform(dataset.data)
    print(f"vectorization done in {time() - t0:.3f} s")
    print(f"n_samples: {X_tfidf.shape[0]}, n_features: {X_tfidf.shape[1]}")
    print(f"non-zero-elem: {X_tfidf.nnz / np.prod(X_tfidf.shape):.3f}")
    terms: np.array = vectorizer.get_feature_names_out()

    # 3. Execute the k-means on sparse matrix with independent random initiations n_init
    kmeans = KMeans(
        n_clusters=true_k,
        max_iter=100,
        n_init=5, random_state=42
    )
    eval = Evaluation(labels)
    eval.fit_and_evaluate(kmeans, X_tfidf, name="KMeans\non tf-idf vectors")
    showCentroids(kmeans, true_k, terms)

    # 4. LSA
    lsa = LatentSemanticAnalysis(100, 5, X_tfidf, terms, labels)
    kmeans = KMeans(
        n_clusters=true_k,
        max_iter=100,
        n_init=5, random_state=42
    )
    eval = Evaluation(labels)
    eval.fit_and_evaluate(kmeans, lsa.X, name="KMeans\nwith LSA adv on tf-idf vectors")
    showCentroids(kmeans, true_k, terms)

if __name__ == "__main__":
    main()