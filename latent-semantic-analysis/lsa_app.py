import numpy as np
import pandas as pd
from time import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from lsa import LatentSemanticAnalysis
from lsa_basic import LSA
from cloud_words import WordCloudImage

categories = ['comp.graphics', 
              'rec.motorcycles', 
              'rec.sport.baseball', 
              'sci.space', 
              'talk.religion.misc']

def showCentroids(km: KMeans, true_k: int, terms: np.array):
    data = pd.DataFrame(km.cluster_centers_, columns=terms)
    print("data\n", data)
    for i in range(true_k):
        positives: pd.Series = data.iloc[i].sort_values(ascending=False).head(20)
        print("Cluster", i, ":", *list(positives.axes[0]))
        WordCloudImage(positives).show()

def main():
    # 1. Import the data
    dataset = fetch_20newsgroups(
        remove=("headers", "footers", "quotes"),
        subset="all",
        categories=categories,
        shuffle=True,
        random_state=1224,
    )
    labels = dataset.target
    unique_labels, category_sizes = np.unique(labels, return_counts=True)
    true_k = unique_labels.shape[0]
    print(f"{len(dataset.data)} documents - {true_k} categories: {category_sizes}")

    # 2. Vectorize the data
    vectorizer = TfidfVectorizer(
        max_df=0.5, # ignoring terms that appear in more than 50% of the documents
        min_df=5,   # ignoring terms that are not present in at least 5 documents
        stop_words="english"
    )
    X_tfidf = vectorizer.fit_transform(dataset.data)
    terms: np.array = vectorizer.get_feature_names_out()
    print(f"documents: {X_tfidf.shape[0]}, terms: {X_tfidf.shape[1]} {terms.shape} {terms}")
    print(f"the table is very sparse. Only around the {X_tfidf.nnz * 100 / np.prod(X_tfidf.shape):.3f}% of the cells are non zeros.")

    # 3. Execute the k-means on sparse matrix with independent random initiations n_init
    t0 = time()
    kmeans = KMeans(
        n_clusters=true_k,
        max_iter=100,
        n_init=20, random_state=1224
    )
    kmeans.fit(X_tfidf)
    adjustedRandomScore = metrics.adjusted_rand_score(labels, kmeans.labels_)
    print("k-means time: ", time() - t0)
    print("adjusted random score: ", adjustedRandomScore)
    showCentroids(kmeans, true_k, terms)

    # 4. LSA only with SVD
    lsa = LSA(X_tfidf)
    lsa.printMostImportantKeywords(true_k, terms)

    # 5. LSA + k-means
    lsa = LatentSemanticAnalysis(lsa, true_k, labels)
    t0 = time()
    kmeans = KMeans(
        n_clusters=true_k,
        max_iter=100,
        n_init=5, random_state=1224
    )
    kmeans.fit(lsa.X)
    adjustedRandomScore = metrics.adjusted_rand_score(labels, kmeans.labels_)
    print("k-means time: ", time() - t0)
    print("adjusted random score: ", adjustedRandomScore)
    showCentroids(kmeans, true_k, terms)

if __name__ == "__main__":
    main()