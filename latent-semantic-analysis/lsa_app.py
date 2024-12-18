import numpy as np
from time import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from evaluation import Evaluation
from single_value_dec import SingleValueDecomposition
from lsa import LatentSemanticAnalysis

categories = ['comp.graphics', 
              'rec.motorcycles', 
              'rec.sport.baseball', 
              'sci.space', 
              'talk.religion.misc']

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

    # 3. Execute the k-means on sparse matrix with single n_init
    for seed in range(5):
        kmeans = KMeans(
            n_clusters=true_k,
            max_iter=100,
            n_init=1,
            random_state=seed,
        ).fit(X_tfidf)
        cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
        print(f"Number of elements assigned to each cluster: {cluster_sizes}")
    print(
        "True number of documents in each category according to the class labels: "
        f"{category_sizes}"
    )

    # 3. Execute the k-means on sparse matrix with independent random initiations n_init
    kmeans = KMeans(
        n_clusters=true_k,
        max_iter=100,
        n_init=5,
    )
    eval = Evaluation(labels)
    eval.fit_and_evaluate(kmeans, X_tfidf, name="KMeans\non tf-idf vectors")

    # 4. SVD
    svd = SingleValueDecomposition(X_tfidf)
    kmeans = KMeans(
        n_clusters=true_k,
        max_iter=100,
        n_init=1,
    )
    eval = Evaluation(labels)
    eval.fit_and_evaluate(kmeans, svd.X_lsa, name="KMeans\nwith LSA on tf-idf vectors")

    # 5. LSA
    terms: np.array = vectorizer.get_feature_names_out()
    lsa = LatentSemanticAnalysis(100, 5, X_tfidf, terms, labels)
    kmeans = KMeans(
        n_clusters=true_k,
        max_iter=100,
        n_init=1,
    )
    eval = Evaluation(labels)
    eval.fit_and_evaluate(kmeans, lsa.X, name="KMeans\nwith LSA adv on tf-idf vectors")

if __name__ == "__main__":
    main()