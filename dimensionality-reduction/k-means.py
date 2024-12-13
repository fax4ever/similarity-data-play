import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.random_projection import SparseRandomProjection as srp
from sklearn.cluster import KMeans
from time import time
import numpy as np

categories = [
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]

def printMetrics(labels: np.ndarray, X: np.ndarray, km: KMeans):
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f"
        % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, km.labels_, sample_size=1000))

def showResults(centroids: np.ndarray, vectorizer: TfidfVectorizer, true_k: 3):
    terms = vectorizer.get_feature_names_out()
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()

def main():
    print("Loading 20 newsgroups dataset for categories: ", categories)
    dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=False, 
                                 remove=('headers', 'footers', 'quotes'))
    labels = dataset.target
    true_k = len(np.unique(labels)) ## This should be 3 in this example
    # First, we download the necessary NLTK resources
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')

    # We next perform lemmatization
    lemmatizer = WordNetLemmatizer()
    for i in range(len(dataset.data)):
        word_list = word_tokenize(dataset.data[i])
        lemmatized_doc = ""
        for word in word_list:
            lemmatized_doc = lemmatized_doc + " " + lemmatizer.lemmatize(word)
        dataset.data[i] = lemmatized_doc

    vectorizer = TfidfVectorizer(stop_words='english') ## Corpus is in English
    X = vectorizer.fit_transform(dataset.data).toarray() ## X is a sparse matrix --> we make it dense for fair comparison
    print(X.shape)

    km = KMeans(n_clusters=true_k, init='k-means++', n_init=20, max_iter=100)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    printMetrics(labels, X, km)

    centroids = km.cluster_centers_.argsort()[:, ::-1] ## Indices of largest centroids' entries in descending order
    showResults(centroids, vectorizer, true_k)

    transformer = srp(n_components=1000, dense_output=True) # Using a dense representation for the matrix
    X_proj = transformer.fit_transform(X)
    print("Data shape:", X_proj.shape)

    km = KMeans(n_clusters=true_k, init='k-means++', n_init=20, max_iter=100)
    t0 = time()
    km.fit(X_proj)
    print("done in %0.3fs" % (time() - t0))
    printMetrics(labels, X, km)
    
    centroids = np.zeros((true_k, X.shape[1])) # Initializing true_k centroid arrays
    cluster_sizes = [0]*true_k # For each cluster, the number of points it contains, needed for taking average
    for i in range(X_proj.shape[0]):
        index = int(km.labels_[i]) # index is the index of the cluster the i-th point belongs to
        centroids[index] += X[i] # Adding component-wise
        cluster_sizes[index] += 1

    for i in range(true_k):
        centroids[i] = centroids[i]/cluster_sizes[i] # Computing centroids: take sum and divide by cluster size

    centroids = centroids.argsort()[:, ::-1] ## Indices of largest centroids' entries in descending order
    showResults(centroids, vectorizer, true_k)    

if __name__ == "__main__":
    main()