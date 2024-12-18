from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from time import time

class SingleValueDecomposition:
    def __init__(self, X_tfidf):
        self.lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
        t0 = time()
        self.X_lsa = self.lsa.fit_transform(X_tfidf)
        explained_variance = self.lsa[0].explained_variance_ratio_.sum()
        print(f"LSA done in {time() - t0:.3f} s")
        print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")
