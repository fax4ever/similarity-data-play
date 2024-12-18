import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from time import time

class LatentSemanticAnalysis:
    def __init__(self, components: int, true_k: int, X: np.array, columns: np.array, labels):
        self.lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
        t0 = time()
        X = self.lsa.fit_transform(X)
        print(f"LSA done in {time() - t0:.3f} s")
        Sigma = self.lsa[0].singular_values_
        V = self.lsa[0].components_
        
        plt.plot(Sigma)
        plt.title('Singular values')
        plt.show()

        # According to plot we have a elbow of a curve approx around m=9
        BASE_M = 4
        bestM = -1
        bestARS = 0
        for i in range(10):
            m = BASE_M + i
            Xi = X[:,:m]
            Vi = V[:m]
            Xii = (Xi).dot(Vi)
            km = KMeans(n_clusters=true_k, init='k-means++', n_init=5, max_iter=100, random_state=42)
            km.fit(Xii)
            ars = metrics.adjusted_rand_score(km.labels_, labels)
            print("m components ", m, " - ARS ", ars)
            if (ars > bestARS):
                bestARS = ars
                bestM = m
        
        X = X[:,:bestM]
        print("U * Sigma", X.shape)
        V = V[:bestM]
        print("V", V.shape)

        data = pd.DataFrame(V, columns=columns)
        print("data\n", data)
        data_squared = data
        print("data squared\n", data_squared)
        
        for i in range(bestM):
            component: pd.Series = data_squared.iloc[i].sort_values(ascending=False).head(20)
            print(list(component.axes[0]))
            
        self.X = (X).dot(V)