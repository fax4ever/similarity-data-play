import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics

class LatentSemanticAnalysis:
    def __init__(self, components: int, true_k: int, X: np.array, columns: np.array, labels):
        self.svd = TruncatedSVD(n_components=components, n_iter=50, random_state=42)
        X = self.svd.fit_transform(X)
        Sigma = self.svd.singular_values_
        V = self.svd.components_

        plt.plot(Sigma)
        plt.title('Singular values')
        plt.show()

        # According to plot we have a elbow of a curve approx around m=10
        BASE_M = 5
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