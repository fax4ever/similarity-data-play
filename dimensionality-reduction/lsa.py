import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import pandas as pd

class LatentSemanticAnalysis:
    def __init__(self, components : int, true_k: int, X: np.array, columns: np.array):
        self.svd = TruncatedSVD(n_components=components, n_iter=50, random_state=42)
        X = self.svd.fit_transform(X)
        Sigma = self.svd.singular_values_
        V = self.svd.components_

        plt.plot(Sigma)
        plt.title('Singular values')
        plt.show()

        X = X[:,:true_k]
        print("U * Sigma", X.shape)
        V = V[:true_k]
        print("V", V.shape)

        data = pd.DataFrame(V, columns=columns)
        print("data\n", data)
        data_squared = data
        print("data squared\n", data_squared)
        
        for i in range(true_k):
            component: pd.Series = data_squared.iloc[i].sort_values(ascending=False).head(20)
            print(list(component.axes[0]))
            
        self.X = (X).dot(V)