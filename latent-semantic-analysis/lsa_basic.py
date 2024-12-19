import numpy as np
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from time import time

class LSA:
    def __init__(self, X: np.array):
        self.pipe = make_pipeline(TruncatedSVD(n_components=100, random_state=1224), Normalizer(copy=False))
        t0 = time()
        self.USigma = self.pipe.fit_transform(X)
        print("U * Sigma", self.USigma.shape)
        print(f"LSA done in {time() - t0:.3f} s")
        Sigma = self.pipe[0].singular_values_
        print("Sigma", Sigma.shape)
        self.V = self.pipe[0].components_
        print("V", self.V.shape)

    def printMostImportantKeywords(self, m: int, columns: np.array):    
        # Applying only **m** components
        Vm = self.V[:m]
        print("V, using m components", Vm.shape)
        data = pd.DataFrame(Vm, columns=columns)
        print("data\n", data)
        
        for i in range(m):
            print("component ", i)
            positives: pd.Series = data.iloc[i].sort_values(ascending=False).head(20)
            print("most positive terms:", list(positives.axes[0]))
            negatives: pd.Series = data.iloc[i].sort_values(ascending=True).head(20)
            print("most negative terms:", list(negatives.axes[0]))