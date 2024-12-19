from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn import metrics
from time import time
from lsa_basic import LSA
import matplotlib.pyplot as plt

class LatentSemanticAnalysis:
    def __init__(self, lsa: LSA, true_k: int, labels):
        USigma = lsa.USigma
        Sigma = lsa.pipe[0].singular_values_
        V = lsa.pipe[0].components_
        
        plt.plot(Sigma)
        plt.title('Singular values')
        plt.show()

        # According to plot we have a elbow of a curve approx around m=9
        BASE_M = 4
        bestM = -1
        bestARS = 0
        for i in range(10):
            m = BASE_M + i
            USigmai = USigma[:,:m]
            Vi = V[:m]
            Xi = (USigmai).dot(Vi)
            km = KMeans(n_clusters=true_k, init='k-means++', n_init=5, max_iter=100, random_state=1224)
            km.fit(Xi)
            ars = metrics.adjusted_rand_score(labels, km.labels_)
            print("m components ", m, " - ARS ", ars)
            if (ars > bestARS):
                bestARS = ars
                bestM = m
        
        USigma = USigma[:,:bestM]
        print("U * Sigma", USigma.shape)
        V = V[:bestM]
        print("V", V.shape)
        self.X = (USigma).dot(V)