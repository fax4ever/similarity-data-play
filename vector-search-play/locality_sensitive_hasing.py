import numpy
import helper
import faiss
import timeit

class LocalitySensitiveHasing:
    def __init__(self, dim: int):
        print("# Locality Sentive Hasing")
        nbits = dim*4  # resolution of bucketed vectors
        self.index = faiss.IndexLSH(dim, nbits)

    def indexing(self, data: numpy.ndarray):
        self.index.add(data)
        print("index size:", helper.get_memory(self.index))

    def query(self, query: numpy.ndarray, k: int):
        start = timeit.default_timer()
        self.distances, self.docIndexes = self.index.search(query, k)
        print("query time:", timeit.default_timer() - start)
        print("knn doc indexes:", self.docIndexes)
        print("distances:", self.distances)