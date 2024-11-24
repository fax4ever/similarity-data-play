import numpy
import helper
import faiss
import timeit

class FlatL2Distance:
    def __init__(self, dim: int):
        print("# Flat L2 distance")
        self.index = faiss.IndexFlatL2(dim)

    def indexing(self, data: numpy.ndarray):
        self.index.add(data)
        print("index size:", helper.get_memory(self.index))

    def query(self, query: numpy.ndarray, k: int):
        start = timeit.default_timer()
        self.distances, self.docIndexes = self.index.search(query, k)
        print("query time:", timeit.default_timer() - start)
        print("knn doc indexes:", self.docIndexes)
        print("distances:", self.distances)

    def score(self, base: numpy.ndarray):
        print("accuracy:", helper.score(self.docIndexes, base))