import numpy
import helper
import faiss
import timeit

class FlatCosSimilarity:
    def __init__(self, dim: int):
        print("# Flat cosine similarity (inner product) index")
        self.index = faiss.IndexFlatIP(dim)

    def indexing(self, data: numpy.ndarray):
        self.index.add(data)
        print("index size:", helper.get_memory(self.index))

    def query(self, query: numpy.ndarray, k: int):
        start = timeit.default_timer()
        self.innerProducts, self.docIndexes = self.index.search(query, k)
        print("query time :", timeit.default_timer() - start)
        print("knn doc indexes", self.docIndexes)
        print("doc inner products", self.innerProducts)