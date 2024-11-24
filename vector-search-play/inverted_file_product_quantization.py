import numpy
import helper
import faiss
import timeit

class InvertedFileProductQuantization:
    def __init__(self, dim: int):
        print("# Inverted File Product Quantization")
        m = 8
        assert dim % m == 0
        vecs = faiss.IndexFlatL2(dim)
        nlist = 2048  # how many Voronoi cells (must be >= k* which is 2**nbits)
        nbits = 8  # when using IVF+PQ, higher nbits values are not supported
        self.index = faiss.IndexIVFPQ(vecs, dim, nlist, m, nbits)

    def indexing(self, data: numpy.ndarray):
        self.index.train(data)
        self.index.add(data)
        print("index size:", helper.get_memory(self.index))

    def query(self, query: numpy.ndarray, k: int):
        start = timeit.default_timer()
        self.distances, self.docIndexes = self.index.search(query, k)
        print("query time:", timeit.default_timer() - start)
        print("knn doc indexes:", self.docIndexes)
        print("distances:", self.distances)