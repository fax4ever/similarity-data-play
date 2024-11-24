import numpy
import helper
import faiss
import timeit

class SmallWorldGraph:
    def __init__(self, dim: int):
        print("# Hierarchical Navigable Small World Graphs")
        M = 8  # number of connections each vertex will have
        ef_search = 8  # depth of layers explored during search
        ef_construction = 8  # depth of layers explored during index construction
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

    def indexing(self, data: numpy.ndarray):
        self.index.add(data)
        print("index size:", helper.get_memory(self.index))

    def query(self, query: numpy.ndarray, k: int):
        start = timeit.default_timer()
        self.innerProducts, self.docIndexes = self.index.search(query, k)
        print("query time :", timeit.default_timer() - start)
        print("knn doc indexes", self.docIndexes)
        print("doc inner products", self.innerProducts)