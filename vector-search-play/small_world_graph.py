import numpy
import helper
import faiss
import timeit

class SmallWorldGraph:
    def __init__(self, dim: int):
        print("# Hierarchical Navigable Small World Graphs")
        M = 8  # number of connections each vertex will have
        ef_search = 32  # depth of layers explored during search
        ef_construction = 32  # depth of layers explored during index construction
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

    def indexing(self, data: numpy.ndarray):
        self.index.add(data)
        print("index size:", helper.get_memory(self.index))

    def query(self, queries: numpy.ndarray, k: int, base: numpy.ndarray):
        self.experiment = helper.runExperiment(self.index, queries, k, base)
        print("index memory:", self.experiment.memory)
        print("query times:", self.experiment.times)
        print("query accuracies:", self.experiment.accuracies)    