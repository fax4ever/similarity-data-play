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

    def query(self, queries: numpy.ndarray, k: int, base: numpy.ndarray):
        self.experiment = helper.runExperiment(self.index, queries, k, base)
        print("index memory:", self.experiment.memory)
        print("query times:", self.experiment.times)