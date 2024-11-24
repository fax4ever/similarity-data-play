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

    def query(self, query: numpy.ndarray, k: int, base: numpy.ndarray):
        self.experiment = helper.runExperiment(self.index, query, k, base)
        print("index memory:", self.experiment.memory)
        print("query times:", self.experiment.times)
        print("query accuracies:", self.experiment.accuracies)    