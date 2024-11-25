import numpy
import helper
import faiss
import timeit

class ProductQuantization:
    def __init__(self, dim: int):
        print("# Product Quantization")
        m = 8
        assert dim % m == 0
        nbits = 8  # number of bits per subquantizer, k* = 2**nbits
        self.index = faiss.IndexPQ(dim, m, nbits)

    def indexing(self, data: numpy.ndarray):
        self.index.train(data)
        self.index.add(data)
        print("index size:", helper.get_memory(self.index))

    def query(self, queries: numpy.ndarray, k: int, base: numpy.ndarray):
        self.experiment = helper.runExperiment(self.index, queries, k, base)
        print("index memory:", self.experiment.memory)
        print("query times:", self.experiment.times)
        print("query accuracies:", self.experiment.accuracies)    