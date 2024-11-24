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
        nlist = 256  # how many Voronoi cells (must be >= k* which is 2**nbits)
        nbits = 8  # when using IVF+PQ, higher nbits values are not supported
        self.index = faiss.IndexIVFPQ(vecs, dim, nlist, m, nbits)
        self.index.nprobe = 48

    def indexing(self, data: numpy.ndarray):
        self.index.train(data)
        self.index.add(data)
        print("index size:", helper.get_memory(self.index))

    def query(self, query: numpy.ndarray, k: int, base: numpy.ndarray):
        self.experiment = helper.runExperiment(self.index, query, k, base)
        print("index memory:", self.experiment.memory)
        print("query times:", self.experiment.times)
        print("query accuracies:", self.experiment.accuracies)