import numpy
import helper
import faiss
import timeit

class VectorQuantization:
    def __init__(self, dim: int):
        print("# Vector Quantization")
        nlist = 128  # number of cells/clusters to partition data into
        quantizer = faiss.IndexFlatL2(dim)  # how the vectors will be stored/compared
        self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        self.index.nprobe = 8  # set how many of nearest cells to search

    def indexing(self, data: numpy.ndarray):
        self.index.train(data)
        self.index.add(data)
        print("index size:", helper.get_memory(self.index))

    def query(self, query: numpy.ndarray, k: int, base: numpy.ndarray):
        self.experiment = helper.runExperiment(self.index, query, k, base)
        print("index memory:", self.experiment.memory)
        print("query times:", self.experiment.times)
        print("query accuracies:", self.experiment.accuracies)    