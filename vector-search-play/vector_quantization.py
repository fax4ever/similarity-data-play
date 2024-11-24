import numpy
import helper
import faiss
import timeit

class VectorQuantization:
    def __init__(self, dim: int):
        print("# Vector Quantization")
        nlist = 128  # number of cells/clusters to partition data into
        quantizer = faiss.IndexFlatIP(dim)  # how the vectors will be stored/compared
        self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        self.index.nprobe = 8  # set how many of nearest cells to search

    def indexing(self, data: numpy.ndarray):
        self.index.train(data)
        self.index.add(data)
        print("index size:", helper.get_memory(self.index))

    def query(self, query: numpy.ndarray, k: int):
        start = timeit.default_timer()
        self.innerProducts, self.docIndexes = self.index.search(query, k)
        print("query time :", timeit.default_timer() - start)
        print("knn doc indexes", self.docIndexes)
        print("doc inner products", self.innerProducts)