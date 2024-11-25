import faiss
import os
import numpy
import timeit 

TIMES = 7

def get_memory(index):
    # write index to file
    faiss.write_index(index, './temp.index')
    # get file size
    file_size = os.path.getsize('./temp.index')
    # delete saved index
    os.remove('./temp.index')
    return file_size

def score(sub: numpy.ndarray, base: numpy.ndarray):
    return numpy.mean([1 if i in sub else 0 for i in base[0]])

class ExperimentResult:
    def __init__(self, memory: int):
        self.memory = memory
        self.times = []
        self.accuracies = []
    def time_mean(self):
        return numpy.array(self.times).mean()
    def time_std(self):
        return numpy.array(self.times).std()
    def accuracy(self):
        return numpy.array(self.accuracies).mean()    

def runExperiment(index, queries: numpy.ndarray, k: int, baselineDocs: numpy.ndarray) -> ExperimentResult:
    result = ExperimentResult(get_memory(index))
    for i in range(TIMES):
        now = timeit.default_timer()
        _, docs = index.search(queries[0:1], k)
        result.times.append(timeit.default_timer() - now)
        result.accuracies.append(score(docs, baselineDocs))
        result.docs = docs
    return result