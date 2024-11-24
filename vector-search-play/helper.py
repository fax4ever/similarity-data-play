import faiss
import os
import numpy

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