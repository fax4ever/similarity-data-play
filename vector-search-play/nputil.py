import numpy as np

# now define a function to read the fvecs file format of Sift1M dataset
def read_fvecs(fp):
    a = np.fromfile(fp, dtype='int32')
    d = a[0]
    result = a.reshape(-1, d + 1)[:, 1:].copy().view('float32')
    return result