import faiss
import numpy as np

def read_fvecs(fp):
    a = np.fromfile(fp, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view('float32')

## ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
xb = read_fvecs('vector-faiss/sift/sift_base.fvecs')
xq = read_fvecs('vector-faiss/sift/sift_query.fvecs')
xq_0 = xq[0:1]
print(xb, xq, xq_0)

d = 128
k = 10
index = faiss.IndexFlatIP(d)
index.add(xb)
D, I = index.search(xq_0, k)
print(D, I)

