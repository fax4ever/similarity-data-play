import faiss
import fileutil
import nputil

def main():
    # uncomment if you need to download the dataset
    # fileutil.sift1m()

    # data we will search through
    xb = nputil.read_fvecs("./sift/sift_base.fvecs") # 1M samples
    # also get some query vectors to search with
    xq = nputil.read_fvecs('./sift/sift_query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    xq_0 = xq[0:1] #0:1 to keep first query point
    print(xb.shape, xq.shape, xq_0.shape)
    print(xb)
    print(xq)

    d = 128  # dimensionality of Sift1M data
    k = 10  # number of nearest neighbors to return
    index = faiss.IndexFlatIP(d)
    index.add(xb)
    D, I = index.search(xq_0, k) #use xq --> using all queries [it probably runs out of memory on Colab]
    print(D.shape, I.shape)

    # D = "distances"
    # INSTEAD: scalar product of vectors, so in fact: the opposite of distance
    print(D)
    #document indices
    print(I)
    #all documents content (vectorial representation)
    print(xb[I])

    first_doc = xb[I][:,0]
    last_doc = xb[I][:,-1]
    print((xq_0*first_doc).sum(), (xq_0*last_doc).sum())

    nbits = d*4  # resolution of bucketed vectors
    # initialize index and add vectors
    index = faiss.IndexLSH(d, nbits)
    index.add(xb)
    # and search
    D, I = index.search(xq_0, k)

    # set HNSW index parameters
    M = 8  # number of connections each vertex will have
    ef_search = 8  # depth of layers explored during search
    ef_construction = 8  # depth of layers explored during index construction

    # initialize index (d == 128)
    index = faiss.IndexHNSWFlat(d, M)
    # set efConstruction and efSearch parameters
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    # add data to index
    index.add(xb)

    # search as usual
    D, I = index.search(xq_0, k)

    nlist = 128  # number of cells/clusters to partition data into

    quantizer = faiss.IndexFlatIP(d)  # how the vectors will be stored/compared
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.train(xb)  # we must train the index to cluster into cells
    index.add(xb)

    index.nprobe = 8  # set how many of nearest cells to search
    D, I = index.search(xq_0, k)

    gamm = xb.shape[1]
    m = 8
    assert gamm % m == 0
    nbits = 8  # number of bits per subquantizer, k* = 2**nbits
    index = faiss.IndexPQ(gamm, m, nbits)
    index.train(xb)
    k = 100
    index.add(xb)
    dist, I = index.search(xq_0, k)

if __name__ == "__main__":
    main()