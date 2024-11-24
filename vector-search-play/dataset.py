import numpy
import shutil
import urllib.request as request
from contextlib import closing
import tarfile

def sift1m():
    # first we download the Sift1M dataset
    with closing(request.urlopen('ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz')) as r:
        with open('sift.tar.gz', 'wb') as f:
            shutil.copyfileobj(r, f)
    print("file sift.tar.gz: downloaded")        
    # the download leaves us with a tar.gz file, we unzip it
    tar = tarfile.open('sift.tar.gz', "r:gz")
    print("file sift.tar.gz: extracted")
    tar.extractall()

def sift10k():
    # first we download the SIFT10K dataset
    with closing(request.urlopen('ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz')) as r:
        with open('siftsmall.tar.gz', 'wb') as f:
            shutil.copyfileobj(r, f)
    print("file siftsmall.tar.gz: downloaded")
    # the download leaves us with a tar.gz file, we unzip it
    tar = tarfile.open('siftsmall.tar.gz', "r:gz")
    print("file siftsmall.tar.gz: extracted")
    tar.extractall()

def read_fvecs(fp):
    a = numpy.fromfile(fp, dtype='int32')
    d = a[0]
    result = a.reshape(-1, d + 1)[:, 1:].copy().view('float32')
    return result

class Dataset:
    def __init__(self):
        pass

    def download(self):
        sift1m()

    def data(self) -> numpy.ndarray:
        return read_fvecs("./sift/sift_base.fvecs")
    
    def query(self) -> numpy.ndarray:
        xq = read_fvecs('./sift/sift_query.fvecs')
        return xq[0:1]
    
    def dim(self) -> int:
        return 128