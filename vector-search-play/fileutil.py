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