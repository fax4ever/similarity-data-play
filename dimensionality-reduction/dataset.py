import numpy as np
import pandas as pd
import shutil
import urllib.request as request
from contextlib import closing
import zipfile

def ml_latest_small():
    with closing(request.urlopen('http://files.grouplens.org/datasets/movielens/ml-latest-small.zip')) as r:
        with open('ml-latest-small.zip', 'wb') as f:
            shutil.copyfileobj(r, f)
    print("file ml-latest-small.zip: downloaded")
    with zipfile.ZipFile('ml-latest-small.zip', 'r') as zip_ref:
        zip_ref.extractall()

class Dataset:
    def __init__(self, large: bool):
        self.large = large

    def download(self):
        if self.large:
            ml_latest_small()
        else:    
            ml_latest_small()

    def ratings(self) -> pd.DataFrame:
        if self.large:
            return pd.read_csv('ml-latest-small/ratings.csv')
        else:    
            return pd.read_csv('ml-latest-small/ratings.csv')

    def movies(self) -> pd.DataFrame:
        if self.large:
            return pd.read_csv('ml-latest-small/movies.csv', index_col=0)
        else:    
            return pd.read_csv('ml-latest-small/movies.csv', index_col=0)            

