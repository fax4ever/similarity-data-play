import numpy as np
import pandas as pd
from dataset import Dataset
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

def fullSVD(ratings: pd.DataFrame, movies: pd.DataFrame):
    # Limit to top movies so that we don't run out of memory
    n_movies = 1000
    movie_idx_map = {v: i for i, v in enumerate(ratings.groupby('movieId').size().sort_values().tail(n_movies).index)}
    ratings = ratings[ratings.movieId.isin(movie_idx_map)]
    print(ratings)
    
    n_users = len(ratings['userId'].unique())
    print('users', n_users)
    n_movies = len(ratings['movieId'].unique())
    print('movies', n_movies)

    X = np.zeros((n_users, n_movies))
    for u, m, r, _ in ratings.values: # userId movieId rating timestamp
        X[int(u) - 1, movie_idx_map[int(m)]] = r

    k = 20
    # https://www.youtube.com/watch?v=YPe5OP7Clv4
    # U: users to generes (--> rotation)
    # S: generes significance (--> stretch)
    # V: generes to movies (--> rotation)
    U, S, V = np.linalg.svd(X)
    # data: first k-generes to movies 
    data = pd.DataFrame(V[:k], columns=movies.loc[movie_idx_map.keys()].title)
    print(data)

    # Now using any distance measure we can try to extract similar movies directly from V!
    g = (data - data[['Toy Story (1995)']].values).pow(2).mean().sort_values().head(10)
    print(g)
    g = (data - data[['Matrix, The (1999)']].values).pow(2).mean().sort_values().head(10)
    print(g)

def truncatedSVD(ratings: pd.DataFrame, movies: pd.DataFrame):
    l = [x for x in ratings["movieId"].values]
    l = list(set(l))
    l.sort()
    movie_idx_map = {l[i]: i for i in range(len(l))}    # movieId's cannot be directly used as array indices
    n_users = len(ratings['userId'].unique())
    print('users', n_users)
    n_movies = len(ratings['movieId'].unique())
    print('movies', n_movies)

    # Create utility matrix
    X = np.zeros((n_users, n_movies))
    for u, m, r, _ in ratings.values:
        X[int(u) - 1, movie_idx_map[int(m)]] = r

    print(X.shape)

    k = 100
    svd = TruncatedSVD(n_components=k, random_state=42)
    X = svd.fit_transform(X)

    explained_variance_ratio = svd.explained_variance_ratio_
    plt.plot(explained_variance_ratio)
    plt.title('Singular values')
    plt.show()

    X = X[:,:20]
    print(X.shape)

    V = svd.components_
    data = pd.DataFrame(V[:20], columns=movies.loc[movie_idx_map.keys()].title)
    print(data)

    g = (data - data[['Toy Story (1995)']].values).pow(2).mean().sort_values().head(10)
    print(g)
    g = (data - data[['Matrix, The (1999)']].values).pow(2).mean().sort_values().head(10)
    print(g)   

def main():
    dataset = Dataset(False)
    dataset.download()
    ratings = dataset.ratings()
    movies = dataset.movies()
    print(ratings)
    print(movies)
    fullSVD(ratings, movies)
    truncatedSVD(ratings, movies)

if __name__ == "__main__":
    main()