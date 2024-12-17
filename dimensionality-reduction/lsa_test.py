import numpy as np
from lsa import LatentSemanticAnalysis

def main():
    movies = ['Matrix', 'Alien', 'Serenity', 'Casablanca', 'Amelie']
    X = np.array([[1,1,1,0,0], [3,3,3,0,0], [4,4,4,0,0], [5,5,5,0,0], [0,2,0,4,4], [0,0,0,5,5],
                   [0,1,0,2,2]])
    print(X)
    lsa = LatentSemanticAnalysis(5, 2, X, movies)
    with np.printoptions(precision=2, suppress=True):
        print(lsa.X)

if __name__ == "__main__":
    main()