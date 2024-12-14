import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def printSVD(U: np.array, S: np.array, V: np.array):
    print("U: ", U.shape)
    print(U, "\n")
    print("S: ", S.shape)
    print(S, "\n")
    print("V: ", V.shape)
    print(V, "\n")

def smoke(df: pd.DataFrame, singluarValues: int):
    print(df, "\n")
    U, S, V = np.linalg.svd(df)

    plt.plot(S)
    plt.title('Singular values')
    plt.show()

    u = U[:,:singluarValues] # take the first rows
    s = S[:singluarValues] # take the first singluar values
    v = V[:singluarValues] # take the first columns
    printSVD(u, s, v)
    print((u * s).dot(v), "\n")    

def main():
    df = pd.DataFrame([[1,1,1,0,0], [3,3,3,0,0], [4,4,4,0,0], [5,5,5,0,0], [0,0,0,4,4], [0,0,0,5,5],
                   [0,0,0,2,2]], columns=['Matrix', 'Alien', 'Serenity', 'Casablanca', 'Amelie'])
    smoke(df, 2)

    df = pd.DataFrame([[1,1,1,0,0], [3,3,3,0,0], [4,4,4,0,0], [5,5,5,0,0], [0,2,0,4,4], [0,0,0,5,5],
                   [0,1,0,2,2]], columns=['Matrix', 'Alien', 'Serenity', 'Casablanca', 'Amelie'])
    smoke(df, 2)

if __name__ == "__main__":
    main()