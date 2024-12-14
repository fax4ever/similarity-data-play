import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def computeSvdSize(U: np.array, S: np.array, V: np.array):
    return U.shape[0] * U.shape[1] + S.shape[0] + V.shape[0] * V.shape[1]

def plot(X : np.array):
    plt.imshow(X, cmap='gray', vmin=0, vmax=256)
    plt.axis('off')
    plt.show()

def info(U: np.array, S: np.array, V: np.array, xSize: int):
    print('U shape - {0}\nS shape - {1}\nV shape - {2}'.format(U.shape, S.shape, V.shape))
    svdSize = computeSvdSize(U, S, V)
    print(f'SVD size - {svdSize}')
    print('Compression rate with all singular values - {0:.02%}'.format(1 - svdSize / xSize))
    plot((U * S).dot(V))

def main():
    # Original matrix
    X = np.array(Image.open('yolka.jpg').convert('L'))
    xSize = X.shape[0] * X.shape[1]
    print(f'Image shape {X.shape}\nImage size - {xSize}')
    plot(X)

    # SVD (full rank)
    U, S, V = np.linalg.svd(X)
    r = len(S)
    u = U[:,:r]     # U is (n x r)
    s = S[:r]
    v = V[:r]       # V is (r x m)
    info(u, s, v, xSize)

    # SVD (using first 10 values)
    i = 10
    u = U[:,:i]     # U is (n x r)
    s = S[:i]
    v = V[:i]       # V is (r x m)
    info(u, s, v, xSize)

    # SVD (using first 25 values)
    i = 25
    u = U[:,:i]     # U is (n x r)
    s = S[:i]
    v = V[:i]       # V is (r x m)
    info(u, s, v, xSize)

    # SVD (using first 50 values)
    i = 50
    u = U[:,:i]     # U is (n x r)
    s = S[:i]
    v = V[:i]       # V is (r x m)
    info(u, s, v, xSize)

    # SVD (using first 75 values)
    i = 75
    u = U[:,:i]     # U is (n x r)
    s = S[:i]
    v = V[:i]       # V is (r x m)
    info(u, s, v, xSize)

    # SVD (using first 100 values)
    i = 100
    u = U[:,:i]     # U is (n x r)
    s = S[:i]
    v = V[:i]       # V is (r x m)
    info(u, s, v, xSize)

    # SVD (using first 300 values)
    i = 300
    u = U[:,:i]     # U is (n x r)
    s = S[:i]
    v = V[:i]       # V is (r x m)
    info(u, s, v, xSize)

    plt.plot(S)
    plt.title('Singular values')
    plt.show()

if __name__ == "__main__":
    main()