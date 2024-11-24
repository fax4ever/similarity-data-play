from dataset import Dataset

def main():
    dataset = Dataset()
    # uncomment if you need to download the dataset files:
    # dataset.download()
    dim = dataset.dim()
    data = dataset.data()
    query = dataset.query()
    k = 10  # number of nearest neighbors to return
    print("data shape:", data.shape, "query shape:", query.shape)

if __name__ == "__main__":
    main()