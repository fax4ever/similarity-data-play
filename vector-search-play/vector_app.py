from dataset import Dataset
from flat_cos_similarity import FlatCosSimilarity

def main():
    dataset = Dataset()
    # uncomment if you need to download the dataset files:
    # dataset.download()
    dim = dataset.dim()
    data = dataset.data()
    query = dataset.query()
    k = 10  # number of nearest neighbors to return
    print("data shape:", data.shape, "query shape:", query.shape)
    flatCosSimilarity = FlatCosSimilarity(dim)
    flatCosSimilarity.indexing(data)
    flatCosSimilarity.query(query, k)
    print("PART 2: locality sentive hasing cosine similarity index")

if __name__ == "__main__":
    main()