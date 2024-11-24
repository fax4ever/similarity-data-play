from dataset import Dataset
from flat_cos_similarity import FlatCosSimilarity
from locality_sensitive_hasing import LocalitySensitiveHasing
from small_world_graph import SmallWorldGraph
from vector_quantization import VectorQuantization
from product_quantization import ProductQuantization

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
    localitySensitiveHasing = LocalitySensitiveHasing(dim)
    localitySensitiveHasing.indexing(data)
    localitySensitiveHasing.query(query, k)
    smallWorldGraph = SmallWorldGraph(dim)
    smallWorldGraph.indexing(data)
    smallWorldGraph.query(query, k)
    vectorQuantization = VectorQuantization(dim)
    vectorQuantization.indexing(data)
    vectorQuantization.query(query, k)
    productQuantization = ProductQuantization(dim)
    productQuantization.indexing(data)
    productQuantization.query(query, k)

if __name__ == "__main__":
    main()