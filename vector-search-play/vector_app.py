import sys
from dataset import Dataset
from flat_cos_similarity import FlatCosSimilarity
from flat_l2_distance import FlatL2Distance
from locality_sensitive_hasing import LocalitySensitiveHasing
from small_world_graph import SmallWorldGraph
from vector_quantization import VectorQuantization
from product_quantization import ProductQuantization
from inverted_file_product_quantization import InvertedFileProductQuantization
from prettytable import PrettyTable 

def main():
    args = sys.argv[1:]
    if not args:
        large = False
    elif args[0] == 'small':
        large = False    
    else:
        large = True    

    dataset = Dataset(large)

    # comment if you don't need to download the dataset files:
    # dataset.download()
    
    dim = dataset.dim()
    data = dataset.data()
    queries = dataset.queries()
    k = 100  # number of nearest neighbors to return
    print("data shape:", data.shape, "queries shape:", queries.shape)

    flatL2Distance = FlatL2Distance(dim)
    flatL2Distance.indexing(data)
    flatL2Distance.query(queries, k)
    accuracyBase = flatL2Distance.docIndexes
    flatL2Distance.score(accuracyBase)

    flatCosSimilarity = FlatCosSimilarity(dim)
    flatCosSimilarity.indexing(data)
    flatCosSimilarity.query(queries, k, accuracyBase)
    
    localitySensitiveHasing = LocalitySensitiveHasing(dim)
    localitySensitiveHasing.indexing(data)
    localitySensitiveHasing.query(queries, k, accuracyBase)
    
    smallWorldGraph = SmallWorldGraph(dim)
    smallWorldGraph.indexing(data)
    smallWorldGraph.query(queries, k, accuracyBase)

    vectorQuantization = VectorQuantization(dim)
    vectorQuantization.indexing(data)
    vectorQuantization.query(queries, k, accuracyBase)

    productQuantization = ProductQuantization(dim)
    productQuantization.indexing(data)
    productQuantization.query(queries, k, accuracyBase)

    invertedFileProductQuantization = InvertedFileProductQuantization(dim)
    invertedFileProductQuantization.indexing(data)
    invertedFileProductQuantization.query(queries, k, accuracyBase)

    myTable = PrettyTable(["Algorithm", "Index size", "Query time avg.", "Query time std. dev.", "Accuracy"])
    myTable.add_row(["Locality Sensitive Hasing", localitySensitiveHasing.experiment.memory, 
                     localitySensitiveHasing.experiment.time_mean(), localitySensitiveHasing.experiment.time_std(), 
                     localitySensitiveHasing.experiment.accuracy()])
    myTable.add_row(["Hierarchical Navigable Small World Graphs", smallWorldGraph.experiment.memory, 
                     smallWorldGraph.experiment.time_mean(), smallWorldGraph.experiment.time_std(), 
                     smallWorldGraph.experiment.accuracy()])
    myTable.add_row(["Vector Quantization", vectorQuantization.experiment.memory, 
                     vectorQuantization.experiment.time_mean(), vectorQuantization.experiment.time_std(), 
                     vectorQuantization.experiment.accuracy()])
    myTable.add_row(["ProductQuantization", productQuantization.experiment.memory, 
                     productQuantization.experiment.time_mean(), productQuantization.experiment.time_std(), 
                     productQuantization.experiment.accuracy()])
    myTable.add_row(["Inverted File Product Quantization", invertedFileProductQuantization.experiment.memory, 
                     invertedFileProductQuantization.experiment.time_mean(), invertedFileProductQuantization.experiment.time_std(), 
                     invertedFileProductQuantization.experiment.accuracy()])
    print(myTable)
if __name__ == "__main__":
    main()