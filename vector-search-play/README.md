## Install Faiss

1. Create new env for the project:

``` bash
conda create -n similarity-data
```

2. Activate the evn

``` bash
conda activate similarity-data
```

3. Install FAISS

``` bash
conda install -c pytorch faiss-cpu=1.9.0
```

## Run the project

4. Using a small data set

``` bash
python vector_app.py
```

4. Using a large data set

``` bash
python vector_app.py
```

# Results

accuracy based on `Flat L2 distance` results

## data shape: (1000000, 128) query shape: (1, 128)

### Flat L2 distance
index size: 512000045
query time: 0.021491901001354563
accuracy: 1.0

### Flat cosine similarity (inner product)
index size: 512000045
query time: 0.023051189000398153
accuracy: 0.93

### Locality Sentive Hasing
index size: 64262237
query time: 0.020210244998452254
accuracy: 0.29

### Hierarchical Navigable Small World Graphs
index size: 592566706
query time: 0.00016362499991373625
accuracy: 0.16

### Vector Quantization
index size: 520066699
query time: 0.004490281000471441
accuracy: 0.97

### Product Quantization
index size: 8131158
query time: 0.018357291000938858
accuracy: 0.39

### Inverted File Product Quantization
index size: 17196212
query time: 0.00027614999999059364
accuracy: 0.38

## data shape: (10000, 128) query shape: (1, 128)

### Flat L2 distance
index size: 5120045
query time: 0.0002623189993755659
accuracy: 1.0

### Flat cosine similarity (inner product)
index size: 5120045
query time: 0.00022708399956172798
accuracy: 0.98

### Locality Sentive Hasing
index size: 902237
query time: 0.019899192000593757
accuracy: 0.56

### Hierarchical Navigable Small World Graphs
index size: 5925650
query time: 3.027700040547643e-05
accuracy: 0.26

### Vector Quantization
index size: 5266699
query time: 7.814200034772512e-05
accuracy: 0.84

### Product Quantization
index size: 211158
query time: 0.012105967998650158
accuracy: 0.66

### Inverted File Product Quantization
index size: 424372
query time: 0.00013154400039638858
accuracy: 0.63
