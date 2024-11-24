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

``` bash
python vector_app.py
```