# Similarity Data Play

## Create Conda Env

1. List all evns:

``` bash
conda info --envs
```

2. Create new env for the project:

``` bash
conda create -n similarity-data
```

3. Activate the evn

``` bash
conda activate similarity-data
```

4. Install FAISS

``` bash
conda install -c pytorch faiss-cpu=1.9.0
```

5. Install Jupiter

``` bash
conda install jupyter
```

6. Run Jupiter Notebook

{from the local directory used as base for the Notebook}
for instance `/home/fax/code/similarity-data-play/vector-faiss`

``` bash
jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0
```

