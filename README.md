# Transformesh
Mesh Classification Using Transformers - 3D Printing Course Project 2021

## Setup

### Download Mesh Datasets

We use the SHREC and Engraved Cubes datasets used in [MeshCNN](https://ranahanocka.github.io/MeshCNN/).
* [**SHREC (Split 16)**](https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz) 🐉
* [**Engraved Cubes**](https://www.dropbox.com/s/2bxs5f9g60wa0wr/cubes.tar.gz) 🧊

### Setup the running environment

To run Transformeh Walker locally, use a *conda virtual environment* as follows:

1. Create a virtual environment
```
conda create -n [ENV_NAME] python=3.8
conda activate [ENV_NAME]
```

2. Install requirements
```
pip install -r requirements.txt 
```

3. Move the mesh classification datasets to a data directory, for example:
```
/mydir/Transformesh/data/shrec_16
/mydir/Transformesh/data/cubes
```

### Generate Random Walks 🤖

Generate a dataset of random walks over mesh vertices. The generated random walk sequences together with their respective mesh labels, are used to train a sequence classification model.


```
PYTHONPATH="." python mesh_random_walks/build_walk_dataset.py 
```

### Train Mesh Calssification Models 📈

Using the mesh random walk sequences and their labels we train a mesh classification model. 

* To train the Transformer model run:
```
PYTHONPATH="." python3.7 scripts/evaluate_predictions.py 
--dataset_file=/labels/labels.csv \
--preds_file=/predictions/predictions.csv \
--no_cache \
--output_file_base=/results/results \
--metrics ged_scores exact_match sari normalized_exact_match \
```

* To train the GRU (vanilla MeshWalker) model run:
```
PYTHONPATH="." python3.7 scripts/evaluate_predictions.py 
--dataset_file=/labels/labels.csv \
--preds_file=/predictions/predictions.csv \
--no_cache \
--output_file_base=/results/results \
--metrics ged_scores exact_match sari normalized_exact_match \
```

## Refernces ✍🏽
MeshWalker
MeshCNN
Transformer
