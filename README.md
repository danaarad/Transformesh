# Transformesh
Mesh Classification Using Transformers - 3D Printing Course Project 2021

## Setup ⚙️
To run the evaluation script locally, using a *conda virtual environment*, do the following:

1. Create a virtual environment
```
conda create -n [ENV_NAME] python=3.8
conda activate [ENV_NAME]
```

2. Install requirements
```
pip install -r requirements.txt 
```

3. Run in shell
```
PYTHONPATH="." python3.7 scripts/evaluate_predictions.py 
--dataset_file=/labels/labels.csv \
--preds_file=/predictions/predictions.csv \
--no_cache \
--output_file_base=/results/results \
--metrics ged_scores exact_match sari normalized_exact_match \
```


### Download Datasets
https://ranahanocka.github.io/MeshCNN/

https://www.dropbox.com/s/2bxs5f9g60wa0wr/cubes.tar.gz  🧊

https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz 🐉

### Generate Random Walks

### Run
Our evaluator should receive three files as input, the dataset true labels, the model's prediction file and the path to the output file. We therefore *bind mount* the relevant files when using `docker run`. 
The specific volume mounts, given our relevant files are storem in `tmp`, will be:
```
-v "$(pwd)"/tmp/results/:/results:rw
-v "$(pwd)"/tmp/predictions/:/predictions:ro
-v "$(pwd)"/tmp/labels/:/labels:ro
```

## Refernces
MeshWalker
MeshCNN
Transformer
