# Transformesh
3D mesh classification using Transformer.
*3D Printing Course Project, Tel Aviv University February 2022.*

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

## Generate Random Walks 🤖

Generate a dataset of random walks over mesh vertices. The generated random walk sequences together with their respective mesh labels, are used to train a sequence classification model.


```
PYTHONPATH="." python mesh_random_walks/build_walk_dataset.py 
```

Alternatively, manually configure the dataset parameters by using the `generate_walks_from_dataset` function in `mesh_random_walks/build_walk_dataset.py` as follows:

```
generate_walks_from_dataset(dataset_name="cubes",
                            dataset_path="./data/cubes/",
                            data_split="test",
                            walk_params={'num_walks_per_mesh': 128, 'walk_len': None, 'walk_len_vertices_ratio': 0.5},
                            output_file=output_json,
                            data_augment_rotation=True)
```
* `dataset_name`: The mesh classification dataset, either `cubes` or `shrec`
* `dataset_path`: The path to the mesh dataset directory
* `walk_params`: Set the number of random walks generated for each mesh objest in the dataset, control the walk length (`walk_len`) or limit the length to a ratio of the mesh vertices (`walk_len_vertices_ratio`)
* `output_file`: File to store the generated dataset in
* `data_augment_rotation`: Whether to augment the data with random walks over rotated mesh objects


## Train Mesh Calssification Models 📈

Using the mesh random walk sequences and their labels we train a mesh classification model. 

* To train the **Transformer** model run:
```
PYTHONPATH="." python model/main.py --train_json="/mydir/Transformesh/data/walks/walks_cubes_train_dev_walks_256_ratio_05V_scaled.json" --dev_json="/mydir/Transformesh/data/walks/walks_cubes_train_dev_walks_256_ratio_05V_scaled.json" --test_json="/mydir/Transformesh/data/walks/walks_cubes_test_walks_256_ratio_05V_scaled.json"  --step_features=3 --features_type="dxdydz" --cuda=1 --save="transformesh_large_cubes_scaled_walks_128_05V_epochs_40.pt" --emsize=512 --nhid=2048 --nlayers=6 --nhead=8 --lr=1e-4 --dropout=0.1 --batch_size=128 --epochs=40 --num_walks=128 --max_walk_len=125 --nclasses=22
```

* To train the **GRU** (vanilla MeshWalker) model run:
```
PYTHONPATH="." python gru_model/main.py --train_json="/mydir/Transformesh/data/walks/walks_cubes_train_dev_walks_64_ratio_05V_scaled_rotated.json" --dev_json="/mydir/Transformesh/data/walks/walks_cubes_train_dev_walks_64_ratio_05V_scaled_rotated.json" --test_json="/mydir/Transformesh/data/walks/walks_cubes_test_walks_64_ratio_05V_scaled_rotated.json" --step_features=3 --cuda=1 --save="meshwalker_cubes_rotated_walks_32_05V_epochs_40.pt" --lr=1e-4 --dropout=0.2 --batch_size=128 --epochs=40 --num_walks=32 --max_walk_len=125 --nclasses=22
```

* **Notes:**
  * `nclasses`: Determines the number of potential classes for our classifier, must be set to 22 for `cubes` and 30 for `shrec`
  * `features_type`: Whether to encode walk features using the coordinate translation (`dxdydz`) or the invariant walk features (`edges_ratio_angle`)
  * `step_features`: The number of walk features, set to 3 for coordinate translation (`dxdydz`), set to 2 for invariant walk features (`edges_ratio_angle`)
* **Top network hyperparams for Tranformer:**
  *  `--emsize=512 --nhid=2048 --nlayers=6 --nhead=8 --lr=1e-4 --dropout=0.1 --batch_size=128 --epochs=40 --num_walks=128 --max_walk_len=125`
      * `emsize`: Embedding size (dimension)
      * `nhid`: Feed-forward dimension
      * `nlayers`: Number of Transformer Encoder layers
      * `lr`: Learning rate
      * `dropout`: Dropout
      * `batch_size`: Training batch size
      * `num_walks`: Number of random walks per mesh object to train on
      * `max_walk_len`: What is the maximum random walk length to be used
      * `nhead`: Number of self-attention heads
* **Top network hyperparams for GRU:**
  *   `--lr=1e-4 --dropout=0.2 --batch_size=128 --epochs=40 --num_walks=32 --max_walk_len=125`

### Results

Top model results:

| Model | SHREC (16) Acc. | Cubes Acc. |
|-----------|-------------------------|-------------------------|
| Tranformer      | 76.7%                   | 87.9%                 | 
| GRU             | 94.4%                   | 98.8%                  | 


## Example walks 🐍

* Different random walks for the same mesh object (SHREC dataset):

<img src="/images/gorilla_random_walks.png" alt="drawing" width="600"/>

* Random walks generated for different meshes (SHREC dataset):

<img src="/images/shrec_random_walks.png" alt="drawing" width="600"/>


## References ✍🏽
* *MeshWalker: Deep Mesh Understanding by Random Walks*. Lahav and Tal, 2020. [Repository](https://github.com/AlonLahav/MeshWalker)
* *MeshCNN: A Network with an Edge*. Hanocka et al., 2019. [Repository](https://ranahanocka.github.io/MeshCNN/)
* *Attention Is All You Need*. Vaswani et al., 2017. [Paper](https://arxiv.org/abs/1706.03762)
