# Experiments

This directory contains training scripts, configuration files and documentation to reproduce the experiments 
of the [HodgeFormer paper](https://arxiv.org/abs/2509.01839). Each dataset used in the paper experiments has 
its own subfolder. In total, there are experiments for four different datasets on the tasks of *mesh classification*
and *mesh segmentation*:

- SHREC-11 - *mesh classification*: [link](./classification_shrec/)
- Cube Engraving - *mesh classification*: [link](./classification_cube_engraving/)
- COSEG Chairs, Aliens, Vases - *mesh segmentation*: [link](./segmentation_coseg/)
- Human - *mesh segmentation*: [link](./segmentation_human/)

### Datasets

To download the datasets used, follow the instructions in the *Data* section of the documentation for each
experiment. These sections includes the links to the original sources from where the datasets were retrieved.
Alternatively, we provide a link to a Google Drive folder with all the used datasets [here](https://drive.google.com/drive/folders/1zloiErQ4i-WwupksrTCe6LeACF5SJvHn?usp=sharing).


### Training HodgeFormer models : Configuration File 

Training HodgeFormer models is configured with configuration files in `.toml` format. Example configuration
files are included for each dataset. The configuration file includes the following sections:
- `[dataset]`
- `[training]`
- `[wandb]`


#### The `[dataset]` section

The `[dataset]` section is used for specifying the input dataset and arguments on data loading and dataloaders 
configuration. Due to differences in input dataset format and structure, the `[dataset]` section in the configuration
files are slightly different regarding input paths and split ratios. All other sections are similar.

- `[dataset.train]` and `[dataset.test]` sections:

These include arguments about the applied transformations on input meshes and the neighbor selection operations 
for applying sparse attention. These are configured in separate sections for train and test set. For neighbor
selection the same parameters should be used across splits.

```toml
[dataset.train.kw.extract_kw]
mode = "bigbird"         # mode used for extracting neighbors. Selected from {"neighbors", "bigbird"}

[dataset.train.kw.extract_kw.nbor_kw]
v_max = 16               # number of maximum selected neighbors for vertices
e_max = 32               # number of maximum selected neighbors for edges
f_max = 16               # number of maximum selected neighbors for faces
dilations = 1            # experimental: use dilations=1
v_k = 8                  # number of maximum BFS hop operations to be used to reach `v_max`
e_k = 8                  # number of maximum BFS hop operations to be used to reach `e_max`
f_k = 8                  # number of maximum BFS hop operations to be used to reach `f_max`
clip_nbors = false       # whether to clip neighbors to enforce O(n**1.5) complexity
# Below keyword arguments apply only to `bigbird` mode
modes = ["r", "l"]       # Specify the type of node selection modes with "r" = random, "l" = local (BFS) nodes. 
percs = [0.2, 0.8]       # Specify the ratio of random ("r") vs local ("l") nodes.

[dataset.test.kw]
# Same parameters
```

- `[dataset.dataloader.train]` and `[dataset.dataloader.test]` sections:

These are keyword arguments passed straight to the torch `torch.utils.data.DataLoader` class. For these check the 
official `torch` [documentation](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). 

```toml
[dataset.dataloader.train]
batch_size = 16
shuffle = true
num_workers = 12
multiprocessing_context = "forkserver"    # Here must use `forkserver` 
persistent_workers = true                  
drop_last = false

[dataset.dataloader.test]
# Same parameters
```

> [!NOTE]
> Use `multiprocessing_context == "forkserver"` to avoid undefined behavior due to how the child processes are forked. 
  Using `fork` results to a crash most likely due to OpenMP having been initialized after GraphBLAS import.


#### The `[training]` section

The `[training]` section is used for specifying training parameters such as number of epochs and learning rate, model 
architecture configurations, task head configuration as well as loss and optimizer parameters.

```toml
[train]
epochs = 300                                 # Number of epochs
lr = 5e-4                                    # Learning rate

[train.model]
v_in = 7                                     # Number of input vertex features
e_in = 21                                    # Number of input edge features
f_in = 13                                    # Number of input face features
d_v = 256                                    # Vertex embedding dimension
d_e = 256                                    # Edge embedding dimension 
d_f = 256                                    # Face embedding dimension 
d_hidden = 512                               # Hidden embedding dimension used in transformer MLPs
N = 6                                        # Number of Hodgeformer layers
dropout = 0.1                                # Dropout applied in MLPs
modes = "v"                                  # Activated hodgeformer attention on mesh elements. String like 'v', 'e', 've' etc.
embed_type = "neighbor"                      # Choose from {'neighbor', 'linear'}
attn_type = "hodge"                          # Only 'hodge' is supported. It is exposed for other experimental features
task_type = "classification"                 # Choose from {'classification', 'segmentation'}
layer_layout = "2:1e"                        # How the hodge and vanilla layers are interleaved.

# Embedding layer related parameters
[train.model.embed_kw]
n = 2                         # Number of embedding layers

# HodgeFormer self-attention related parameters
[train.model.attn_kw]
h = 4                                        # Number of attention heads
dropout_attn = 0.1                           # Dropout applied on self attention values
attn_sym = false                             # Enforce symmetry K = Q
attn_mask = false                            # NOT SUPPORTED
norm_type = "layer_norm"                     # Normalization layer applied on Q and K values
hodge_type = "grouped"                       # Type of self attention. 'grouped' corresponds to paper results.
attn_acc = false                             # NOT SUPPORTED

# Vanilla Transformer self-attention related parameters
[train.model.attn_basic_kw]
h = 4                                        # Number of heads for vanilla transformer
dropout_attn = 0.2                           # Dropout on vanilla attention transformer
norm_type = false                            # Normalization layer on Q, K for vanilla transformer 
attn_type = "full-linear-fast-transformers"  # Type of attention used in vanilla transformer
```

Under the `training` section, we specify parameters for regarding the task head based on the task at hand. 
* For a mesh classification task:  

```toml
[train.model.task_kw]
num_classes = 22                             # Number of classes
modes = "v"                                  # Mesh element to get the embeddings from
dropout = 0.1                                # Dropout applied on head layer
head_type = "linear"                         # Type of classification head. Only 'linear' is supported. 
```

* For a mesh segmentation task:
```toml
[train.model.task_kw]
num_classes = 8                              # Number of classes
modes = "v"                                  # Mesh element to get the embeddings from
out = "f"                                    # Mesh element to calculate logits on
dropout = 0.1                                # Dropout applied on head layer
```


#### The `[wandb]` section

For the paper experiments `wandb` was used for visualizing and organizing the results of the experiments. 
Through this section `wandb` can be enabled and configured accordingly.

```toml
[wandb]
WANDB_MODE = "online"         # Options: 'online', 'offline', 'disabled'
name = "hodgeformer-shrec11"  # run name

[wandb.init]
project = "project-name"      # wandb project
entity = "user-wandb-account" # wandb entity
```


### Inference

For using trained models on new input data, one can use the entry point `infer-hodgeformer` installed along 
with the `mesh_opformer` package. Alternatively, one can use directly the script found in the [inference folder](./inference/).

Required inputs:
- `--model`: Path to the trained `.pth` model
- `--cfg_path`: Path to the configuration file used for training the model
- `--dataset_path`: Path to a folder with mesh instances
- `--out`: Folder where the inference results will be stored (as `.json` file).  

Note: For performing inference on single meshes, one can use the `--data_path` argument instead of `--dataset_path` 
and pass the path to the mesh instance. 

Example:
```bash
infer-hodgeformer \
    --model ./runs/classification_cube_engraving_ckpt_300.pth \
    --cfg_path ./cfg/classification_cube_engraving_cfg.toml \
    --dataset_path ./data/cubes/fork/test \
    --out ./out.json
```


### Training HodgeFormer models on New Datasets & Limitations

Models of the HodgeFormer architecture operate agnostically on triangular meshes, and do not have hard constraints whether input meshes
are strictly manifold, have holes or disconnected components. Variation in experiments comes mostly from different dataset formats and 
dataset folder structure. 

For training HodgeFormer models on new datasets, one must define a torch `torch.utils.data.Dataset` class similar to the `MeshMapDataset` 
class found in the module `mesh_opformer.dataset.dataset_map` (or adjust accordingly the existing one).

Limitations in current code version include:
* Memory usage in sparse operations of complexity $O(n^{1.5})$, which limits training large models on meshes >10K vertices. Applying
  a sparse attention layer in FlashAttention style could help result in a complexity of $O(n)$.
* Some rather slow CPU operations during data input (compactify operations coming from GraphBLAS)
