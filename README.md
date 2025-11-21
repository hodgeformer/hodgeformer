# HodgeFormer: Transformers for Learnable Operators on Triangular Meshes through Data-Driven Hodge Matrices

This repository holds code for the HodgeFormer deep learning architecture operating on mesh data.



## Paper & Code availability 

*Paper Arxiv link:*  [https://arxiv.org/abs/2509.01839](https://arxiv.org/abs/2509.01839)


*Project page:* 
<a href="https://hodgeformer.github.io/" target="_blank">https://hodgeformer.github.io/</a>


*Code available in the following link:* 
<a href="https://github.com/hodgeformer/hodgeformer" target="_blank">https://github.com/hodgeformer/hodgeformer</a>


*Reviewed on Openreview:* 
<a href="https://openreview.net/forum?id=PCbFYiMhlO" target="_blank">https://openreview.net/forum?id=PCbFYiMhlO</a>



## Abstract

Currently, prominent Transformer architectures applied on graphs and meshes for shape analysis tasks employ traditional attention layers that heavily utilize spectral features requiring costly eigenvalue decomposition-based methods. To encode the mesh structure, these methods derive positional embeddings that heavily rely on eigenvalue decomposition based operations, e.g. on the Laplacian matrix, or on heat-kernel signatures, which are then concatenated to the input features.

This paper proposes a novel approach inspired by the explicit construction of the Hodge Laplacian operator in Discrete Exterior Calculus as a product of discrete Hodge operators and exterior derivatives, i.e. $L := \star_0^{-1} d_0^T \star_1 d_0$. We adjust the Transformer architecture in a novel deep learning layer that utilizes the multi-head attention mechanism to approximate Hodge matrices $\star_0, \star_1$ and $\star_2$ and learn families of discrete operators $L$ that act on mesh vertices, edges and faces.

Our approach results in a computationally-efficient architecture that achieves comparable performance in mesh segmentation and classification tasks, through a direct learning framework, while eliminating the need for costly eigenvalue decomposition operations or complex preprocessing operations.

## Problem statement 

Existing methods for 3D mesh analysis using spectral features rely on costly eigendecomposition of Laplacian matrices, creating a computational bottleneck and exhibiting high complexity. 

Alternatives convolutional based methods are often constrained by architectural limitations: some require specific mesh connectivity to construct their operators or use fixed operators that cannot adapt to the underlying data. 

Modern Transformer-based still depend on pre-computed spectral features for positional encoding. This reliance on expensive, rigid, and often complex preprocessing steps limits the efficiency, scalability, and flexibility of deep learning on meshes.


## Core Contribution


This paper proposes a novel approach inspired by the explicit construction of the Hodge Laplacian operator in Discrete Exterior Calculus as a product of discrete Hodge operators and exterior derivatives, i.e. $(L := \star_0^{-1} d_0^T \star_1 d_0)$. We adjust the Transformer architecture in a novel deep learning layer that utilizes the multi-head attention mechanism to approximate Hodge matrices $\star_0$, $\star_1$ and $\star_2$ and learn families of discrete operators $L$ that act on mesh vertices, edges and faces. Our approach results in a computationally-efficient architecture that achieves comparable performance in mesh segmentation and classification tasks, through a direct learning framework, while eliminating the need for costly eigenvalue decomposition operations or complex preprocessing operations. 




## Modules

The project is split into three separate python packages:

- The `mesh_sim` package supports reading meshes using Signed Incidence Matrices as base data
structure and includes functionalities for extracting useful geometric features. For experiments 
`mesh_o3d` is used instead of `mesh_sim` in several places.

- The `mesh_opformer` package includes the layer definitions of the HodgeFormer architecture
along with utility modules for training and evaluation.


## Installation

The packages follow the `src` structure format and need to be installed in a python environment.
It is recommended to use a clean environment using `venv` or `conda`. All experiments were conducted
with Python `3.10`.


### Create and activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`
```

### Install Packages

For each package, navigate to the package directories' top-level and install the package in development mode.

```bash
cd ./packages/mesh_sim
pip install -e .
```

```bash
cd ./packages/mesh_opformer
pip install -e .
```

## Configuration

The configuration files control dataset paths, data preprocessing, model architecture, training, and evaluation parameters. Results are stored using the `wandb` library. If you have a `wandb` account, you can enable it by configuring the following sections in your config file:

```toml
[wandb]
WANDB_MODE = "online"     # Options: 'online', 'offline', 'disabled'
name = "hodgeformer-shrec11"

[wandb.init]
project = "project-name"
entity = "user-wandb-account"
```


## Execution Examples

All commands should be executed from the `experiments/training` directory for training scripts and `experiments/inference` directory for inference scripts.

### Cube Engraving - Classification

**Training:**

```bash
cd experiments/training
python classification_cube_engraving.py --cfg_path ../cfg/classification_cube_engraving_cfg.toml --out ../runs/runs
```

**Inference:**

```bash
cd experiments/inference;
python infer_hodgeformer.py \
--model ../runs/runs/classification_cube_engraving_ckpt_300.pth \
--cfg_path ../cfg/classification_cube_engraving_cfg.toml \
--out ./out.json \
--dataset_path ../data/cube_engraving/cubes/fork/test
```

### SHREC11 - Classification

**Training:**

```bash
cd experiments/training
python classification_shrec.py --cfg_path ../cfg/classification_shrec11_cfg.toml --out ../runs/runs
```

**Inference:**

```bash
cd experiments/inference;
python infer_hodgeformer.py \
--model ../runs/runs/classification_shrec11_ckpt_300.pth \
--cfg_path ../cfg/classification_shrec11_cfg.toml \
--out ./out.json \
--dataset_path ../data/classification_shrec_11/data/simplified/raw/alien/test
```

### COSEG - Segmentation

#### COSEG Aliens

**Training:**

```bash
cd experiments/training
python segmentation_coseg.py --cfg_path ../cfg/segmentation_coseg_aliens.toml --out ../runs/runs
```
**Inference:**

```bash
cd experiments/inference;
python infer_hodgeformer.py \
--model ../runs/runs/segmentation_coseg_hodgeformer-coseg-aliensckpt_200.pth \
--cfg_path ../cfg/segmentation_coseg_aliens.toml \
--out ./out.json \
--dataset_path ../data/coseg/coseg-hanocka_et_al_2019/coseg_aliens/test
```


#### COSEG Chairs

**Training:**

```bash
cd experiments/training
python segmentation_coseg.py --cfg_path ../cfg/segmentation_coseg_chairs.toml --out ../runs/runs
```


**Inference:**

```bash
cd experiments/inference;
python infer_hodgeformer.py \
--model ../runs/runs/segmentation_coseg_hodgeformer-coseg-chairsckpt_200.pth \
--cfg_path ../cfg/segmentation_coseg_chairs.toml \
--out ./out.json \
--dataset_path ../data/coseg/coseg-hanocka_et_al_2019/coseg_chairs/test
```


#### COSEG Vases

**Training:**

```bash
cd experiments/training
python segmentation_coseg.py --cfg_path ../cfg/segmentation_coseg_vases.toml --out ../runs/runs
```

**Inference:**

```bash
cd experiments/inference;
python infer_hodgeformer.py \
--model ../runs/runs/segmentation_coseg_hodgeformer-coseg-vasesckpt_300.pth \
--cfg_path ../cfg/segmentation_coseg_vases.toml \
--out ./out.json \
--dataset_path ../data/coseg/coseg-hanocka_et_al_2019/coseg_vases/test
```


### Human Segmentation

**Training:**

```bash
cd experiments/training
python segmentation_human_simplified.py --cfg_path ../cfg/segmentation_human_simplified_cfg.toml --out ../runs/runs
```

**Inference:**

```bash
cd experiments/inference;
python infer_hodgeformer.py \
--model ../runs/runs/hodgeformer_human_simplified-id_ioywp1l4-lr_0.0005.pth \
--cfg_path ../cfg/segmentation_human_simplified_cfg.toml \
--out ./out.json \
--dataset_path ../data/human_benchmark_sig_17/pd_meshnet20_seg_benchmark/test
```


## Citation 

```
Nousias, A. and Nousias, S., 2025. HodgeFormer: Transformers for Learnable Operators on Triangular Meshes through Data-Driven Hodge Matrices. arXiv preprint arXiv:2509.01839.
```

```
@article{nousias2025hodgeformer,
  title={HodgeFormer: Transformers for Learnable Operators on Triangular Meshes through Data-Driven Hodge Matrices},
  author={Nousias, Akis and Nousias, Stavros},
  journal={arXiv preprint arXiv:2509.01839},
  year={2025}
}
```