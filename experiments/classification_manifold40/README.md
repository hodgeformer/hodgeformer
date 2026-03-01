# Mesh Classification on the *Manifold40* dataset

### Description

The *Manifold40* dataset is a processed variation of the original *ModelNet40* dataset introduced by Wu et al. 2015.
The original dataset contains 12,311 shapes in 40 categories and is a widely used benchmark for 3D geometric learning. 

Manifold40 was created by the authors of *SubDivNet* to handle issues with 3D shapes in ModelNet40 that were not
watertight or 2-manifold, and that lead to remeshing failures. The reconstructed shapes in Manifold40 were all closed 
manifolds and simplified to 500 faces. 


### Data

The dataset (with no remeshing or other operations) is provided by the *SubDivNet* authors in the following link:
- `https://cg.cs.tsinghua.edu.cn/dataset/subdivnet/datasets/Manifold40.zip`

Download it and extract it to the `./data` folder. 


### Training & Evaluation

To train and evaluate a Hodgeformer model on the Manifold40 dataset use the training script and config file.
Before running, check that the dataset paths are correct and configure accordingly the `[wandb]` and `[wandb.init]` 
fields of the configuration file to add your `wandb` credentials. Then run:

```bash
python classification_manifold40.py --cfg_path ./classification_manifold40_cfg.toml --out ./runs
```
