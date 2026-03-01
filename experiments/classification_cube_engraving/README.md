# Mesh Classification on the *Cube Engraving* dataset

### Description

The *Cube Engraving* dataset, introduced by Hanocka et al., 2019, in "MeshCNN: A Network with an Edge" is a synthetic
dataset of 2D shapes that have been engraved on a randomly chosen location (position, rotation and face) of a cube. 
The final dataset consists of 4,381 shapes from 22 distinct categories, pre-split in training and test sets. 

The dataset is used for the task of mesh classification.

### Data

The dataset is provided by the *MeshCNN* authors in the following dropbox link:
- `https://www.dropbox.com/s/2bxs5f9g60wa0wr/cubes.tar.gz`

Download it and extract it to the `./data` folder. 

Alternatively, from this folder path (`experiments/classification_cube_engraving`) run the following bash commands:
```bash
DATADIR="data" 
 
echo "downloading the data and putting it in: " $DATADIR
mkdir -p $DATADIR
wget https://www.dropbox.com/s/2bxs5f9g60wa0wr/cubes.tar.gz
tar -xzvf cubes.tar.gz -C $DATADIR && rm cubes.tar.gz
```

### Training & Evaluation

To train and evaluate a Hodgeformer model on the Cube engraving dataset use the corresponding training script and config file.
Before running, check that the dataset paths are correct and configure accordingly the `[wandb]` and `[wandb.init]` 
fields of the configuration file to add your `wandb` credentials. Then run:

```bash
python classification_cube_engraving.py --cfg_path ./classification_cube_engraving_cfg.toml --out ./runs
```

### Inference

To perform inference on a dataset subset (e.g. the *forks* category from the cubes engraving dataset), use the inference entry
point `infer-hodgeformer` (or the provided inference script):

```bash
infer-hodgeformer \
    --model ./runs/<model-name>.pth \
    --cfg_path ./classification_cube_engraving_cfg.toml \
    --dataset_path ../data/cubes/fork/test \
    --out ./out.json \
```