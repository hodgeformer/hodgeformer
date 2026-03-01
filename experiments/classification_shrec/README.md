# Mesh Classification on the *SHREC-11* dataset

### Description

SHREC is an annual international challenge on evaluating and comparing the effectiveness of 3D-shape 
retrieval algorithm. The *SHREC-11* dataset comes from the 2011 challenge. The dataset includes meshes
divided across 30 categories, with 20 meshes for each category, for a total of 600 meshes. 

In past work, the dataset is used for evaluating mesh classification models in different splits, with
the *split-16* and *split-10* being traditionally used. For the *split-16*, for each class 16 samples 
are used for training and 4 for testing, whereas for the *split 10*, for each class 10 samples are used
for training and 10 for testing. In this work we limited our results only to the (more difficult) *split-10*.

The SHREC11 dataset version used here is the lower-resolution version provided by Hanocka et. al, 2019.
The simplified version consists of watertight meshes with 750 edges and 500 faces each.

### Data

The dataset is provided by the *MeshCNN* authors in the following dropbox link:
- `https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz`

Note, that despite the filename `shrec_16`, this is indeed the shapes from the SHREC 2011 dataset in their
simplified version. Extract it to the data/simplified/raw/ directory.

Download it and extract it to the `./data` folder. 

Alternatively, from this folder path (`experiments/classification_shrec`) run the following bash commands:
```bash
DATADIR="data" 
 
echo "downloading the data and putting it in: " $DATADIR
mkdir -p $DATADIR
wget https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz
tar -xzvf shrec_16.tar.gz -C $DATADIR && rm shrec_16.tar.gz
```

### Training & Evaluation

To train and evaluate a Hodgeformer model on the SHREC-11 dataset use the corresponding training script and config file.
Before running, check that the dataset paths are correct and configure accordingly the `[wandb]` and `[wandb.init]` 
fields of the configuration file to add your `wandb` credentials. Then run:

```bash
python classification_shrec.py --cfg_path ./classification_shrec11_cfg.toml --out ./runs
```

This dataset is pre-split in training and testing sets, but for our evaluation we ignore the splits following the original 
dataset. On each training run, a new random train/test split is generated. The split can be controlled by the `random_state`
parameter in the configuration file:
```toml
[dataset.split]
train_size = 0.50
valid_size = 0.0
test_size = 0.50
random_state = 123
```

### Inference

To perform inference on a dataset subset use the inference entrypoint `infer-hodgeformer` (or the provided inference script):

```bash
infer-hodgeformer \
    --model ./runs/<model-name>.pth \
    --cfg_path ./classification_shrec11_cfg.toml \
    --dataset_path ../data/shrec16/dinosaur/test \
    --out ./out.json \
```