# Mesh Segmentation on the *COSEG* dataset

### Description

The COSEG dataset contains meshes with segmentation labels belonging to three categories,
for each of which a different experiment is performed:
* Category *aliens*, consisting of 198 samples with 4 class labels
* Category *chairs*, that contains 397 samples with 3 class labels
* Category *vases*, made of 297 samples  with 4 class labels

The used version is provided by Milano et al. 2020, the authors of *PD-MeshNet*, and is
a processed version of the dataset version provided by Hanocka et al. 2019,the authors of
*MeshCNN* so as to convert the original ground-truth labels on the edge to ground-truth
labels on the faces.

The dataset is pre-split in train and test sets in an 85/15 ratio. In our evaluation, in
each training run we shuffle and resplit the dataset. The results are averaged over five
runs. 

### Data

The dataset is provided by the *PD-MeshNet* authors in the following dropbox link:
- `https://www.dropbox.com/s/zro2o7y42wag5yp/coseg.zip`

Download it and extract it to the `./data` folder. 

Alternatively, from this folder path (`experiments/segmentation_coseg`) run the following bash commands:
```bash
DATADIR="data" 
 
echo "downloading the data and putting it in: " $DATADIR
mkdir -p $DATADIR
wget https://www.dropbox.com/s/zro2o7y42wag5yp/coseg.zip
unzip coseg.zip -d $DATADIR && rm coseg.zip
```

### Training & Evaluation

To train and evaluate a Hodgeformer model on a category of the COSEG dataset, use the 
corresponding training script along the corresponding configuration file. Before running, 
check that the dataset paths are correct and configure accordingly the `[wandb]` and 
`[wandb.init]`  fields of the configuration file to add your `wandb` credentials. 

For *COSEG Aliens* run:
```bash
python segmentation_coseg.py --cfg_path segmentation_coseg_aliens.toml --out ./runs
```

For *COSEG Chairs* run:
```bash
python segmentation_coseg.py --cfg_path segmentation_coseg_chairs.toml --out ./runs
```

For *COSEG Vases* run:
```bash
python segmentation_coseg.py --cfg_path segmentation_coseg_vases.toml --out ./runs
```

**Inference:**

To perform inference on a dataset subset use the inference entrypoint `infer-hodgeformer` 
(or the provided inference script). Below is an example for running a trained model on a 
subset of the aliens category:

```bash
infer-hodgeformer \
    --model ./runs/<path-to-model>.pth \
    --cfg_path ./cfg/segmentation_coseg_aliens.toml \
    --dataset_path ./data/coseg_aliens/test
    --out ./out.json \
```
