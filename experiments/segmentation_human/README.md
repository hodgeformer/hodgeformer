# Human: Mesh Segmentation

### Description

The Human dataset (a.k.a Human Body dataset, and Human-part segmentation dataset) consists
of 381 training samples and 18 test samples of meshes of human figures in various poses.
It was originally published by Maron et al. 2017, as a mashup of different smaller datasets
(Adobe, FAUST, SCAPE, MIT Animation and SHREC) for the paper *Convolutional Neural Networks on Surfaces via Seamless Toric Covers*.
The meshes in the original dataset had a resolution ranging between ~13K and 20K faces.  

A simplified version with a resolution of 1500 faces per mesh was provided by Hanocka et al. 2019,
the authors of *MeshCNN*, but with ground-truth segmentation labels on the mesh edges.

The dataset version used in this set of experiments is provided by Milano et al. 2020, 
the authors of *PD-MeshNet*, and is a processed variant where the ground-truth labels on 
the edges were mapped to ground-truth labels on the faces.

### Data

The dataset is provided by the *PD-MeshNet* authors in the following dropbox link:
- `https://www.dropbox.com/s/byk8oisbm75g5yb/human_seg.zip`

Download it and extract it to the `./data` folder. 

Alternatively, from this folder path (`experiments/segmentation_human`) run the following bash commands:
```bash
DATADIR="data" 
 
echo "downloading the data and putting it in: " $DATADIR
mkdir -p $DATADIR
wget https://www.dropbox.com/s/byk8oisbm75g5yb/human_seg.zip
unzip human_seg.zip -d $DATADIR && rm human_seg.zip
```

### Training & Evaluation

To train and evaluate a Hodgeformer model on the Human dataset use the corresponding training 
script and config file. Before running, check that the dataset paths are correct and configure
accordingly the `[wandb]` and `[wandb.init]` fields of the configuration file to add your `wandb`
credentials. Then run:

```bash
python segmentation_human_simplified.py --cfg_path segmentation_human_simplified_cfg.toml --out ./runs
```

### Inference

To perform inference on a dataset subset use the inference entrypoint `infer-hodgeformer` (or the 
provided inference script). Below is an example for running a trained model on a dataset subset:

```bash
infer-hodgeformer \
    --model ./runs/<path-to-model>.pth \
    --cfg_path ./cfg/segmentation_human_simplified_cfg.toml \
    --dataset_path ./data/human_seg/test
    --out ./out.json \
```
