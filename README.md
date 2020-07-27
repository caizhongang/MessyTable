# **messytable**
> This repository is for **MessyTable: Instance Association in Multiple Camera Views** accepted on ECCV 2020.

## Paper


## Dataset Page
[MessyTable dataset](https://caizhongang.github.io/projects/MessyTable/)


## Environment
```
(ananconda3, python 3.7.3)
conda create -n mt
conda activate mt
pip install torch==1.1.0 torchvision==0.3.0
pip install opencv-python==3.4.2.17
pip install scipy==1.2.0
```
## Get started
### Pretrained Models
* Pretrained resnet18: `pretrained_models/model.pth' (Place at src/ for training)
* Pretrained ASNet: `pretrained_models/model.pth' (Place at models/asnet/ for evaluation)
### Dataset Preparation
* Download the dataset
* Place the dataset under folder `data/`, alternatively, indicate 'data_dir' in the config file `models/<config_dir>/train.yaml`.
* Make sure that all the images are in `images/` and all the json files are in `labels/`.
### Training
* In the slurm environment, with 8 GPUs
```
srun -u --partition=<partition_name> -n1 --gres=gpu:8 --ntasks-per-node=1 \
    python mt_train.py \
    --config_dir asnet
```
* Alternatively
```
python mt_train.py --config_dir asnet
```
### Evaluating Appearance Feature Model
* Here, the appearance model refers to any neural networks trained, such as ASNet and TripletNet.
```
srun -u --partition=<partition_name> -n1 --gres=gpu:1 --ntasks-per-node=1 \
    python test.py --config_dir asnet \
    --eval_json <eval_data_split>.json \
    --load_features false \
    --save_features true \
    --eval_model true 
```

### Evaluating Appearance Feature Model with Epipolar Soft Constraint
```
srun -u --partition=<partition_name> -n1 --gres=gpu:1 --ntasks-per-node=1 \
    python test.py --config_dir asnet \
    --eval_json <eval_data_split>.json \
    --load_features true \
    --eval_model_esc true 
```
### Evaluating by Angle Differences
```
srun -u --partition=<partition_name> -n1 --gres=gpu:1 --ntasks-per-node=1 \
    python test.py --config_dir asnet \
    --eval_json <eval_data_split>.json \
    --load_features true \
    --eval_by_angle true 
```
