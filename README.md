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
### Pretrained Model
* Pretrained ASNet: `pretrained_models/ASNet.pth` (Place at `models/asnet/` or `models/asnet_1gpu` for evaluation)
### Dataset Preparation
* Download the dataset
* Place the dataset under folder `data/`, alternatively, indicate 'data_dir' in the config file `models/<config_dir>/train.yaml`.
* Make sure that all the images are in `images/` and all the json files are in `labels/`.
### Training
* Please take note that our results are obtained on 8 GPU, with batch_size = 512. 
```
python train.py --config_dir asnet
```
* On a single GPU, you can try the following with batch_size = 64.
```
python train.py --config_dir asnet_1gpu
```
### Evaluating Appearance Feature Model
* Here, the appearance model refers to any neural networks trained, such as ASNet and TripletNet. You can change the json file to evaluate different data splits.
```
python test.py --config_dir asnet \
--eval_json test.json \
--save_features \
--eval_model
```

### Evaluating Appearance Feature Model with Epipolar Soft Constraint
```
python test.py --config_dir asnet \
--eval_json test.json \
--load_features \
--eval_model_esc 
```
### Evaluating by Angle Differences
```
python test.py --config_dir asnet \
--eval_json test.json \
--load_features \
--eval_by_angle 
```
