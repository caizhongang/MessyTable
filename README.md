# **MessyTable: Instance Association in Multiple Camera Views**

Useful Links:
* Visit our [[Project Homepage]](https://caizhongang.github.io/projects/MessyTable/) for an overview of MessyTable dataset
* Read our Paper (Accepted in ECCV 2020) [[Preprint]](https://arxiv.org/pdf/2007.14878.pdf) for complete technical details

## Setup
### Environment
```
(ananconda3)
conda create -n mt python=3.7
conda activate mt
pip install torch==1.1.0 torchvision==0.3.0
pip install opencv-python==3.4.2.17
pip install scipy==1.2.0
```

### Dataset Preparation
* Download MessyTable.zip (~22 GB) from [[Aliyun]](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/MessyTable.zip) or [[Google Drive]](https://drive.google.com/file/d/1i4mJz9xsDwhzWes7sVLXuhLKP9eNtbBG/view?usp=sharing)
* Unzip MessyTable.zip, check the unzipped folder includes `images/` and `labels/`
* Rename the unzipped folder to `data/`, place `data/` in this repository as follows:
```
MessyTable
├── models
├── src
├── data
    ├── images
    ├── labels
```

### Pretrained Model
* Download pretrained ASNet (ASNet.pth) from [[Google Drive]](https://drive.google.com/file/d/1VMKYeUSlUpnwLRdtDtygpkYpVWgGAm46/view?usp=sharing)
* Place model in `models/asnet/` for evaluation

## Get started
### Evaluation
This example evaluates pretrained ASNet: 
```
python test.py --config_dir asnet \
--eval_json test.json \
--save_features \
--eval_model
```
Arguments:
* --config_dir: the directory that contains the specific config file `train.yaml` (checkpoints are automatically saved in the same dir)
* --eval_json: data split name in `data/labels/` to evaluate (test.json, val.json and by difficulty levels, see Paper Sec 5.3)
* --save_features: (optional) save extrated features in `models/<config_dir>` for faster evaluation in the future
* --load_features: (optional) load saved features from `models/<config_dir>`, if the features have been saved in the past
* --eval_model: evaluate using the appearance features only
* --eval_model_esc: evaluate using the appearance features with epipolar soft constraint (See Paper Sec 5.2)
* --eval_by_angle: (optional) evaluate by angle differences (See Paper Sec 5.3)

### Training
* Training on 8 GPUs with batch size 512 (we use this setting in our paper)
```
python train.py --config_dir asnet
```
* Training on a single GPU with batch size 64
```
python train.py --config_dir asnet_1gpu
```
## Citation
If you find this repo or dataset useful, please consider citing our paper
```bibtex
@inproceedings{
    CaiZhang2020MessyTable,
    title={MessyTable: Instance Association in Multiple Camera Views},
    author={Zhongang Cai and Junzhe Zhang and Daxuan Ren and Cunjun Yu and Haiyu Zhao and Shuai Yi and Chai Kiat Yeo and Chen Change Loy},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    month = {August},
    year={2020}
}
```

