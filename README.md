# CtS: Complete to Segment
Official Code for "Boosting Multi-Modal Unsupervised Domain Adaptation for LiDAR Semantic Segmentation by Self-Supervised Depth Completion", published at IEEE Access 2023. Authors: Adriano Cardace, Andrea Conti, Pierluigi Zama Ramirez, Riccardo Spezialetti, Samuele Salti, Luigi Di Stefano


## Preparation

1) Install conda or mamba (mamba is faster)
2) Create environment with the command  ``` mamba env create -f requirements.yaml ```
3) Activate with ``` conda activate cts ``


## Datasets
Please, follow the instruction from [XMUDA](https://github.com/valeoai/xmuda).
Note that differently from XMUDA, we require 3D points to be exxpressed in the camera reference frame. For this reason, we modified their preprocessing steps accordingly. As example on how to do that on Nuscenes, look at `lib/dataset/preprocessing_nuscenes.py`.


## Training & Testing

To launch an experiment you can use the following code:

`CUDA_VISIBLE_DEVICES=0 python experiments_day_night/cts/run.py`

Then, to test a model, change the defaults.run to test in `experiments_day_night/cts/config/config.yaml`