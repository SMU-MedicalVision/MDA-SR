# MDA-SR
This repository is an official PyTorch implementation of the paper **"MDA-SR: Multi-level Domain Adaptation Super-Resolution for Wireless Capsule Endoscopy Images"**

## 🔥: Dependencies
* Python 3.7
* PyTorch >= 1.7.0
* matplotlib
* yaml
* importlib
* functools
* scipy
* numpy
* tqdm
* PIL

In this project, we propose a multi-level domain adaptation training framework for the SR of capsule endoscopy images.

## 🚉: Pre-Trained Models

To achieve SR of capsule endoscopy images, download these [2x](https://drive.google.com/drive/folders/1jfKLOgH47ZqiCgX6bkcBnEFkhvKraRoa?usp=sharing), [4x](https://drive.google.com/drive/folders/1URuR8R3C3Gp6PWXzyKkpwdejwD2h3Yp3?usp=sharing) models, and place them in the path `./experiment/ckpt`
## 🚋: Training

We first train adaptive downsampling model alone for 50 epochs, and then train domain adaptation SR model together for 50 epoch.
The detailed training command as here:
```
CUDA_VISIBLE_DEVICE=0 python train.py --name {EXP_PATH} --scale {SCALE} --adv_w 0.01 --batch_size 10 --patch_size_down 256 --decay_batch_size_sr 400000 --decay_batch_size_down 50000 --epochs_sr_start 51 --gpu cuda:0 --sr_model endosr --training_type endosr --joint --save_results --save_log
```
with following options:
- `EXP_PATH` is the folder name of experiment results
- `scale` is the scale of the SR
- `adv_w` is th hyperparameter. (default: `0.01)

## 🧩: Evaluation

The detailed evaluation command as here:
**test sr: generation of super-resolution images**
```
CUDA_VISIBLE_DEVICE=0 python predict.py --test_mode sr_patch --name scale_4x --scale 4 --crop 480 --pretrain_sr ./experiment/ckpt/scale_4x/model_sr_0110.pth --test_lr Capsule_Data/TestSet/side_480 --gpu cuda:6 --sr_model endosr --training_type endosr --save_results --realsr
```
**test down: generation of low-resolution images**
```
CUDA_VISIBLE_DEVICE=0 python predict.py --test_mode down --name down_x4 --scale 4 --resume_down ./experiment/ckpt/scale_4x/model_down_0110.pth --patch_size_down 512 --test_range 1-2000 --gpu cuda:6
```