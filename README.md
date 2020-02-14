> # FQ-GAN
>

This repository contains source code necessary to reproduce the results presented in the paper Feature Quantization Improves GAN Training

[TOC]



# FQ-BigGAN
## Dependencies
> This code is based on [PyTorchGAN](https://github.com/ajbrock/BigGAN-PyTorch). Here we will give more details of the code usage. Basically, you will need 
>
> **python 3.x, pytorch 1.x, tqdm ,h5py**

## Prepare datasets
1. CIFAR-10 (change C10 to C100 to prepare CIFAR-100)
```
	python make_hdf5.py --dataset C10 --batch_size 256 --data_root data
	python calculate_inception_moments.py --dataset C10 --data_root data --batch_size 128
```
2. ImageNet, first you need to manually download ImageNet dataset, then execute the following command to prepare ImageNet (128x128)

```
	python make_hdf5.py --dataset I128 --batch_size 256 --data_root data
	python calculate_inception_moments.py --dataset I128_hdf5 --data_root data --batch_size 128

```


## Training 
We put four bash scripts in  FQ-BigGAN/scripts to train CIFAR-10, CIFAR-100, ImageNet (64x64) and ImageNet (128x128). For example, to train CIFAR-100, simply execute

```
sh scripts/launch_C100.sh
```

To modify the FQ parameters, we provide the following options in each script as arguments:

1. ```--discrete_layer```: it  specifies which layers you want quantization to be added, i.e. 0123 
2. ```--commitment``` : it is the commitment coefficient, default=1.0
3. ```--dict_size```:  the size of the EMA dictionary, default=8, meaning there are 2^8 keys in the dictionary.
4. ```--dict_decay```:  the momentum when learning the dictionary, default=0.8.

