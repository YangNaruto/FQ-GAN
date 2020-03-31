#!/bin/bash
# export CUDA_VISIBLE_DEVICES=3
python make_hdf5.py --dataset C100 --batch_size 256 --data_root data
python calculate_inception_moments.py --dataset C100 --data_root data --batch_size 128
