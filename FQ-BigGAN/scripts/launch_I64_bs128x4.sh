#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2
python train.py \
--dataset I64_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 128 --load_in_mem  \
--num_G_accumulations 4 --num_D_accumulations 4 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 32 --D_attn 32 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch 64 --D_ch 64 \
--ema --use_ema --ema_start 20000 \
--test_every 1000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--discrete_layer 2 --commitment 0.5 --dict_size 10 --dict_decay 0.7 \
--use_multiepoch_sampler --name_suffix test