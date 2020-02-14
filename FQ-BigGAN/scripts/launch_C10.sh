#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0,1
python3 train.py --shuffle --batch_size 64 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 \
--D_lr 2e-4 --dataset C10 --G_ortho 0.0 \
--G_attn 0 --D_attn 0 --G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 1000 --save_every 1000 \
--num_best_copies 5 --num_save_copies 2 --seed 0 \
--discrete_layer 0123 --commitment 1.0 --dict_size 6 --dict_decay 0.9 \
--name_suffix quant
