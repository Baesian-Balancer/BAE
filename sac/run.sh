#!/bin/sh
python3 train.py \
--device cpu \
--episode_eval_frequency 3 \
--num_train_steps 5000000 \
--exploration_steps 100000 \
--save_cp 
# --wandb_on \
# --multi_step 3 \
