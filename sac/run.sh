#!/bin/sh
python3 train.py \
--device cpu \
--episode_eval_frequency 5 \
--num_train_steps 5000000 \
--exploration_steps 20000 \
--wandb_on \
--save_cp 