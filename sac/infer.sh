#!/bin/sh
python3 real_infer.py \
--device cuda \
--cp_path best_model_833326.pt \
--env_id Real-monopod-simple-v1 \
--num_eval_episodes 10 \
