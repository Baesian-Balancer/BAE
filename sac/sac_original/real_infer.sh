#!/bin/sh
python3 real_infer.py \
--device cpu \
--cp_path best_model_3595722.pt \
--env_id Real-monopod-hop-v1 \
--num_eval_episodes 10 \
