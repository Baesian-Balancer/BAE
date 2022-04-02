#!/bin/bash
python3 ppo.py \
--epochs=200 \
--lam_ent=-1e-3 \
--lam_ts=1e-5 \
--lam_sps=5e-5 \
--hid=96 \
--lam_mdmu=-.1 \
--lam_a=-1e-3 \
--eps_s=0.001 \
--lam_fft=-1e-2 \
--lam_rp=-1e-2 \
# --load_model_path="exp/2022_04_01_08_17_08/best_model_step_3279999.pt" \
--load_model_path="exp/2022_04_02_05_37_14/best_model_step_3259999.pt" \
--distribution_type="gaussian"
