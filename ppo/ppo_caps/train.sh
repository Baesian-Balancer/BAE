#!/bin/bash
python3 ppo.py \
--epochs=200 \
--load_model_path=exp/2022_04_01_08_17_08/ \
--distribution_type='gaussian' \
--lam_ent=-1e-3 \
--lam_ts=2e-5 \
--lam_sps=1e-4 \
--hid=96 \
--lam_mdmu=-.1 \
--lam_a=-1e-3 \
--eps_s=0.001 \
--lam_fft=-1e-2 \
--lam_rp=-1e-2
