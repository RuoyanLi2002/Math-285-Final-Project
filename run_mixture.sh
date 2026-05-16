#!/bin/bash

export CUDA_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

python main.py \
    --exp_name "exp/mixture/run0" \
    --seed 0 \
    --dataset_root "/local2/shared_data/us_crsp_nyse" \
    --data_save_path "" \
    --seq_length 20 \
    --split_interval 11 \
    --config "config/mixture.yaml"