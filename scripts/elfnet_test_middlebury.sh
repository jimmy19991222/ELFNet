#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py  --epochs 1\
                --batch_size 1\
                --checkpoint test\
                --num_workers 2\
                --dataset middlebury\
                --dataset_directory /path_to_middlebury/trainingQ/\
                --name elfnet_test_middlebury\
                --resume ./ckpt/elfnet_pretrain.tar\
                --eval\
                --validation validation_all

# sh scripts/elfnet_test_middlebury.sh |tee logs/elfnet_test_middlebury.log