#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py  --epochs 1\
                --batch_size 1\
                --checkpoint test\
                --num_workers 2\
                --dataset sceneflow\
                --dataset_directory /path_to_sceneflow/\
                --name elfnet_test_sceneflow\
                --resume ./ckpt/elfnet_pretrain.tar\
                --eval\
                --validation validation_all

# sh scripts/elfnet_test_sceneflow.sh |tee logs/elfnet_test_sceneflow.log