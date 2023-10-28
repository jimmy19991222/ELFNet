#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py --epochs 16\
                --batch_size 1\
                --checkpoint pretrain\
                --pre_train\
                --num_workers 2\
                --dataset sceneflow\
                --dataset_directory /path_to_sceneflow/\
                --weight_reg 0.5\
                --name elfnet_pretrain\
                --lr_sttr 0.0002\
                --lr_backbone 0.0002\
                --lr_regression 0.0004\
                --lr_pcw 0.002\
                --lrepochs "4,8,10,12"

# sh scripts/elfnet_pretrain.sh |tee logs/elfnet_pretrain.log