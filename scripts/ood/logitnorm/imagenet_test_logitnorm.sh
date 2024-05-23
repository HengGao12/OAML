#!/bin/bash
# sh scripts/ood/logitnorm/imagenet_test_logitnorm.sh

############################################
# we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood_imagenet.py
export CUDA_VISIBLE_DEVICES='0'
# available architectures:
# resnet50
# ood
python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/train/train_vos.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/knn.yml \
    --network.pretrained True \
    --network.checkpoint ./results/imagenet_resnet50_logitnorm_e30_lr0.001_alpha0.04_default/s0/best.ckpt \
    --feature_dim 2048 \
    --optimizer.lr 0.001 \
    --optimizer.num_epochs 30 \
    --dataset.train.batch_size 128 \
    --num_gpus 1 --num_workers 16 \
    --merge_option merge
# python scripts/eval_ood_imagenet.py \
#   --ckpt-path ./results/imagenet_resnet50_logitnorm_e30_lr0.001_alpha0.04_default/s0/best.ckpt \
#   --arch resnet50 \
#   --postprocessor ebo \
#   --save-score --save-csv #--fsood


# full-spectrum ood
# python scripts/eval_ood_imagenet.py \
#   --ckpt-path ./results/imagenet_resnet50_logitnorm_e30_lr0.001_alpha0.04_default/s0/best.ckpt \
#   --arch resnet50 \
#   --postprocessor msp \
#   --save-score --save-csv --fsood
