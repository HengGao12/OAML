#!/bin/bash
# sh scripts/ood/logitnorm/cifar10_test_logitnorm.sh

export CUDA_VISIBLE_DEVICES='0'
PYTHONPATH='.':$PYTHONPATH \
python main.py \
    --config configs/datasets/cifar10/cifar10_224x224.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --num_workers 8 \
    --network.checkpoint '/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_resnet18_224x224_logitnorm_e100_lr0.1_alpha0.04_default/s0/best.ckpt' \
    --mark 0 \
