#!/bin/bash
# sh scripts/ood/mixoe/cifar100_train_mixoe.sh
export CUDA_VISIBLE_DEVICES='0'

SEED=0
python main.py \
    --config configs/datasets/cifar100/cifar100_224x224.yml \
    configs/datasets/cifar100/cifar100_oe.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/train_mixoe.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint ./results/cifar100_resnet18_224x224_base_e100_lr0.1_default/s${SEED}/best.ckpt \
    --optimizer.lr 0.001 \
    --optimizer.num_epochs 5 \
    --dataset.train.batch_size 128 \
    --dataset.oe.batch_size 128 \
    --seed ${SEED}
