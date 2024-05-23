#!/bin/bash
# sh scripts/ood/cider/cifar100_train_cider.sh
export CUDA_VISIBLE_DEVICES='4'

python main.py \
    --config configs/datasets/cifar100/cifar100_224x224.yml \
    configs/networks/cider_net.yml \
    configs/pipelines/train/train_cider.yml \
    configs/preprocessors/base_preprocessor.yml \
    --preprocessor.name cider \
    --network.backbone.name resnet18_224x224 \
    --dataset.train.batch_size 512 \
    --trainer.trainer_args.proto_m 0.5 \
    --num_workers 8 \
    --optimizer.num_epochs 100 \
    --seed 0
