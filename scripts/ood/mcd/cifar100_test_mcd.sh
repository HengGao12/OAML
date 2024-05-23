#!/bin/bash
# sh scripts/ood/mcd/cifar100_test_mcd.sh

# NOTE!!!!
# need to manually change the checkpoint path
# remember to use the last_*.ckpt because mcd only trains for the last 10 epochs
# and the best.ckpt (according to accuracy) is typically not within the last 10 epochs
# therefore using best.ckpt is equivalent to early stopping with standard cross-entropy loss

python main.py \
    --config configs/datasets/cifar100/cifar100_224x224.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/mcd_net.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/test/test_ood.yml \
    configs/postprocessors/knn.yml \
    --network.backbone.name resnet18_224x224 \
    --network.pretrained True \
    --network.checkpoint '/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_oe_mcd_mcd_e100_lr0.1_default/s0/best.ckpt' \
    --num_workers 8 \
    --seed 0
