#!/bin/bash
# sh scripts/ood/npos/cifar100_test_npos.sh

export CUDA_VISIBLE_DEVICES='6'

python main.py --config configs/datasets/cifar100/cifar100_224x224.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --num_workers 8 \
    --network.checkpoint '/home1/gaoheng/gh_workspace/OAML/results/cifar100_npos_net_npos_e100_lr0.1_default/s0/best.ckpt' \
    --mark 0
    

# /home1/gaoheng/gh_workspace/OAML/results/cifar100_npos_net_npos_e100_lr0.1_default/s0/best.ckpt

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
# python scripts/eval_ood.py \
#    --id-data cifar100 \
#    --root ./results/cifar100_npos_net_npos_e100_lr0.1_default \
#    --postprocessor npos \
#    --save-score --save-csv
