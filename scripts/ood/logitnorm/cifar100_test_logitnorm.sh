#!/bin/bash
# sh scripts/ood/logitnorm/cifar100_test_logitnorm.sh
export CUDA_VISIBLE_DEVICES='0'
PYTHONPATH='.':$PYTHONPATH \
python main.py \
    --config configs/datasets/cifar100/cifar100_224x224.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/gen.yml \
    --num_workers 8 \
    --network.checkpoint '/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_logitnorm_e100_lr0.1_alpha0.04_default/s0/best.ckpt' \
    --mark 0 \

# ############################################
# # alternatively, we recommend using the
# # new unified, easy-to-use evaluator with
# # the example script scripts/eval_ood.py
# # especially if you want to get results from
# # multiple runs
# python scripts/eval_ood.py \
#    --id-data cifar100 \
#    --root ./results/cifar100_resnet18_32x32_logitnorm_e100_lr0.1_alpha0.04_default \
#    --postprocessor msp \
#    --save-score --save-csv
