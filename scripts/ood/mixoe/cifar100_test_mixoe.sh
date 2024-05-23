#!/bin/bash
# sh scripts/ood/mixoe/cifar100_test_mixoe.sh
export CUDA_VISBLE_DEVICES='5'

python main.py \
    --config configs/datasets/cifar100/cifar100_224x224.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/gen.yml \
    --num_workers 8 \
    --network.checkpoint '/public/home/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_oe_resnet18_224x224_mixoe_e5_lr0.001_alpha0.1_beta1.0_cutmix_lam1.0_default/s0/best.ckpt' \
    --mark 0 --merge_option merge

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
# python scripts/eval_ood.py \
#    --id-data cifar100 \
#    --root ./results/cifar100_oe_resnet18_224x224_mixoe_e10_lr0.001_alpha0.1_beta1.0_cutmix_lam1.0_default \
#    --postprocessor ebo \
#    --save-score --save-csv
