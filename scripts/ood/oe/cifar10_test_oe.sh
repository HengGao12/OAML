#!/bin/bash
# sh scripts/ood/oe/cifar10_test_oe.sh
export CUDA_VISIBLE_DEVICES='5'
GPU=1
CPU=1
node=63
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
    --config configs/datasets/cifar10/cifar10_224x224.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --num_workers 8 \
    --network.checkpoint '/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_oe_resnet18_224x224_oe_e100_lr0.1_lam0.5_default/s0/best.ckpt' \
    --mark 0 --merge_option merge

# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt
# /public/home/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_oe_resnet18_224x224_oe_e100_lr0.1_lam0.5_default/s0/best.ckpt
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_oe_resnet18_224x224_oe_e100_lr0.1_lam0.5_default/s0/best.ckpt
############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
# python scripts/eval_ood.py \
#    --id-data cifar10 \
#    --root ./results/cifar10_oe_resnet18_32x32_oe_e100_lr0.1_lam0.5_default \
#    --postprocessor msp \
#    --save-score --save-csv
