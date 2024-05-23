#!/bin/bash
# sh scripts/ood/udg/cifar10_test_udg.sh
export CUDA_VISIBLE_DEVICES='3'
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
    configs/networks/udg_net.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/gen.yml \
    --num_workers 8 \
    --network.backbone.name resnet18_224x224 \
    --network.checkpoint '/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_oe_udg_udg_e100_lr0.1_default/s0/best_epoch91_acc0.8520.ckpt' \
    --mark 0
# /public/home/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_oe_udg_udg_e50_lr0.1_default/s0/best.ckpt
############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
# python scripts/eval_ood.py \
#    --id-data cifar10 \
#    --root ./results/cifar10_oe_udg_udg_e100_lr0.1_default \
#    --postprocessor ebo \
#    --save-score --save-csv
