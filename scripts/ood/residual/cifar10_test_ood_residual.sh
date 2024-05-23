#!/bin/bash
# sh scripts/ood/residual/cifar10_test_ood_residual.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood
export CUDA_VISIBLE_DEVICES='6'
PYTHONPATH='.':$PYTHONPATH \
# srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \

python main.py \
--config configs/datasets/cifar10/cifar10_224x224.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/resnet18_224x224.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/residual.yml \
--num_workers 8 \
--network.checkpoint '/home1/gaoheng/gh_workspace/openood-main/results/cifar100_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt' \
--mark 0
