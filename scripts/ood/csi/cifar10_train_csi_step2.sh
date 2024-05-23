#!/bin/bash
# sh scripts/ood/csi/cifar10_train_csi_step2.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood
export CUDA_VISIBLE_DEVICES='3'
PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \
# -w SG-IDC1-10-51-2-${node} \

SEED=0
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/csi_net.yml \
    configs/pipelines/train/train_csi.yml \
    configs/preprocessors/csi_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint /home1/gaoheng/gh_workspace/openood-main/results/cifar10_csi_net_csi_step1_e100_lr0.1/s0/best.ckpt \
    --optimizer.num_epochs 100 \
    --dataset.train.batch_size 128 \
    --mode csi_step2 \
    --seed ${SEED}
