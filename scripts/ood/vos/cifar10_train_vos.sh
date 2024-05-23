#!/bin/bash
# sh scripts/ood/vos/cifar10_train_vos.sh
export CUDA_VISIBLE_DEVICES="7"
PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
    --config configs/datasets/cifar10/cifar10_224x224.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/train_vos.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --num_workers 8 \
    --optimizer.num_epochs 100 --seed 0
    # --merge_option merge \

