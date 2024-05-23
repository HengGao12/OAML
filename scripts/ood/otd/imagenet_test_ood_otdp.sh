#!/bin/bash
# sh scripts/ood/she/cifar100_test_ood_she.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood
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
    --config configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_ood.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/otdp.yml \
    --num_workers 4 \
    --ood_dataset.image_size 256 \
    --dataset.test.batch_size 256 \
    --dataset.val.batch_size 256 \
    --network.pretrained True \
    --network.checkpoint 'results/pretrained_weights/resnet50_imagenet1k_v1.pth' \
    --merge_option merge


#python main.py \
#    --config configs/datasets/cifar100/cifar100_224x224.yml \
#    configs/datasets/cifar100/cifar100_ood.yml \
#    configs/networks/resnet18_224x224.yml \
#    configs/pipelines/test/test_ood.yml \
#    configs/preprocessors/base_preprocessor.yml \
#    configs/postprocessors/otdp.yml \
#    --network.checkpoint 'results/pretrained_weights/resnet50_imagenet1k_v1.pth'
