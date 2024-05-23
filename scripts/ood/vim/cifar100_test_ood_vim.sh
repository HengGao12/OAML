#!/bin/bash
# sh scripts/ood/vim/cifar100_test_ood_vim.sh
export CUDA_VISIBLE_DEVICES="3"
GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \

python main.py \
    --config configs/datasets/cifar100/cifar100_224x224.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/vim.yml \
    --num_workers 8 \
    --network.checkpoint '/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt' \
    --mark 0 \
    --postprocessor.postprocessor_args.dim 256
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_dev4_2kl_4fd/s0/best_epoch361_acc0.8090.ckpt
############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
# python scripts/eval_ood.py \
#    --id-data cifar100 \
#    --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
#    --postprocessor vim \
#    --save-score --save-csv
