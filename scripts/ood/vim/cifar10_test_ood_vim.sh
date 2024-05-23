#!/bin/bash
# sh scripts/ood/vim/cifar10_test_ood_vim.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood
export CUDA_VISIBLE_DEVICES='0'

PYTHONPATH='.':$PYTHONPATH \
# srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \

python main.py --config configs/datasets/cifar10/cifar10_224x224.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/vim.yml \
    --num_workers 8 \
    --network.checkpoint '/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_resnet18_224x224_base_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_e500/s0/best_epoch469_acc0.9550.ckpt' \
    --mark 0 \
    --postprocessor.postprocessor_args.dim 256
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_resnet18_224x224_base_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_e500/s0/best_epoch469_acc0.9550.ckpt
# /home1/hezhuolin/9_gh_workspace/OpenOOD-main/OpenOOD-main/results/cifar10_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt
############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
# python scripts/eval_ood.py \
#    --id-data cifar10 \
#    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
#    --postprocessor vim \
#    --save-score --save-csv
