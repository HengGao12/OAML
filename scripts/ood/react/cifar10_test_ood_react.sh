#!/bin/bash
# sh scripts/ood/react/cifar10_test_ood_react.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood
export CUDA_VISIBLE_DEVICES='0'
PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
    --config configs/datasets/cifar10/cifar10_224x224.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/react_net.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/react.yml \
    --network.pretrained False \
    --network.backbone.name resnet18_224x224 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint '/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_resnet18_224x224_base_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_e500/s0/best_epoch469_acc0.9550.ckpt' \
    --num_workers 8 \
    --mark 0

# configs/networks/react_net.yml
# /home1/hezhuolin/9_gh_workspace/OpenOOD-main/OpenOOD-main/results/pytorch_model.bin
# /home1/hezhuolin/9_gh_workspace/OpenOOD-main/OpenOOD-main/results/cifar10_resnet18_224x224_distill_trainer_e100_lr0.01_default_w_fea_n_logit_dist/s0/best.ckpt/home1/hezhuolin/9_gh_workspace/OpenOOD-main/OpenOOD-main/results/cifar10_resnet18_224x224_distill_trainer_e100_lr0.01_default_w_fea_n_logit_dist/s0/best.ckpt
# /home1/hezhuolin/9_gh_workspace/OpenOOD-main/OpenOOD-main/results/cifar10_resnet18_224x224_distill_trainer_e100_lr0.01_default_w_fea_dist/s0/best.ckpt
# /home1/hezhuolin/9_gh_workspace/OpenOOD-main/OpenOOD-main/results/cifar10_resnet18_224x224_distill_trainer_e100_lr0.01_default_w_logit_dist/s0/best.ckpt
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
#    --postprocessor react \
#    --save-score --save-csv
