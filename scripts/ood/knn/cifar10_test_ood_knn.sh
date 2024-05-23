#!/bin/bash
# sh scripts/ood/knn/cifar10_test_ood_knn.sh

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
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/knn.yml \
    --num_workers 8 \
    --network.checkpoint '/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_resnet18_224x224_base_generative_ood_distill_trainer_e500_lr0.001_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_0.2dcl_0.001mi_loss_0.001ml_0.001dml/s0/best_epoch387_acc0.9530.ckpt' \
    --mark 0

# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_resnet18_224x224_base_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_e500/s0/best_epoch469_acc0.9550.ckpt
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_resnet18_224x224_base_generative_ood_distill_trainer_e500_lr0.001_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_0.2dcl_0.001mi_loss_0.001ml_0.001dml/s0/best_epoch387_acc0.9530.ckpt
# /home1/gaoheng/gh_workspace/openood-main/results/cifar10_resnet18_224x224_distill_trainer_e100_lr0.01_default_w_fea_n_logit_dist/s0/best.ckpt
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_resnet18_224x224_base_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_e500/s0/best_epoch469_acc0.9550.ckpt
############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
# python scripts/eval_ood.py \
#    --id-data cifar10 \
#    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
#    --postprocessor knn \
#    --save-score --save-csv
