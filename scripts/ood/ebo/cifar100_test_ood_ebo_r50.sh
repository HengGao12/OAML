#!/bin/bash
export CUDA_VISIBLE_DEVICES='7'
PYTHONPATH='.':$PYTHONPATH \

python main.py \
    --config configs/datasets/cifar100/cifar100_224x224.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --network.checkpoint 'results/cifar100_resnet50_base_generative_ood_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_0.2dcl_0.0001mi_loss_0.0001ml_0.0001dml_0.01dt-for-rebuttal-1/s0/best.ckpt' --merge_option merge
# r50 oaml /home1/gaoheng/gh_workspace/OAML/results/cifar100_resnet50_base_generative_ood_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_0.2dcl_0.0001mi_loss_0.0001ml_0.0001dml_0.01dt-for-rebuttal-1/s0/best.ckpt
# results/cifar100_resnet50_base_e100_lr0.1_default/s0/best.ckpt