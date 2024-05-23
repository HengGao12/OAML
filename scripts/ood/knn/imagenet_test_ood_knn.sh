#!/bin/bash
# sh scripts/ood/knn/imagenet_test_ood_knn.sh
export CUDA_VISIBLE_DEVICES='0'
GPU=1
CPU=1
node=37
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
     configs/postprocessors/knn.yml \
     --num_workers 4 \
     --ood_dataset.image_size 256 \
     --dataset.test.batch_size 256 \
     --dataset.val.batch_size 256 \
     --network.pretrained True \
     --network.checkpoint './results/imagenet_resnet50_logitnorm_e30_lr0.001_alpha0.04_default/s0/best.ckpt' \
     --merge_option merge

# ./results/imagenet_resnet50_logitnorm_e30_lr0.001_alpha0.04_default/s0/best.ckpt 

# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_ood_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_for_ablation/s0/best.ckpt
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_ood_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_for_ablation/s0/best.ckpt
# results/pretrained_weights/resnet50_imagenet1k_v1.pth
# gd: /public/home/gaoheng/gh_workspace/GOLDEN_HOOP/results/imagenet_resnet50_base_generative_ood_distill_trainer_e50_lr5e-05_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_0.2dcl_0.0001mi_loss_0.0001ml_0.0001dml_0.01dt/s0/best.ckpt
# /public/home/gaoheng/gh_workspace/GOLDEN_HOOP/results/imagenet_resnet50_base_generative_ood_distill_trainer_e50_lr5e-05_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_0.2dcl_0.0001mi_loss_0.0001ml_0.0001dml_0.01dt/s0/best.ckpt
############################################
# we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood_imagenet.py

# available architectures:
# resnet50, swin-t, vit-b-16
# ood
#python scripts/eval_ood_imagenet_foundation_models.py \
#   --postprocessor msp \
#   --arch ViT-L/14 \
#   --save-score --save-csv #--fsood

# # full-spectrum ood
# python scripts/eval_ood_imagenet.py \
#    --tvs-pretrained \
#    --arch resnet50 \
#    --postprocessor knn \
#    --save-score --save-csv --fsood
