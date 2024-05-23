export CUDA_VISIBLE_DEVICES='5'

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/cifar100/cifar100_224x224.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/distill_postprocessor.yml \
    --num_workers 8 \
    --network.checkpoint '/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt' \
    --mark 0

# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_dev4_2kl_4fd/s0/best_epoch361_acc0.8090.ckpt