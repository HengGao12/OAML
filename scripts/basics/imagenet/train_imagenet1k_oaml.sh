export CUDA_VISIBLE_DEVICES="5"
GPU=4
CPU=1
node=73
jobname=ood

PYTHONPATH='.'$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \
# -w SG-IDC1-10-51-2-${node} \
python main.py --config configs/datasets/imagenet/imagenet.yml configs/datasets2/imagenet/imagenet.yml configs/preprocessors/base_preprocessor.yml configs/networks/resnet50.yml configs/pipelines/train/generative_ood_distill_pipeline.yml --seed 0