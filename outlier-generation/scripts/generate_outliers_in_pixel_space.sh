export CUDA_VISIBLE_DEVICES='1'
python /home1/gaoheng/gh_workspace/dream-ood-main/dream_ood.py --plms \
    --n_iter 50 --n_samples 3 \
    --outdir /home1/gaoheng/gh_workspace/dream-ood-main/nobackup-fast/txt2img-samples-in1k-demo/ \
    --loaded_embedding /home1/gaoheng/gh_workspace/dream-ood-main/cifar100_outlier_npos_embed_noise_0.07_select_50_KNN_300.npy \
    --ckpt /home1/gaoheng/gh_workspace/dream-ood-main/nobackup-slow/dataset/my_xfdu/diffusion/sd-v1-4.ckpt \
    --id_data cifar100 \
    --skip_grid
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/in1k_outlier_npos_embed_noise_0.07_select_50_KNN_300.npy
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/cifar10_outlier_npos_embed_noise_0.07_select_50_KNN_300.npy
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/in1k_outlier_npos_embed_noise_0.07_select_50_KNN_300.npy