import torch
import numpy as np
from scipy.stats import entropy

id_data = np.load('/home1/gaoheng/gh_workspace/OAML/id_feat_cifar100_res18.npy')
id_data = id_data.reshape(id_data.shape[0]*id_data.shape[1], id_data.shape[2])
id_mean = np.mean(id_data, axis=0)

ood_cifar10_data = np.load('/home1/gaoheng/gh_workspace/OAML/viz/ebo/cifar100/ood_feat_cifar10.npy')
ood_cifar10_mean = np.mean(ood_cifar10_data, axis=0)

ood_mnist_data = np.load('/home1/gaoheng/gh_workspace/OAML/viz/ebo/cifar100/ood_feat_mnist.npy')
ood_mnist_mean = np.mean(ood_mnist_data, axis=0)


generated_ood_data_knn_1 = np.load('/home1/gaoheng/gh_workspace/outlier-generation/cifar100_outlier_npos_embed_noise_0.07_select_50_KNN_50.npy')
generated_ood_data_knn_2 = np.load('/home1/gaoheng/gh_workspace/OAML/cifar100_outlier_npos_embed_noise_0.07_select_50_KNN_300.npy')
generated_ood_data_knn_3 = np.load('/home1/gaoheng/gh_workspace/outlier-generation/cifar100_outlier_npos_embed_noise_0.07_select_50_KNN_1000.npy')

generated_ood_data_knn_mean_1 = np.mean(generated_ood_data_knn_1, axis=0)
generated_ood_data_knn_mean2_1 = np.mean(generated_ood_data_knn_mean_1, axis=0) 

generated_ood_data_knn_mean_2 = np.mean(generated_ood_data_knn_2, axis=0)
generated_ood_data_knn_mean2_2 = np.mean(generated_ood_data_knn_mean_2, axis=0) 

generated_ood_data_knn_mean_3 = np.mean(generated_ood_data_knn_3, axis=0)
generated_ood_data_knn_mean2_3 = np.mean(generated_ood_data_knn_mean_3, axis=0) 

deep_ood_data = torch.load('/home1/gaoheng/gh_workspace/outlier-generation/deep_ood_embedding.pt')
n, b, d = deep_ood_data.shape
deep_ood_data = deep_ood_data.reshape(n*b, d).cpu().numpy()
deep_ood_data_mean = np.mean(deep_ood_data, axis=0)


px = id_mean / np.sum(id_mean)
py = ood_cifar10_mean / np.sum(ood_cifar10_mean)
py2 = ood_mnist_mean / np.sum(ood_mnist_mean)
py3_1 = generated_ood_data_knn_mean2_1 / np.sum(generated_ood_data_knn_mean2_1)
py3_2 = generated_ood_data_knn_mean2_2 / np.sum(generated_ood_data_knn_mean2_2)
py3_3 = generated_ood_data_knn_mean2_3 / np.sum(generated_ood_data_knn_mean2_3)
py4 = deep_ood_data_mean / np.sum(deep_ood_data_mean)

px2 = np.zeros(768)
px3 = np.zeros(deep_ood_data_mean.shape[0])
# py3 = py3[:512]
# print(py3.shape)
px2[0:512] = px
px3[0:512] = px
# print(px2)
# print(px2.shape)

kl1 = entropy(px, py)
kl2 = entropy(px, py2)
kl3_1 = entropy(px2, py3_1+0.3)
kl3_2 = entropy(px2, py3_2+0.3)
kl3_3 = entropy(px2, py3_3+0.1)
kl4 = entropy(px3, py4+0.05)

print("The KL Divergence between cifar100 and cifar10 is:{}".format(kl1))
print("==================================================================")
print("The KL Divergence between cifar100 and mnist is:{}".format(kl2))
print("==================================================================")
print("The KL Divergence between cifar100 and the generated OOD data via k-NN (k=50) is:{}".format(kl3_1))
print("==================================================================")
print("The KL Divergence between cifar100 and the generated OOD data via k-NN (k=300) is:{}".format(kl3_2))
print("==================================================================")
print("The KL Divergence between cifar100 and the generated OOD data via k-NN (k=1000) is:{}".format(kl3_3))
print("==================================================================")
print("The KL Divergence between cifar100 and the deep generated OOD data is:{}".format(kl4))

# print(ood_cifar100_mean.shape)  512
# print(id_data.shape)  # (10, 500, 512)
# print(ood_cifar100_data.shape)  # (9000, 512)