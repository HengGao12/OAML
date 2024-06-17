import torch
import numpy as np
from scipy.stats import entropy

id_data = np.load('/home1/gaoheng/gh_workspace/OAML/id_feat_cifar10_res18.npy')
id_data = id_data.reshape(id_data.shape[0]*id_data.shape[1], id_data.shape[2])
id_mean = np.mean(id_data, axis=0)
# print(id_mean.shape)  512
ood_cifar100_data = np.load('/home1/gaoheng/gh_workspace/OAML/viz/ebo/cifar10/ood_feat_cifar100.npy')
ood_cifar100_mean = np.mean(ood_cifar100_data, axis=0)
ood_mnist_data = np.load('/home1/gaoheng/gh_workspace/OAML/viz/ebo/cifar10/ood_feat_mnist.npy')
ood_mnist_mean = np.mean(ood_mnist_data, axis=0)
generated_ood_data_knn = np.load('/home1/gaoheng/gh_workspace/OAML/cifar10_outlier_npos_embed_noise_0.07_select_50_KNN_300.npy')
# print("gh debug, the shape of knn generated ood data:{}".format(generated_ood_data_knn.shape))
# generated_ood_data_knn = np.reshape(10000, generated_ood_data_knn[2])
generated_ood_data_knn_mean = np.mean(generated_ood_data_knn, axis=0)
generated_ood_data_knn_mean2 = np.mean(generated_ood_data_knn_mean, axis=0) # 768
# print("gh debug, the shape of knn generated ood data mean:{}".format(generated_ood_data_knn_mean2.shape))
deep_ood_data = torch.load('/home1/gaoheng/gh_workspace/dream-ood-main/deep_ood_embedding_cifar10.pt')   # (50, 3, 16384)
n, b, d = deep_ood_data.shape
deep_ood_data = deep_ood_data.reshape(n*b, d).cpu().numpy()
deep_ood_data_mean = np.mean(deep_ood_data, axis=0)
print(deep_ood_data_mean.shape)

px = id_mean / np.sum(id_mean)
py = ood_cifar100_mean / np.sum(ood_cifar100_mean)
py2 = ood_mnist_mean / np.sum(ood_mnist_mean)
py3 = generated_ood_data_knn_mean2 / np.sum(generated_ood_data_knn_mean2)
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
kl3 = entropy(px2, py3+0.05)
kl4 = entropy(px3, py4+0.05)

print("The KL Divergence between cifar10 and cifar100 is:{}".format(kl1))
print("==================================================================")
print("The KL Divergence between cifar10 and mnist is:{}".format(kl2))
print("==================================================================")
print("The KL Divergence between cifar10 and the generated OOD data is:{}".format(kl3))
print("==================================================================")
print("The KL Divergence between cifar10 and the deep generated OOD data is:{}".format(kl4))

# print(ood_cifar100_mean.shape)  512
# print(id_data.shape)  # (10, 500, 512)
# print(ood_cifar100_data.shape)  # (9000, 512)