# OAML: Outlier Aware Metric Learning for OOD Detection Enhancement

## Overview

In this work, we propose an **Outlier Aware Metric Learning (OAML)** framework for generating OOD data in a distribution free manner and learning effectively, which improves OOD detection performance of different kinds of score functions on various benchemark to a great extent.

![image](fig/pipeline.jpg)

## Usage

### Installation

```sh
conda create -n openood python=3.8
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
wget https://codeload.github.com/Vastlab/libMR/zip/refs/heads/master
cd python
pip install -r requirements.txt
cd ../
pip install .
cd ../
git clone https://github.com/Jingkang50/OpenOOD.git
cd OpenOOD
pip install -e .
pip install timm
```

In order to better adapt to the OpenOOD framework, we changed the  `vision_transformer.py`  in the timm library as follows:

```python
...
    def forward(self, x, return_feature):
        x = self.forward_features(x)
        x, pre_logits = self.forward_head(x)
        if return_feature:
            return x, pre_logits  
        else:
            return x
   
    def get_fc(self):
        fc = self.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.head
...
```

### Data Preparation

Our codebase accesses the datasets from `./data/` and pretrained models from `./results/checkpoints/` . One can download the datasets via running  `./scripts/download/download.py`.

```
├── ...
├── data
│   ├── benchmark_imglist
│   ├── images_classic
│   └── images_largescale
├── openood
├── results
│   ├── checkpoints
│   └── ...
├── scripts
├── main.py
├── ...
```
### Training and Testing

##### Training

```sh
# Train with OAML on CIAFR-10
bash scripts/basics/cifar10/train_cifar10_oaml.sh
# Train with OAML on CIFAR-100
bash scripts/basics/cifar100/train_cifar100_oaml.sh
# Train with OAML on ImageNet-1k
bash scripts/basics/imagenet/train_imagenet1k_oaml.sh
```

##### Testing

```sh
# Test on CIFAR-10 with EBO score
bash scripts/ood/cifar10_test_ood_ebo.sh
```

## Model Zoo

Here we provide pretrained ResNet-18 on CIFAR-10/100 and ImageNet-1k.

| ID Dataset  |                 Vanilla                  |             Train with OAML              |
| :---------: | :--------------------------------------: | :--------------------------------------: |
|  CIFAR-10   | [link](https://onedrive.live.com/?id=F86DF442193FFBCB%21111&cid=F86DF442193FFBCB) | [link](https://onedrive.live.com/?id=F86DF442193FFBCB%21115&cid=F86DF442193FFBCB) |
|  CIFAR-100  | [link](https://onedrive.live.com/?id=F86DF442193FFBCB%21118&cid=F86DF442193FFBCB) | [link](https://onedrive.live.com/?id=F86DF442193FFBCB%21121&cid=F86DF442193FFBCB) |
| ImageNet-1k | [link](https://onedrive.live.com/?id=F86DF442193FFBCB%21123&cid=F86DF442193FFBCB) | [link](https://onedrive.live.com/?id=F86DF442193FFBCB%21125&cid=F86DF442193FFBCB) |



## Acknowledgments

OAML is developed based on [OpenOOD](https://github.com/Jingkang50/OpenOOD/tree/main) repo. Thanks to their great work.
