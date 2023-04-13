# Multi-view Adversarial Discriminator: Mine the Non-causal Factors for Object Detection in Unseen Domains

本代码为论文Multi-view Adversarial Discriminator: Mine the Non-causal Factors for Object Detection in Unseen Domains的配套代码。[**[Paper]**](https://arxiv.org/abs/2304.02950)

## 摘要
目标检测模型在生活中实际使用时，会因为部署的领域的不同而造成模型的实际性能下降。为了减轻这种领域偏移造成的影响，有很多现有的方法利用领域对抗学习（Domain Adversarial Learning，DAL）来解耦特征中的领域不变部分和领域变化部分，并指导网络学习其中的领域不变（公共）特征。然而，过去的方法忽略了领域公共特征中的潜在非因果因素，一方面因为有标注源域是有限的，导致其中的共有部分可能非因果因素，另一方面DAL方法中单个领域分类器往往更关注有利于领域分类的显著非因果因素，而会忽略潜在的非因果因素的学习。我们基于生活中对事物观察“横看成林侧成峰”的启发，提出了通过多个视角观察源域特征，把源域特征映射到不同的潜在特征空间（视角），在不同的特征空间中非显著的特则将被该视角的领域判别器识别并指导特征提取器进一步剔除非显著的非因果特征。我们提出的基于多视角对抗判别器（MAD）的领域泛化模型包含两个模块，分别是通过随机增强来扩充源域多样性的假相关生成器（SCG）和将特征映射到多个潜在空间的多视图域分类器（MVDC），通过这两个模块能够更好的提出非因果因素，从而得到更加纯净的领域不变特征。此外，我们在六个常用的基准数据集上做了广泛的实验表明，我们的MAD算法在目标检测任务上有着最佳的泛化性能。

## 本代码由两个版本
1. Mindspore 版本  
   使用华为Mindspore架构，在昇腾910计算卡进行训练和测试，数据集为COCO 2017。
2. Pytorch 版本  
   使用Pytorch 0.4.0版本，在Nvidia TITAN XP上进行训练和测试。

## 引用
```
@article{xu2023multi,
  title={Multi-view Adversarial Discriminator: Mine the Non-causal Factors for Object Detection in Unseen Domains},
  author={Xu, Mingjun and Qin, Lingyun and Chen, Weijie and Pu, Shiliang and Zhang, Lei},
  journal={arXiv preprint arXiv:2304.02950},
  year={2023}
}
```