# Faster R-CNN MAD 描述
目标检测模型在生活中实际使用时，会因为部署的领域的不同而造成模型的实际性能下降。为了减轻这种领域偏移造成的影响，有很多现有的方法利用领域对抗学习（Domain Adversarial Learning，DAL）来解耦特征中的领域不变部分和领域变化部分，并指导网络学习其中的领域不变（公共）特征。然而，过去的方法忽略了领域公共特征中的潜在非因果因素，一方面因为有标注源域是有限的，导致其中的共有部分可能非因果因素，另一方面DAL方法中单个领域分类器往往更关注有利于领域分类的显著非因果因素，而会忽略潜在的非因果因素的学习。我们基于生活中对事物观察“横看成林侧成峰”的启发，提出了通过多个视角观察源域特征，把源域特征映射到不同的潜在特征空间（视角），在不同的特征空间中非显著的特则将被该视角的领域判别器识别并指导特征提取器进一步剔除非显著的非因果特征。我们提出的基于多视角对抗判别器（MAD）的领域泛化模型包含两个模块，分别是通过随机增强来扩充源域多样性的假相关生成器（SCG）和将特征映射到多个潜在空间的多视图域分类器（MVDC），通过这两个模块能够更好的提出非因果因素，从而得到更加纯净的领域不变特征。此外，我们在六个常用的基准数据集上做了广泛的实验表明，我们的MAD算法在目标检测任务上有着最佳的泛化性能。

# 数据集

使用的数据集：[COCO 2017](<https://cocodataset.org/>)

- 数据集大小：19G
    - 训练集：18G，118,000个图像  
    - 验证集：1G，5000个图像
    - 标注集：241M，实例，字幕，person_keypoints等
- 数据格式：图像和json文件
    - 注意：数据在dataset.py中处理。

# 环境要求

- 硬件（Ascend）

- 下载数据集COCO 2017。

- 在 ModelArts 进行训练

    ```python
    # 在 ModelArts 上使用单卡训练
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "distribute=True"
    #          在 default_config.yaml 文件中设置 "data_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "epoch_size: 20"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "distribute=True"
    #          在网页上设置 "data_path=/cache/data"
    #          在网页上设置 "epoch_size: 20"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，请上传你的预训练模型到桶上
    # (4) 上传原始数据集到桶上。
    # (5) 在网页上设置你的代码路径为 "/path/faster_rcnn"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"coco_root"、"output_path"、"config_path=default_config_ascend.yaml"等
    # (8) 创建训练作业
    #
    # 在 ModelArts notbook上验证
    # (1) 给环境中的numpy进行降级到1.16.0，否则验证时报错
    # (2) 导入obs中的相关数据集和代码
        #		import moxing as mox
    #		mox.file.copy_parallel(data_obs_url, data_local_url)
    # (3) 设置好test_sd.sh中的相关参数，并在Terminal中执行sh test_sd.sh
    # (4) 在./eval路径中查看训练的结果文件
    ```

开始训练后每个100step会打印相关loss的值
```
step:  100 	 loss:  1.9138364791870117
step:  200 	 loss:  0.9567146301269531
step:  300 	 loss:  1.5999830961227417
step:  400 	 loss:  1.691389799118042
step:  500 	 loss:  1.2009999752044678
step:  600 	 loss:  2.0246150493621826
step:  700 	 loss:  2.338803291320801
```

测试结束后会展示相关精度
```
Accumulating evaluation results...
DONE (t=15.17s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.303
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.518
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.320
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.178
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.332
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.391
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.274
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.438
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.461
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.294
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.499
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.583
summary_metrics: 
{'Precision/mAP': 0.30298552890307656, 'Precision/mAP@.50IOU': 0.517589029644264, 'Precision/mAP@.75IOU': 0.3197388645299048, 'Precision/mAP (small)': 0.17803261788080235, 'Precision/mAP (medium)': 0.33244364714731667, 'Precision/mAP (large)': 0.3912777887265828, 'Recall/AR@1': 0.27404838298404033, 'Recall/AR@10': 0.43756098598828175, 'Recall/AR@100': 0.46128473583657886, 'Recall/AR@100 (small)': 0.2938816722112075, 'Recall/AR@100 (medium)': 0.4988041376076137, 'Recall/AR@100 (large)': 0.5831302372399844}

Evaluation done!

Done!
Time taken: 840.13 seconds
```

联系 xmjtf@outlook.com

