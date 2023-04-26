## MAD的Pytorch版本

## 运行环境
- Python 3.6
- Pytorch 0.4.0
- CUDA 8.0

## 数据集
数据集采用VOC的格式存放  
  ```
  ├──./VOC2007/
  │    ├── JPEGImages
  │    │    ├── PASCAL VOC提供的所有的图片，其中包括训练图片，测试图片。
  │    ├── Annotations
  │    │    ├── 存放xml格式的标签文件，每个xml对应JPEGImage中的一张图片。
  │    ├── ImageSets
  │    │    ├── Main
  │    │    |   ├── train.txt  
  │    │    |   ├── test.txt
```

## 程序运行
1. SCG数据增强
    ```
    1. 修改SCG.py中的数据集地址：
        img_path = './cityscape_s1/VOC2007/JPEGImages'     # 原始数据集
        save_path = './cityscape_s2/VOC2007/JPEGImages'    # 增强数据集

    2. 运行 python SCG.py
    ```
    已经生成好的数据集存放在https://pan.baidu.com/s/1XTVRXdUIbru3IzKGXY-OhQ 密码bezm
2. MAD模型训练
    ```
    CUDA_VISIBLE_DEVICES=0 python train_MV_3.py \
        --dataset       dg_union \
        --net           vgg16 \
        --cuda          \
        --epochs        10 \
        --bs            1 \
        --save_dir      ./SaveFile/model \
        --Mission       "Your_Task_Name" \
        --mode          train_model \
        --log_flag      1 \
        --lr            2e-3 \
        --lr_decay_step 6 \
        \
        --T_Set         foggy \
        --T_Part        test \
        --T_Type        s1 \
        \
        --S1_Set        cityscape \
        --S1_Part       train \
        --S1_Type       s1 \
        \
        --S2_Set        cityscape \
        --S2_Part       train \
        --S2_Type       s2 \
        \

    ```
3. 测试
    ```
    CUDA_VISIBLE_DEVICES=0 python test.py \
        --net           vgg16 \
        --cuda          \
        --model_dir     "save_model_path" \
        --dataset       dg_union \
        \
        --T_Set         rain \
        --T_Part        test \
        --T_Type        s1 \
        \
        --S1_Set        cityscape \
        --S1_Part       train \
        --S1_Type       s1 \
        \
        --S2_Set        foggy \
        --S2_Part       train \
        --S2_Type       s1 \
    ```

## 参考结果
Citysacpes --> Foggy Cityscapes

|          | bike   | bus    | car    | motor  | person | rider  | train  | truck  | Mean   |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| AP     | 0.3901 | 0.4698 | 0.4514 | 0.3148 | 0.3390 | 0.4627 | 0.3885 | 0.2842 | 0.3876 |
