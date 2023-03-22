#!/bin/bash
# 统一训练脚本

echo "testing program start!"

CUDA_VISIBLE_DEVICES=0

if [ -d "./eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cd ./eval || exit

echo "start eval for device $CUDA_VISIBLE_DEVICES"
env > env.log
pwd

python ../eval.py \
    --config_path="/media/xmj/DATA/ms_pro/FasterRCNN/default_config.yaml" \
    --coco_root=/media/xmj/DATA/ms_pro/DataSet/coco \
    --mindrecord_dir=/media/xmj/DATA/ms_pro/DataSet/coco/MindRecord_COCO_TRAIN/ \
    --device_target="GPU" \
    --anno_path="/media/xmj/DATA/ms_pro/DataSet/coco/annotations/instances_val2017.json" \
    --checkpoint_path="/media/xmj/DATA/ms_pro/FasterRCNN/fasterrcnn_resnetv150_ascend_v190_coco2017_official_cv_AP50acc60.5.ckpt" \
    --backbone=resnet_v1.5_50 &> eval.log &