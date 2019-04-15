#!/usr/bin/env bash
set -x
DATAPATH="/home/xyguo/data/kitti_2015/"
python main.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitti15_train.txt --testlist ./filenames/kitti15_val.txt \
    --epochs 300 --lrepochs "200:10" \
    --model gwcnet-g --logdir ./checkpoints/kitti15/gwcnet-g --loadckpt ./checkpoints/sceneflow/gwcnet-g/pretrained.ckpt \
    --test_batch_size 1