#!/usr/bin/env bash
set -x
DATAPATH="/home/xyguo/data/kitti_2012/"
python main.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitti12_train.txt --testlist ./filenames/kitti12_val.txt \
    --epochs 300 --lrepochs "200:10" \
    --model gwcnet-gc --logdir ./checkpoints/kitti12/gwcnet-gc --loadckpt ./checkpoints/sceneflow/gwcnet-gc/pretrained.ckpt \
    --test_batch_size 1