#!/usr/bin/env bash
set -x
DATAPATH="/home/xyguo/data/kitti_2015/"
python save_disp.py --datapath $DATAPATH --testlist ./filenames/kitti15_test.txt --model gwcnet-g --loadckpt ./checkpoints/kitti15/gwcnet-g/best.ckpt
