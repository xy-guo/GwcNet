#!/usr/bin/env bash
set -x
DATAPATH="/home/xyguo/data/kitti_2012/"
python save_disp.py --datapath $DATAPATH --testlist ./filenames/kitti12_test.txt --model gwcnet-gc --loadckpt ./checkpoints/kitti12/gwcnet-gc/best.ckpt
