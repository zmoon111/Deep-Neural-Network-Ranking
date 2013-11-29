#!/bin/bash

#python ../train-ranknet/convnet.py --data-path=/mnt/iscsi_3/yuefeng/quality_for_wangmeng/batches_list_dir  --test-range=100 --train-range=1-99 --layer-def=./conv_ranknet_cfg/imagenet.full.cfg  --layer-params=./conv_ranknet_cfg/params-init.full.1.cfg --data-provider=pair --test-freq=5 --gpu=2 --epochs=1000 --save-path=./model.20131105 --crop-border=16 

python ../train-ranknet/convnet.py -f ./model.20131105/27.26 --layer-params=./conv_ranknet_cfg/params-init.full.3.cfg  --gpu=3 --save-path=./model.20131105/
#python ../train-ranknet/convnet.py -f ./model.image-quality.20130704/model.image-quality.20130704/199.1 --save-path=./model.image-quality.20130704 --epochs=300
