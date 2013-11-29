#!/bin/bash

python ../train-ranknet/convnet.py --data-path=/mnt/iscsi_3/yuefeng/quality_for_wangmeng/batches_list_dir_0.9  --test-range=20 --train-range=1-19 --layer-def=./conv_ranknet_cfg/imagenet.full.cfg  --layer-params=./conv_ranknet_cfg/params-init.full.cfg --data-provider=pair --test-freq=5 --gpu=0 --epochs=1000 --save-path=./model.20131106_0.9 --crop-border=16 

#python ../train-ranknet/convnet.py -f model.20130816/model.20130810 --layer-params=./example-layers/params-init.full.1.cfg  --gpu=0 --save-path=./model.20130820
#python ../train-ranknet/convnet.py -f ./model.image-quality.20130704/model.image-quality.20130704/199.1 --save-path=./model.image-quality.20130704 --epochs=300
