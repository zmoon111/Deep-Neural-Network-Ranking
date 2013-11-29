#!/bin/bash

python ../train-ranknet/convnet.py --data-path=/mnt/iscsi_1/yuefeng/quality_pair_for_blur/batches  --test-range=50 --train-range=1-49 --layer-def=./conv_ranknet_cfg/imagenet.full.cfg  --layer-params=./conv_ranknet_cfg/params-init.full.cfg --data-provider=pair --test-freq=5 --gpu=0 --epochs=1000 --save-path=./model.20130810 --crop-border=16 

#python ../train-ranknet/convnet.py --data-path=/mnt/iscsi_3/yuefeng/quality_pair_for_blur/batches -f model.20130820/70.9 --layer-params=./conv_ranknet_cfg/params-init.full.2.cfg  --gpu=2 --save-path=./model.20130909
#python ../train-ranknet/convnet.py -f model.20130820/6.5 --layer-params=./conv_ranknet_cfg/params-init.full.2.cfg  --gpu=0 --save-path=./model.20130906
#python ../train-ranknet/convnet.py -f ./model.image-quality.20130704/model.image-quality.20130704/199.1 --save-path=./model.image-quality.20130704 --epochs=300
