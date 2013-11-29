#!/bin/bash

#python ../train-ranknet/convnet.py --data-path=/mnt/iscsi_3/yuefeng/quality_for_clarity/batches_dir/  --test-range=240 --train-range=1-239 --layer-def=./conv_ranknet_cfg/imagenet.full.multi.size.cfg  --layer-params=./conv_ranknet_cfg/params-init.multi.size.cfg --data-provider=pair --test-freq=5 --gpu=1 --epochs=1000 --save-path=./model.20131110 --crop-border=16 

python ../train-ranknet/convnet.py -f model.20131110/2.106 --layer-params=./conv_ranknet_cfg/params-init.multi.size.cfg  --gpu=1 --save-path=./model.20131110
#python ../train-ranknet/convnet.py -f ./model.image-quality.20130704/model.image-quality.20130704/199.1 --save-path=./model.image-quality.20130704 --epochs=300
