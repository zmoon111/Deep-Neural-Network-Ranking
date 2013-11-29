#!/bin/bash

python ../train-ranknet/convnet.py --data-path=/mnt/iscsi_3/yuefeng/quality_for_imgqa/batches_dir/  --test-range=10 --train-range=1-9 --layer-def=./conv_ranknet_cfg/imagenet.full.cfg  --layer-params=./conv_ranknet_cfg/params-init.full.cfg --data-provider=pair --test-freq=5 --gpu=1 --epochs=1000 --save-path=./model.20131114 --crop-border=16 

#python ../train-ranknet/convnet.py -f model.20131114/48.2 --layer-params=./conv_ranknet_cfg/params-init.full.1.cfg  --gpu=0 --save-path=./model.2013114/
#python ../train-ranknet/convnet.py -f ./model.image-quality.20130704/model.image-quality.20130704/199.1 --save-path=./model.image-quality.20130704 --epochs=300
