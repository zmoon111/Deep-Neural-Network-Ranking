#!/bin/bash

python ../train-ranknet/convnet.py --data-path=/mnt/iscsi_3/dudalong/image-quality/train-batch-all  --test-range=460 --train-range=1-459 --layer-def=./example-layers/imagenet.full.cfg  --layer-params=./example-layers/params-init.full.0.cfg --data-provider=pair --test-freq=5 --gpu=2 --epochs=1000 --save-path=./model.20130810 --crop-border=16 

#python ../train-ranknet/convnet.py -f model.20130816/model.20130810 --layer-params=./example-layers/params-init.full.1.cfg  --gpu=0 --save-path=./model.20130820
#python ../train-ranknet/convnet.py -f ./model.image-quality.20130704/model.image-quality.20130704/199.1 --save-path=./model.image-quality.20130704 --epochs=300
