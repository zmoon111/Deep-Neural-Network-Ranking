#!/bin/sh
# Copyright (c) 2013, Yinan Yu (bebekifis@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Author: Yinan Yu 
# Time: Mar/21/2013
# Email: bebekifis@gmail.com
# Filename: gen_patches.sh
# Description: 
if [ $# -ne 5 ]
then
    echo 'usage: sh gen_patches.sh pair_list image_path image_size images_per_batch savepath'
    exit -1;
fi

list=$1
echo 'shuffle list'
#
# shuffle
#
python shuffle.py $list pair_list

echo 'generate ground train list and label list'
# pairwise list path
# label contain vote value
cat pair_list | awk -v path=$2 '{print path"/"$1".jpg";print path"/"$2".jpg";}' > train_list
cat pair_list | awk '{print $3;print $4;}' > label

nfiles=`wc -l label | awk '{print $1}'`
echo 'total ' $nfiles ' samples'

batches=`echo $nfiles/$4 + 1 | bc`
echo 'generate' ${batches} 'batches'

rm -rf $5
mkdir -p $5
for ((i=0;i<${batches}/10;i++))
do
    for((j=0;j<9;j++));

    do
        let "k=$i*10+$j"
        python gen_train.py $k ${3} ${4} ${5} &
    done
    let "k=$i*10+9"
    python gen_train.py $k ${3} ${4} ${5}
    sleep 60
done
sleep 300
ls ${5} > batch_list
#python gen_meta.py $5

