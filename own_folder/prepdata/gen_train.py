# Copyright (c) 2013, Yinan Yu (yuyinan@baidu.com)
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

import numpy
import zipfile
import os
import cPickle
import re
from scipy.misc import imread
from scipy.misc import imresize



def pickle(filename, data, compress=False):
    if compress:
        fo = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
        fo.writestr('data', cPickle.dumps(data, -1))
    else:
        fo = open(filename, "w")
        cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)

    fo.close()

def unpickle(filename):
    if not os.path.exists(filename):
        raise UnpickleError("Path '%s' does not exist." % filename)

    fo = open(filename, 'rb')
    dict = cPickle.load(fo)

    fo.close()
    return dict

import sys
if __name__ == '__main__':
    # argv1 :  batch_index
    # argv2 :  imgsize
    # argv3 :  batch size
    # argv4 :  savepath
    rank = int(sys.argv[1])
    imsz = int(sys.argv[2])
    psize = int(sys.argv[3])
    savepath = sys.argv[4]
    print rank, imsz, psize, savepath
    fo = open('train_list','r')
    lists = fo.readlines()
    lists = lists[rank*psize:(rank+1)*psize]
    if len(lists) == 0:
        exit(-1)

    fo.close();
    fo = open('label','r')
    gt = fo.readlines()
    gt = gt[rank*psize:(rank+1)*psize]
    fo.close()
    data = numpy.zeros((imsz**2*3, psize), dtype='uint8')
    label = []
    filenames = []
    k = 0
    for i in xrange(len(lists)):
        if i % 2 == 1:
            continue
        # pairs
        fim1 = lists[i][:-1]
        fim2 = lists[i+1][:-1]
        # load image
        try:
            im1 = imread(fim1)
            im2 = imread(fim2)
        except:
            print 'error image', fim1, fim2
            continue
        if len(im1.shape)==0 or len(im2.shape)==0:
            print 'error image', fim1, fim2
            continue
        #if numpy.abs(gt[i]-gt[i+1]) <= 1:
        #    continue
        # resize image
        im1 = imresize(im1, [imsz,imsz])
        im2 = imresize(im2, [imsz,imsz])
        if im1.ndim == 2:
            # one channel image
            # first reshape
            # second chanel1 = im1 channel2 = im1 channel3 = im1
            im1 = im1.reshape(imsz, imsz, 1)
            im1 = numpy.concatenate((im1,im1,im1), axis=2)
        if im1.shape[2] > 3:
            im1 = im1[:,:,:3]
        if im2.ndim == 2:
            im2 = im2.reshape(imsz, imsz, 1)
            im2 = numpy.concatenate((im2,im2,im2), axis=2)
        if im2.shape[2] > 3:
            im2 = im2[:,:,:3]
        # BGR->RBG
        # row first, col second and color third
        im1 = im1.transpose(2, 0, 1)
        im2 = im2.transpose(2, 0, 1)
        # reshape to vector
        im1 = im1.reshape(-1, 1)
        im2 = im2.reshape(-1, 1)
        # add to data list
        data[:,k] = im1[:,0]
        k += 1
        data[:,k] = im2[:,0]
        k += 1
        # get image label
        label.append(int(gt[i]))
        label.append(int(gt[i+1]))
        # get image name
        filenames.append(fim1)
        filenames.append(fim2)
        # calculate data_mean
        if (i % 100) == 0:
            print i
    data = data[:,0:k]
    assert(data.shape[1] == len(label))
    #subdata = numpy.array(data, dtype='uint8')
    # remove dimension with 1 dims
    #subdata = subdata.squeeze()
    # change first dimension to number of cases
    #subdata = subdata.transpose()
    # prepare output batch
    output = {}
    output['labels'] = label
    output['data'] = data
    output['filenames'] = filenames
    output['batch_label'] = 'train batch ' + str(rank + 1)
    outputpath = os.path.join(savepath + '/data_batch_' + str(rank+1))
    pickle(outputpath, output)

