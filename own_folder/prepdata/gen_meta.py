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
# Filename: shuffle.py
# Description:
import sys
import numpy
import cPickle
import os
if __name__ == '__main__':
    assert(len(sys.argv)==2)
    batch_list = os.listdir(sys.argv[1])
    m = []
    c = numpy.zeros((3,3))
    mm = numpy.zeros(3)
    k = 0
    for bl in batch_list:
        print bl
        if bl == "batches.meta":
            continue
        data = cPickle.load(open(sys.argv[1] + bl,'rb'))
        m.append(data['data'].mean(axis=1, dtype=numpy.single))
        # added by Yinan Yu, for jitter
        #d = numpy.double(data['data']).reshape(3, -1)
        #c += numpy.dot(d, d.T)/d.shape[1]
        #mm+= d.mean(axis=1)
        k += 1
    # added by Yinan Yu, for jiiter
    #c/=k
    #mm/=k
    #mm = mm.reshape(3, 1)
    #cov = c - numpy.dot(mm, mm.T)
    #std = numpy.sqrt(numpy.diag(cov)).reshape(3,1);
    #std2 = numpy.dot(std, std.T)
    #a,b = numpy.linalg.eig(cov/std2)

    me = numpy.zeros(m[0].shape, dtype=numpy.single)
    for i in xrange(len(m)):
        me += m[i]
    me /= len(m)

    meta = {}
    meta['data_mean'] = me.reshape(-1,1)
    meta['num_cases_per_batch'] = data['data'].shape[1]
    meta['num_vis'] = data['data'].shape[0]
    #meta['jitter'] = {}
    #meta['jitter']['a'] = a
    #meta['jitter']['b'] = b
    #meta['jitter']['std'] = std
    meta['label_names'] = [0,1,2,3,4]
    cPickle.dump(meta, open(sys.argv[1] + '/batches.meta', 'wb'))
