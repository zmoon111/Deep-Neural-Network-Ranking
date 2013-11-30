
import numpy as np
import sys
import struct
import cPickle

def modelTrans(model_path):
    newLayers = []
    flag = True
    model = cPickle.load(open(model_path, "rb"))
    layers = model['model_state']['layers']
    # ignore cost layer
    for n in xrange(len(layers)-1):
        layer = layers[n]
        # ignore labels layer
        if layer['type'] == 'data' and layer['name'] == 'labels':
            continue
        elif layer['type'] == 'conv':
            for m in xrange(len(layer['weights'])):
                weightArr = layer['weights'][m]
                weightTmp = np.array(weightArr.transpose())
                for i in xrange(int(layer['filters'])):
                    for j in xrange(int(layer['filterPixels'][m])):
                        for k in xrange(int(layer['filterChannels'][m])):
                            weightTmp[i][j*int(layer['filterChannels'][m])+k] = weightArr[k*int(layer['filterPixels'][m])+j][i]
                layer['weights'][m] = weightTmp
            layer['biases'] = layer['biases'].transpose()

        elif layer['type'] == 'fc' and flag == True:
            flag = False
            outputsX = int(layers[n-1]['outputsX'])
            channels = int(layers[n-1]['channels'])
            for m in xrange(len(layer['weights'])):
                weightTmp = np.array(layer['weights'][m])
                for i in xrange(outputsX*outputsX):
                    for j in xrange(channels):
                        weightTmp[i*channels+j][:] = layer['weights'][m][j*outputsX*outputsX+i][:]
                layer['weights'][m] = weightTmp
        if 'dropRate' in layer.keys():
            for m in xrange(len(layer['weights'])):
                layer['weights'][m] *= 1-np.float32(layer['dropRate'])
            layer['biases'] *= 1-np.float32(layer['dropRate'])
        if layer['type'] != 'data':
            for n in xrange(len(layer['inputs'])):
                if layer['inputs'][n] > 1:
                    layer['inputs'][n] -= 1
        newLayers.append(layer)

    return newLayers

def pack_data(layer):
    pack = ''
    pack += struct.pack('%ds'%len(layer['type']), layer['type'])
    pack += struct.pack('c', '\0')
    pack += struct.pack('%ds'%len(layer['name']), layer['name'])
    pack += struct.pack('c', '\0')
    #dataDim, dim of inputs == dim of outputs
    pack += struct.pack('i', layer['outputs'])
    return pack

def pack_conv(layer):
    pack = ''
    pack += struct.pack('%ds'%len(layer['type']), layer['type'])
    pack += struct.pack('c', '\0')
    pack += struct.pack('%ds'%len(layer['name']), layer['name'])
    pack += struct.pack('c', '\0')
    pack += struct.pack('i', layer['modulesX'])
    pack += struct.pack('i', layer['filters'])
    pack += struct.pack('i', layer['sharedBiases'])
    #numInputs
    pack += struct.pack('i', len(layer['inputs']))
    for n in xrange(len(layer['channels'])):
        pack += struct.pack('i', layer['inputs'][n])
        pack += struct.pack('i', layer['channels'][n])
        pack += struct.pack('i', layer['imgSize'][n])
        pack += struct.pack('i', layer['filterChannels'][n])
        pack += struct.pack('i', layer['filterSize'][n])
        pack += struct.pack('i', layer['padding'][n])
        pack += struct.pack('i', layer['stride'][n])
        pack += struct.pack('i', layer['groups'][n])
        tmp = layer['weights'][n].tostring()
        pack += struct.pack('ii%ds'%len(tmp), layer['weights'][n].shape[0], layer['weights'][n].shape[1], tmp)
    #biases
    tmp = layer['biases'].tostring()
    pack += struct.pack('ii%ds'%len(tmp), layer['biases'].shape[0], layer['biases'].shape[1], tmp)
    return pack

def pack_l2(layer):
    pack = ''
    pack += struct.pack('%ds'%len(layer['type']), layer['type'])
    pack += struct.pack('c', '\0')
    pack += struct.pack('%ds'%len(layer['name']), layer['name'])
    pack += struct.pack('c', '\0')
    pack += struct.pack('i', len(layer['inputs']))
    for n in xrange(len(layer['weights'])):
        pack += struct.pack('i', layer['inputs'][n])
        tmp = layer['weights'][n].tostring()
        pack += struct.pack('ii%ds'%len(tmp), layer['weights'][n].shape[0], layer['weights'][n].shape[1], tmp)
    #tmp = layer['biases'].tostring()
    #ack += struct.pack('ii%ds'%len(tmp), layer['biases'].shape[0], layer['biases'].shape[1], tmp)
    return pack


def pack_fc(layer):
    pack = ''
    pack += struct.pack('%ds'%len(layer['type']), layer['type'])
    pack += struct.pack('c', '\0')
    pack += struct.pack('%ds'%len(layer['name']), layer['name'])
    pack += struct.pack('c', '\0')
    pack += struct.pack('i', len(layer['inputs']))
    for n in xrange(len(layer['weights'])):
        pack += struct.pack('i', layer['inputs'][n])
        tmp = layer['weights'][n].tostring()
        pack += struct.pack('ii%ds'%len(tmp), layer['weights'][n].shape[0], layer['weights'][n].shape[1], tmp)
    tmp = layer['biases'].tostring()
    pack += struct.pack('ii%ds'%len(tmp), layer['biases'].shape[0], layer['biases'].shape[1], tmp)
    return pack

def pack_pool(layer):
    pack = ''
    pack += struct.pack('%ds'%len(layer['type']), layer['type'])
    pack += struct.pack('c', '\0')
    pack += struct.pack('%ds'%len(layer['name']), layer['name'])
    pack += struct.pack('c', '\0')
    pack += struct.pack('%ds'%len(layer['pool']), layer['pool'])
    pack += struct.pack('c', '\0')
    pack += struct.pack('i', layer['inputs'][0])
    pack += struct.pack('i', layer['channels'])
    pack += struct.pack('i', layer['sizeX'])
    pack += struct.pack('i', layer['start'])
    pack += struct.pack('i', layer['stride'])
    pack += struct.pack('i', layer['outputsX'])
    pack += struct.pack('i', layer['imgSize'])
    return pack

def pack_norm(layer):
    pack = ''
    pack += struct.pack('%ds'%len(layer['type']), layer['type'])
    pack += struct.pack('c', '\0')
    pack += struct.pack('%ds'%len(layer['name']), layer['name'])
    pack += struct.pack('c', '\0')
    pack += struct.pack('i', layer['inputs'][0])
    pack += struct.pack('i', layer['imgSize'])
    pack += struct.pack('i', layer['channels'])
    pack += struct.pack('i', layer['size'])
    pack += struct.pack('f', layer['scale'])
    pack += struct.pack('f', layer['pow'])
    return pack

def pack_neuron(layer):
    pack = ''
    pack += struct.pack('%ds'%len(layer['type']), layer['type'])
    pack += struct.pack('c', '\0')
    pack += struct.pack('%ds'%len(layer['name']), layer['name'])
    pack += struct.pack('c', '\0')
    pack += struct.pack('%ds'%len(layer['neuron']['type']), layer['neuron']['type'])
    pack += struct.pack('c', '\0')
    pack += struct.pack('i', layer['inputs'][0])
    return pack


def pack_softmax(layer):
    pack = ''
    pack += struct.pack('%ds'%len(layer['type']), layer['type'])
    pack += struct.pack('c', '\0')
    pack += struct.pack('%ds'%len(layer['name']), layer['name'])
    pack += struct.pack('c', '\0')
    pack += struct.pack('i', layer['inputs'][0])
    pack += struct.pack('i', layer['outputs'])
    return pack

def modelToBin(layers):
    packAll = ''
    layerNum = len(layers)
    packAll += struct.pack('i', layerNum)
    for layer in layers:
        if layer['type'] == 'data':
            pack = pack_data(layer)
            packAll += struct.pack('i', len(pack)) + pack

        if layer['type'] == 'neuron':
            pack = pack_neuron(layer)
            packAll += struct.pack('i', len(pack)) + pack

        if layer['type'] == 'softmax':
            pack = pack_softmax(layer)
            packAll += struct.pack('i', len(pack)) + pack

        if layer['type'] == 'conv':
            pack = pack_conv(layer)
            packAll += struct.pack('i', len(pack)) + pack

        if layer['type'] == 'pool':
            pack = pack_pool(layer)
            packAll += struct.pack('i', len(pack)) + pack

        if layer['type'] == 'l2':
            pack = pack_l2(layer)
            packAll += struct.pack('i', len(pack)) + pack

        if layer['type'] == 'fc':
            pack = pack_fc(layer)
            packAll += struct.pack('i', len(pack)) + pack

        if layer['type'] == 'cnorm' or layer['type'] == 'rnorm' or layer['type'] == 'cmrnorm':
            pack = pack_norm(layer)
            packAll += struct.pack('i', len(pack)) + pack

    return packAll

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print "usage: python genModel.py model model.bin"
        sys.exit(0)
    layers = modelTrans(sys.argv[1])
    packAll = modelToBin(layers)
    fp = open(sys.argv[2], 'wb')
    fp.write(packAll)
    fp.close()

