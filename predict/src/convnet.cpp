
#include <vector>
#include <iostream>
#include <string>
#include <sys/time.h>

#include "matrix.h"
#include "convnet.h"

using namespace std;

ConvNet::ConvNet(listDictParam_t &layerParams) {
    try {
        int numLayers = layerParams.size();

        for (int i = 0; i < numLayers; i++) {
            dictParam_t &paramsDict = layerParams[i];
            string layerType = dictGetString(paramsDict, "type");
            Layer* l = initLayer(layerType, paramsDict);
            if (i == numLayers - 1) {
                _outputLayer = l;
            }
            // Connect backward links in graph for this layer
            if (i > 0) {
                intv* inputLayers = dictGetIntV(paramsDict, "inputs");
                if (inputLayers != NULL) {
                    for (int i = 0; i < inputLayers->size(); i++) {
                        l->addPrev(getLayer(inputLayers->at(i)));
                    }
                }
                delete inputLayers;
            }
        }

        // Connect the forward links in the graph
        for (int i = 0; i < _layers.size(); i++) {
            vector<Layer*>& prev = _layers[i]->getPrev();
            for (int j = 0; j < prev.size(); j++) {
                prev[j]->addNext(_layers[i]);
            }
        }
    } catch (string& s) {
        cerr << "Error creating ConvNet: " << s << endl;
        exit(1);
    }
}

ConvNet::~ConvNet() {
    
    for (std::vector<Layer*>::iterator iter = _layers.begin(); iter != _layers.end(); iter++) {
        delete *iter;
        *iter = NULL;
    }
}

Layer* ConvNet::initLayer(string& layerType, dictParam_t &paramsDict) {
    if (layerType == "fc") {
        _layers.push_back(new FCLayer(this, paramsDict));
    } else if (layerType == "l2") {
        _layers.push_back(new L2Layer(this, paramsDict));
    } else if (layerType == "conv") {
        _layers.push_back(new ConvLayer(this, paramsDict));
    } else if (layerType == "local") {
        _layers.push_back(new LocalLayer(this, paramsDict));
    } else if (layerType == "pool") {
        _layers.push_back(&PoolLayer::makePoolLayer(this, paramsDict));
    } else if (layerType == "rnorm") {
        _layers.push_back(new ResponseNormLayer(this, paramsDict));
    } else if (layerType == "cmrnorm") {
        _layers.push_back(new CrossMapResponseNormLayer(this, paramsDict));
    } else if (layerType == "cnorm") {
        _layers.push_back(new ContrastNormLayer(this, paramsDict));
    } else if (layerType == "softmax") {
        SoftmaxLayer *o = new SoftmaxLayer(this, paramsDict);
        _layers.push_back(o);
    } else if (layerType == "neuron") {
        _layers.push_back(new NeuronLayer(this, paramsDict));
    } else if (layerType == "data") {
        _layers.push_back(new DataLayer(this, paramsDict));
    } else {
        throw string("Unknown layer type ") + layerType;
    }

    return _layers.back();
}

Layer* ConvNet::operator[](int idx) {
    return _layers[idx];
}

Layer* ConvNet::getLayer(int idx) {
    return _layers[idx];
}

int ConvNet::getNumLayers() {
    return _layers.size();
}


int ConvNet::cnnScore(Matrix& data, Matrix &probs) {
    _outputLayer->fprop(data, probs);
    return 0;
}

int ConvNet::cnnScore(Matrix& data, Matrix &probs, vector<string> &outlayer, MatrixV &feature) {
    _outputLayer->fprop(data, probs, outlayer, feature);
    return 0;
}
