#ifndef CONVNET_H_
#define CONVNET_H_

#include <vector>
#include <string>
#include <time.h>
#include <math.h>

#include "layer.h"
#include "data.h"
#include "weights.h"
#include "util.h"

class Layer;

class ConvNet {
protected:
    std::vector<Layer*> _layers;
    Layer* _outputLayer;
    
    virtual Layer* initLayer(string& layerType, dictParam_t &paramsDict);
public:
    ConvNet(listDictParam_t &layerParams);

    ~ConvNet();

    Layer* operator[](int idx);
    Layer* getLayer(int idx);

    int getNumLayers();

    int cnnScore(Matrix &data, Matrix &probs);
    int cnnScore(Matrix &data, Matrix &probs, vector<string> &outlayer, MatrixV &feature);
};

#endif	/* CONVNET_H_ */

