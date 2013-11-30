
#ifndef LAYER_H_
#define	LAYER_H_

#include <string>
#include <vector>
#include <map>
#include <assert.h>
#include <matrix.h>

#include "convnet.h"
#include "weights.h"
#include "neuron.h"
#include "util.h"
#include "matrix_ssemul.h"

using namespace std;

class ConvNet;

/*
 * Abstract layer.
 */
class Layer {
protected:
    ConvNet* _convNet;
    std::vector<Layer*> _prev, _next;
    
    int _actsTarget;
    std::string _name, _type;
    virtual void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output) = 0;
    
public:    
    Layer(ConvNet *convNet, dictParam_t &paramsDict);
    virtual ~Layer();
    void fprop(Matrix& data, Matrix &output);
    void fprop(Matrix &data, Matrix &output, vector<string> &outlayer, MatrixV &feature);
    std::string& getName();
    std::string& getType();
    void addNext(Layer* l);
    void addPrev(Layer* l);
    std::vector<Layer*>& getPrev();
    std::vector<Layer*>& getNext();

    virtual int getLabelsDim(){return 0;};
    virtual int getDataDim(){return 0;};
};

class NeuronLayer : public Layer {
protected:
    Neuron* _neuron;
    
    virtual void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
public:
    NeuronLayer(ConvNet *convNet, dictParam_t &paramsDict);
    ~NeuronLayer();
};

class WeightLayer : public Layer {
protected:
    WeightList _weights;
    Weights *_biases;
public:
    WeightLayer(ConvNet *convNet, dictParam_t &paramsDict);
    Weights& getWeights(int idx);
    virtual ~WeightLayer() {
        if (_biases != NULL) {
            delete _biases;
        }
    }
};

class L2Layer : public WeightLayer {
protected:
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
public:
    L2Layer(ConvNet * convNet, dictParam_t &paramsDict);
};

class FCLayer : public WeightLayer {
protected:
    int _sparseFlag;
    csc_t **_cscMat;
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
public:
    FCLayer(ConvNet *convNet, dictParam_t &paramsDict);
    int getLabelsDim() {
        return (*_weights[0]).getNumCols();
    }
    ~FCLayer();
};

class SoftmaxLayer : public Layer {
protected:
    int _labelsDim;
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
public:
    SoftmaxLayer(ConvNet *convNet, dictParam_t &paramsDict);
    int getLabelsDim() {
        return _labelsDim;
    }
};


class DataLayer : public Layer {
protected:
    int _inputDim;
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
public:
    DataLayer(ConvNet *convNet, dictParam_t &paramsDict);
    int getDataDim() {
        return _inputDim;
    }
    
};

class LocalLayer : public WeightLayer {
protected:
    intv* _padding, *_stride, *_filterSize, *_channels, *_imgSize, *_groups, *_filterChannels;
    int _modulesX, _modules, _numFilters;
    int **imgOffsetOut, **imgOffsetIn;
    void makeOffset();
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
    
public:
    LocalLayer(ConvNet *convNet, dictParam_t &paramsDict);
    ~LocalLayer();
};

class ConvLayer : public LocalLayer {
protected:
    bool _sharedBiases;
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);

public:
    ConvLayer(ConvNet *convNet, dictParam_t &paramsDict);
    ~ConvLayer(){};
}; 

class PoolLayer : public Layer {
protected:
    int _channels, _sizeX, _start, _stride, _outputsX;
    int _imgSize;
    string _pool;
public:
    PoolLayer(ConvNet *convNet, dictParam_t &paramsDict);
    
    static PoolLayer& makePoolLayer(ConvNet* convNet, dictParam_t &paramsDict);
}; 

class AvgPoolLayer : public PoolLayer {
protected:
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
public:
    AvgPoolLayer(ConvNet *convNet, dictParam_t &paramsDict);
}; 

class MaxPoolLayer : public PoolLayer {
protected:
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
public:
    MaxPoolLayer(ConvNet *convNet, dictParam_t &paramsDict);
};

class ResponseNormLayer : public Layer {
protected:
    int _channels, _size;
    float _scale, _pow;

    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
public:
    ResponseNormLayer(ConvNet *convNet, dictParam_t &paramsDict);
}; 

class CrossMapResponseNormLayer : public ResponseNormLayer {
protected:
    int _imgSize;
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
public:
    CrossMapResponseNormLayer(ConvNet *convNet, dictParam_t &paramsDict);
}; 

class ContrastNormLayer : public ResponseNormLayer {
protected:
    int _imgSize;
    void fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output);
public:
    ContrastNormLayer(ConvNet *convNet, dictParam_t &paramsDict);
};

#endif	/* LAYER_H_ */

