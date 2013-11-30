 
#include <iostream>
#include <malloc.h>
#include "layer.h"
#include "util.h"
#include "conv_util.h"
#include "matrix.h"

using namespace std;

/* 
 * =======================
 * Layer
 * =======================
 */

Layer::Layer(ConvNet* convNet, dictParam_t& paramsDict) :
             _convNet(convNet){
    _name = dictGetString(paramsDict, "name");
    _type = dictGetString(paramsDict, "type");
}

Layer::~Layer() {
}

void Layer::fprop(Matrix &data, Matrix &output) {

    if (_prev[0]->getType() == "data") {
        fpropActs(data, 0, 0, output);
        return;
    }
    MatrixV &matV = *(new MatrixV);
    for (int i = 0; i < _prev.size(); i++) {
        Matrix *mat = new Matrix;
        _prev[i]->fprop(data, *mat);
        matV.push_back(mat);
    }

    for (int i = 0; i < _prev.size(); i++) {
        fpropActs(*(matV[i]), i, i > 0, output);
    }
    for (int i = 0; i < _prev.size(); i++) {
        delete matV[i];
        matV[i] = NULL;
    }
    delete &matV;
}

void Layer::fprop(Matrix &data, Matrix &output, vector<string> &outlayer, MatrixV &feature) {

    if (_prev[0]->getType() == "data") {
        fpropActs(data, 0, 0, output);
        return;
    }
    MatrixV &matV = *(new MatrixV);
    for (int i = 0; i < _prev.size(); i++) {
        Matrix *mat = new Matrix;
        _prev[i]->fprop(data, *mat, outlayer, feature);
        matV.push_back(mat);
    }

    for (int i = 0; i < _prev.size(); i++) {
        fpropActs(*(matV[i]), i, i > 0, output);
    }
    for (int i = 0; i < outlayer.size(); i++)
    {
        if (_name == outlayer[i])
        {
            Matrix *o = &(output.copy());
            feature.push_back(o);
        }
    }

    for (int i = 0; i < _prev.size(); i++) {
        delete matV[i];
        matV[i] = NULL;
    }
    delete &matV;
}

string& Layer::getName() {
    return _name;
}

string& Layer::getType() {
    return _type;
}

void Layer::addNext(Layer* l) {
    _next.push_back(l);
}

void Layer::addPrev(Layer* l) {
    _prev.push_back(l);
}

vector<Layer*>& Layer::getPrev() {
    return _prev;
}

vector<Layer*>& Layer::getNext() {
    return _next;
}

/* 
 * =======================
 * NeuronLayer
 * =======================
 */
NeuronLayer::NeuronLayer(ConvNet* convNet, dictParam_t& paramsDict) 
    : Layer(convNet, paramsDict) {
        _neuron = &Neuron::makeNeuron(paramsDict);
    }
NeuronLayer::~NeuronLayer() {
    if (_neuron != NULL) {
        delete _neuron;
        _neuron = NULL;
    }
}
void NeuronLayer::fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output) {
    output.resize(input);
    _neuron->activate(input, output);
}

/* 
 * =======================
 * WeightLayer
 * =======================
 */
WeightLayer::WeightLayer(ConvNet* convNet, dictParam_t& paramsDict) : 
    Layer(convNet, paramsDict) {

    MatrixV& hWeights = *dictGetMatrixV(paramsDict, "weights");
    Matrix& hBiases = *dictGetMatrix(paramsDict, "biases");

    for (int i = 0; i < hWeights.size(); i++) {
        _weights.addWeights(*new Weights(*hWeights[i]));
    }
    _biases = new Weights(hBiases);

    delete &hWeights;
}

Weights& WeightLayer::getWeights(int idx) {
    return _weights[idx];
}
/* 
 * =======================
 * L2Layer
 * =======================
 */
L2Layer::L2Layer(ConvNet* convNet, dictParam_t& paramsDict) : WeightLayer(convNet, paramsDict) {
    for (int i = 0; i < _weights.getSize(); i++) {
        (*_weights[i]).reverseBlasTrans();
    }
}

void L2Layer::fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output) {
    fcWeightMul(input, *_weights[inpIdx], scaleTargets, 2, output);
    Matrix w2;
    (*_weights[inpIdx]).apply(Matrix::SQUARE, w2);
    w2.sum(0);
    Matrix x2;
    input.apply(Matrix::SQUARE, x2);
    x2.sum(1);
    output.addVector(w2, -1);
    output.addVector(x2, -1);
    //TODO;
    //output = -(input - W)^T*(input - W) = 2*input*W - sum(X.^2, 0) - sum(W.^2, 1)

    
    //if (scaleTargets == 0) {
    //    fcAddBiases(_biases->getW(), output);
    //}
}

/* 
 * =======================
 * FCLayer
 * =======================
 */
FCLayer::FCLayer(ConvNet* convNet, dictParam_t& paramsDict) : WeightLayer(convNet, paramsDict) {
    int weightsNum = _weights.getSize();
    for (int i = 0; i < weightsNum; i++) {
        (*_weights[i]).reverseBlasTrans();
    }

    _sparseFlag = dictGetInt(paramsDict, "sparseFlag");
    _cscMat = NULL;
    _cscMat = (csc_t **)malloc(sizeof(csc_t*) * weightsNum);
    if (_sparseFlag) {
        for (int i = 0; i < weightsNum; i++) {
            _cscMat[i] = NULL;
            cDense2CscAlign16((*_weights[i]).getNumRows(), (*_weights[i]).getNumCols(), (*_weights[i]).getData(), _cscMat[i]);
        }
    }
}

FCLayer::~FCLayer() {
    if (_sparseFlag) {
        for (int i = 0; i < _weights.getSize(); i++)
        {
            releaseCscMat(&(_cscMat[i]));
            _cscMat[i] = NULL;
        }
    }
    free(_cscMat);
    _cscMat = NULL;
}

void FCLayer::fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output) {
    if (_sparseFlag == 0) {
        fcWeightMul(input, *_weights[inpIdx], scaleTargets, 1, output);
    } else {
        fcWeightMulSparse(input, _cscMat[inpIdx], scaleTargets, 1, output);
    }
    if (scaleTargets == 0) {
        fcAddBiases(_biases->getW(), output);
    }
}

/* 
 * =======================
 * LocalLayer
 * =======================
 */
LocalLayer::LocalLayer(ConvNet* convNet, dictParam_t& paramsDict) 
    : WeightLayer(convNet, paramsDict) {
    _modulesX = dictGetInt(paramsDict, "modulesX");
    _numFilters = dictGetInt(paramsDict, "filters");
    _channels = dictGetIntV(paramsDict, "channels");
    _imgSize = dictGetIntV(paramsDict, "imgSize");
    _filterChannels = dictGetIntV(paramsDict, "filterChannels");
    _filterSize = dictGetIntV(paramsDict, "filterSize");
    _padding = dictGetIntV(paramsDict, "padding");
    _stride = dictGetIntV(paramsDict, "stride");
    _groups = dictGetIntV(paramsDict, "groups");

    imgOffsetOut = NULL;
    imgOffsetIn = NULL;
    makeOffset();
}

LocalLayer::~LocalLayer() {
    if (imgOffsetOut != NULL) {
        for (int i = 0; i < _filterSize->size(); i++) {
            free(imgOffsetOut[i]);
        }
        free(imgOffsetOut);
        imgOffsetOut = NULL;
    }
    if (imgOffsetIn != NULL) {
        for (int i = 0; i < _filterSize->size(); i++) {
            free(imgOffsetIn[i]);
        }
        free(imgOffsetIn);
        imgOffsetIn = NULL;
    }
    delete _padding;
    _padding = NULL;
    delete _stride;
    _stride = NULL;
    delete _filterSize;
    _filterSize = NULL;
    delete _channels;
    _channels = NULL;
    delete _imgSize;
    _imgSize = NULL;
    delete _groups;
    _groups = NULL;
    delete _filterChannels;
    _filterChannels = NULL;


}

void LocalLayer::makeOffset() {
    imgOffsetOut = (int **)malloc(sizeof(int *) * _filterSize->size());
    imgOffsetIn = (int **)malloc(sizeof(int *) * _filterSize->size());
    for (int inpIdx = 0; inpIdx < _filterSize->size(); inpIdx++) {
        int moduleX = _modulesX;
        int filterSize = _filterSize->at(inpIdx);
        int channels = _channels->at(inpIdx);
        int filterChannels = _filterChannels->at(inpIdx);
        int moduleStride = _stride->at(inpIdx);
        int filterRowLength = filterSize * filterSize * filterChannels;
        int padding = -_padding->at(inpIdx);
        int exSize = _imgSize->at(inpIdx) + 2 * padding;
        int groups = _groups->at(inpIdx);
        exSize = ((exSize + 3)>>2)<<2;
        if (groups == 1) {
            imgOffsetOut[inpIdx] = (int *)memalign(16, moduleX * moduleX * filterSize * sizeof(int));                              
            imgOffsetIn[inpIdx] = (int *)memalign(16, moduleX * moduleX * filterSize * sizeof(int));
            for (int i = 0; i < moduleX; i++) {                                                                         
                for (int j = 0; j < moduleX; j++) {                                                                     
                    for (int k = 0; k < filterSize; k++) {                                                              
                        imgOffsetOut[inpIdx][(i*moduleX+j)*filterSize+k] = (i*moduleX+j)*filterRowLength + k*filterSize*channels;
                        imgOffsetIn[inpIdx][(i*moduleX+j)*filterSize+k] = ((i*moduleStride+k)*exSize + j*moduleStride)*channels; 
                    }                                                                                                   
                }                                                                                                       
            }
        } else {
            imgOffsetOut[inpIdx] = (int *)memalign(16, moduleX * moduleX * filterSize * filterSize * sizeof(int));
            imgOffsetIn[inpIdx] = (int *)memalign(16, moduleX * moduleX * filterSize * filterSize * sizeof(int));
            for (int i = 0; i < moduleX; i++) {
                for (int j = 0; j < moduleX; j++) {
                    for (int k = 0; k < filterSize; k++) {
                        for (int l = 0; l < filterSize; l++) {
                            imgOffsetOut[inpIdx][(i*moduleX+j)*filterSize*filterSize+k*filterSize+l] = 
                                (i*moduleX+j)*filterRowLength + (k*filterSize+l)*filterChannels;
                            imgOffsetIn[inpIdx][(i*moduleX+j)*filterSize*filterSize+k*filterSize+l] = 
                                ((i*moduleStride+k)*exSize + j*moduleStride + l) * channels;
                        }
                    }
                }
            }
        }
    }
}

void LocalLayer::fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output) {
    localFilterActsUnroll(input, *_weights[inpIdx], output, imgOffsetIn[inpIdx], imgOffsetOut[inpIdx], 
            _imgSize->at(inpIdx), _modulesX,_padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx), 
            scaleTargets, 1);

    if (scaleTargets == 0) {
        localAddBiases(_biases->getW(), output, _modulesX * _modulesX);
    }
}


/* 
 * =======================
 * ConvLayer
 * =======================
 */
ConvLayer::ConvLayer(ConvNet* convNet, dictParam_t& paramsDict) : LocalLayer(convNet, paramsDict) {
    _sharedBiases = dictGetInt(paramsDict, "sharedBiases");
}
void ConvLayer::fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output) {
    convFilterActsUnroll(input, *_weights[inpIdx], output, imgOffsetIn[inpIdx], imgOffsetOut[inpIdx], 
            _imgSize->at(inpIdx), _modulesX,_padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx), 
            _groups->at(inpIdx), scaleTargets, 1);

    if (scaleTargets == 0) {
        convAddBiases(_biases->getW(), output, _modulesX * _modulesX, _sharedBiases);
    }
}

/* 
 * =======================
 * SoftmaxLayer
 * =======================
 */
SoftmaxLayer::SoftmaxLayer(ConvNet* convNet, dictParam_t &paramsDict) : Layer(convNet, paramsDict) {
    _labelsDim = dictGetInt(paramsDict, "outputs");
}

void SoftmaxLayer::fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output) {
    softmax(input, output);
}

/* 
 * =======================
 * DataLayer
 * =======================
 */
DataLayer::DataLayer(ConvNet* convNet, dictParam_t& paramsDict) : Layer(convNet, paramsDict) {
    _inputDim = dictGetInt(paramsDict, "dataDim");
}

void DataLayer::fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output) {
    output = input;    
}
/* 
 * =====================
 * PoolLayer
 * =====================
 */
PoolLayer::PoolLayer(ConvNet* convNet, dictParam_t& paramsDict) 
    : Layer(convNet, paramsDict) {
    _pool = dictGetString(paramsDict, "pool");
    _channels = dictGetInt(paramsDict, "channels");
    _sizeX = dictGetInt(paramsDict, "sizeX");
    _start = dictGetInt(paramsDict, "start");
    _stride = dictGetInt(paramsDict, "stride");
    _outputsX = dictGetInt(paramsDict, "outputsX");
    _imgSize = dictGetInt(paramsDict, "imgSize");
}

PoolLayer& PoolLayer::makePoolLayer(ConvNet* convNet, dictParam_t& paramsDict) {
    string _pool = dictGetString(paramsDict, "pool");
    if (_pool == "max") {
        return *new MaxPoolLayer(convNet, paramsDict);
    } else if(_pool == "avg") {
        return *new AvgPoolLayer(convNet, paramsDict);
    }
    throw string("Unknown pooling layer type ") + _pool;
}

/* 
 * =====================
 * AvgPoolLayer
 * =====================
 */
AvgPoolLayer::AvgPoolLayer(ConvNet* convNet, dictParam_t& paramsDict) : PoolLayer(convNet, paramsDict) {
}

void AvgPoolLayer::fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output) {
    convLocalPoolAvg(input, output, _channels, _sizeX, _start, _stride, _outputsX);
}


/* 
 * =====================
 * MaxPoolLayer
 * =====================
 */
MaxPoolLayer::MaxPoolLayer(ConvNet* convNet, dictParam_t& paramsDict) : PoolLayer(convNet, paramsDict) {
}

void MaxPoolLayer::fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output) {
    convLocalPoolMax(input, output, _channels, _sizeX, _start, _stride, _outputsX);
}

/* 
 * =====================
 * ResponseNormLayer
 * =====================
 */
ResponseNormLayer::ResponseNormLayer(ConvNet* convNet, dictParam_t& paramsDict) : Layer(convNet, paramsDict) {
    _channels = dictGetInt(paramsDict, "channels");
    _size = dictGetInt(paramsDict, "size");

    _scale = dictGetFloat(paramsDict, "scale");
    _pow = dictGetFloat(paramsDict, "pow");
}

void ResponseNormLayer::fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output) {
    convResponseNorm(input, output, _channels, _size, _scale, _pow);
}

/* 
 * =====================
 * CrossMapResponseNormLayer
 * =====================
 */
CrossMapResponseNormLayer::CrossMapResponseNormLayer(ConvNet* convNet, dictParam_t& paramsDict) : ResponseNormLayer(convNet, paramsDict) {
    _imgSize = dictGetInt(paramsDict, "imgSize");
}

void CrossMapResponseNormLayer::fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output) {
    convResponseNormCrossMap(input, output, _channels, _size, _scale, _pow);
}



/* 
 * =====================
 * ContrastNormLayer
 * =====================
 */
ContrastNormLayer::ContrastNormLayer(ConvNet* convNet, dictParam_t& paramsDict) : ResponseNormLayer(convNet, paramsDict) {
    _imgSize = dictGetInt(paramsDict, "imgSize");
}

void ContrastNormLayer::fpropActs(Matrix &input, int inpIdx, float scaleTargets, Matrix &output) {
    Matrix meanDiffs;
    convLocalPoolAvg(input, meanDiffs, _channels, _size, 0, 1, _imgSize);
    meanDiffs.subtract(input);
    convContrastNorm(input, meanDiffs, output, _channels, _size, _scale, _pow);
}

