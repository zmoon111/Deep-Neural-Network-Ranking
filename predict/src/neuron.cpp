
#include "neuron.h"
#include <iostream>

using namespace std;

Neuron& Neuron::makeNeuron(dictParam_t &paramsDict) {
    string type = dictGetString(paramsDict, "neuron");
    if (type == "relu") {
        return *new ReluNeuron();
    }
    
    if (type == "softrelu") {
        return *new SoftReluNeuron();
    }
    
    if (type == "brelu") {
        float a = dictGetFloat(paramsDict, "a");
        return *new BoundedReluNeuron(a);
    }

    if (type == "logistic") {
        return *new LogisticNeuron();
    }
    
    if (type == "tanh") {
        float a = dictGetFloat(paramsDict, "a");
        float b = dictGetFloat(paramsDict, "b");
        
        return *new TanhNeuron(a, b);
    }
    
    if (type == "square") {
        return *new SquareNeuron();
    }
    
    if (type == "sqrt") {
        return *new SqrtNeuron();
    }
    
    if (type == "linear") {
        float a = dictGetFloat(paramsDict, "a");
        float b = dictGetFloat(paramsDict, "b");
        return *new LinearNeuron(a, b);
    }

    if (type == "ident") {
        return *new Neuron();
    }
    
    throw string("Unknown neuron type: ") + type;
}
