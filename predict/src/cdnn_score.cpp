#include <assert.h>
#include <vector>
#include <malloc.h>
#include "matrix.h"
#include "util.h"
#include "convnet.h"

using namespace std;

int cdnnInitModel(const char *filePath, void *&model) {
    if (model != NULL) {
        fprintf(stderr, "model has been initialized.\n");
        return -1;
    }
    listDictParam_t layerParams;
    if (loadParam(filePath, layerParams) == -1) {
        return -1;
    }

    model = (void *)(new ConvNet(layerParams));

    releaseParam(); 

    if (model != NULL) {
        return 0;
    } else {
        return -1;
    }
}

int cdnnFeatExtract(float *data, void *model, int dataNum, int dataDim, vector<string> &outlayer, float *&outFeat, int &outFeatDim) {
    if (model == NULL) {
        fprintf(stderr, "model has not been initialized.\n");
        return -1;
    }
    if (data == NULL) {
        fprintf(stderr, "data must not be NULL.\n");
        return -1;
    }
    if (dataDim != (((ConvNet*)model)->getLayer(0))->getDataDim()) {
        fprintf(stderr, "dataDim must be identify with the model.\n");
        return -1;
    }
    if (dataNum > 512 || dataNum < 0) {
        fprintf(stderr, "dataNum must be less than 512 and greater than 0.\n");
        return -1;
    }

    Matrix dataMat(data, dataNum, dataDim);
    Matrix probsMat;
    MatrixV feature;

    ((ConvNet*)model)->cnnScore(dataMat, probsMat, outlayer, feature);
    outFeatDim = 0;
    for (int i = 0; i < feature.size(); i++)
    {
        outFeatDim += feature[i]->getNumElements();
    }
    outFeat = (float *)memalign(16, outFeatDim * sizeof(float));
    float *p = outFeat;
    for (int i = 0; i < feature.size(); i++)
    {
        memcpy(p, feature[i]->getData(), feature[i]->getNumElements() * sizeof(float));
        p += feature[i]->getNumElements();
        delete feature[i];
        feature[i] = NULL;
    }

    return 0;
}

int cdnnScore(float *data, void *model, int dataNum, int dataDim, float *probs) {
    if (model == NULL) {
        fprintf(stderr, "model has not been initialized.\n");
        return -1;
    }
    if (data == NULL) {
        fprintf(stderr, "data must not be NULL.\n");
        return -1;
    }
    if (probs == NULL) {
        fprintf(stderr, "probs must not be NULL.\n");
        return -1;
    }
    if (dataDim != (((ConvNet*)model)->getLayer(0))->getDataDim()) {
        fprintf(stderr, "dataDim must be identify with the model.\n");
        return -1;
    }
    if (dataNum > 512 || dataNum < 0) {
        fprintf(stderr, "dataNum must be less than 512 and greater than 0.\n");
        return -1;
    }

    Matrix dataMat(data, dataNum, dataDim);
    Matrix probsMat;

    ((ConvNet*)model)->cnnScore(dataMat, probsMat);

    memcpy(probs, probsMat.getData(), probsMat.getNumElements() * sizeof(float));
    return 0;
}

int cdnnReleaseModel(void **model) {
    if (model != NULL) {
        if (*model != NULL) {
            delete (ConvNet *)(*model);
            *model = NULL;
        }
    }
    return 0;
}

int cdnnGetDataDim(void *model) {
    if (model == NULL) {
        fprintf(stderr, "model has not been initialized.\n");
        return -1;
    }
    return (((ConvNet*)model)->getLayer(0))->getDataDim();
}

int cdnnGetLabelsDim(void *model) {
    if (model == NULL) {
        fprintf(stderr, "model has not been initialized.\n");
        return -1;
    }
    int numLayers = ((ConvNet*)model)->getNumLayers();
    return (((ConvNet*)model)->getLayer(numLayers-1))->getLabelsDim();
}



