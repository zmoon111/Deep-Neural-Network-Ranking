

#include <assert.h>
#include <string>
#include <malloc.h>
#include "util.h"

using namespace std;

typedef struct {
    int dim0;
    int dim1;
}dim_t;

char *pubMemory = NULL;

void dictInsert(string key, void *p, dictParam_t &dict) {
    if (dict.end() == dict.find(key)) {
        vector<void *> vec;
        vec.push_back(p);
        dict[key] = vec;
    } else {
        dict[key].push_back(p);
    }
}

void loadDataParam(char *curp, dictParam_t &dict) {
    dictInsert("type", curp, dict);
    curp += strlen(curp) + 1;
    dictInsert("name", curp, dict);
    curp += strlen(curp) + 1;
    dictInsert("dataDim", curp, dict);
}

void loadConvParam(char *curp, dictParam_t &dict) {
    dictInsert("type", curp, dict);
    curp += strlen(curp) + 1;
    dictInsert("name", curp, dict);
    curp += strlen(curp) + 1;
    dictInsert("modulesX", curp, dict);
    curp += sizeof(int);
    dictInsert("filters", curp, dict);
    curp += sizeof(int);
    dictInsert("sharedBiases", curp, dict);
    curp += sizeof(int);
    dictInsert("numInputs", curp, dict);
    int numInputs = *((int *)curp);
    curp += sizeof(int);
    int dim0 = 0, dim1 = 0;
    for (int i = 0; i< numInputs; i++) {
        dictInsert("inputs", curp, dict);
        curp += sizeof(int);
        dictInsert("channels", curp, dict);
        curp += sizeof(int);
        dictInsert("imgSize", curp, dict);
        curp += sizeof(int);
        dictInsert("filterChannels", curp, dict);
        curp += sizeof(int);
        dictInsert("filterSize", curp, dict);
        curp += sizeof(int);
        dictInsert("padding", curp, dict);
        curp += sizeof(int);
        dictInsert("stride", curp, dict);
        curp += sizeof(int);
        dictInsert("groups", curp, dict);
        curp += sizeof(int);
        dim0 = *((int *)curp);
        dim1 = *((int *)(curp+sizeof(int)));
        dictInsert("weights", curp, dict);
        curp += 2 * sizeof(int) + dim0 * dim1 * sizeof(float);
    }
    dim0 = *((int *)curp);
    dim1 = *((int *)(curp+sizeof(int)));
    dictInsert("biases", curp, dict);

}

void loadL2Param(char *curp, dictParam_t &dict) {
    dictInsert("type", curp, dict);
    curp += strlen(curp) + 1;
    dictInsert("name", curp, dict);
    curp += strlen(curp) + 1;
    int numInputs = *((int *)curp);
    curp += sizeof(int);
    int dim0 = 0, dim1 = 0;
    for (int i = 0; i < numInputs; i++) {
        dictInsert("inputs", curp, dict);
        curp += sizeof(int);
        dim0 = *((int *)curp);
        dim1 = *((int *)(curp+sizeof(int)));
        dictInsert("weights", curp, dict);
        curp += 2 * sizeof(int) + dim0 * dim1 * sizeof(float);
    }
    //dim0 = *((int *)curp);
    //dim1 = *((int *)(curp+sizeof(int)));
    //dictInsert("biases", curp, dict);

}

void loadLocalParam(char *curp, dictParam_t &dict) {
    dictInsert("type", curp, dict);
    curp += strlen(curp) + 1;
    dictInsert("name", curp, dict);
    curp += strlen(curp) + 1;
    dictInsert("modulesX", curp, dict);
    curp += sizeof(int);
    dictInsert("filters", curp, dict);
    curp += sizeof(int);
    dictInsert("numInputs", curp, dict);
    int numInputs = *((int *)curp);
    curp += sizeof(int);
    int dim0 = 0, dim1 = 0;
    for (int i = 0; i< numInputs; i++) {
        dictInsert("inputs", curp, dict);
        curp += sizeof(int);
        dictInsert("channels", curp, dict);
        curp += sizeof(int);
        dictInsert("imgSize", curp, dict);
        curp += sizeof(int);
        dictInsert("filterChannels", curp, dict);
        curp += sizeof(int);
        dictInsert("filterSize", curp, dict);
        curp += sizeof(int);
        dictInsert("padding", curp, dict);
        curp += sizeof(int);
        dictInsert("stride", curp, dict);
        curp += sizeof(int);
        dictInsert("groups", curp, dict);
        curp += sizeof(int);
        dim0 = *((int *)curp);
        dim1 = *((int *)(curp+sizeof(int)));
        dictInsert("weights", curp, dict);
        curp += 2 * sizeof(int) + dim0 * dim1 * sizeof(float);
    }
    dim0 = *((int *)curp);
    dim1 = *((int *)(curp+sizeof(int)));
    dictInsert("biases", curp, dict);

}

void loadFcParam(char *curp, dictParam_t &dict) {
    dictInsert("type", curp, dict);
    curp += strlen(curp) + 1;
    dictInsert("name", curp, dict);
    curp += strlen(curp) + 1;
    int numInputs = *((int *)curp);
    curp += sizeof(int);
    dictInsert("sparseFlag", curp, dict);
    curp += sizeof(int);
    int dim0 = 0, dim1 = 0;
    for (int i = 0; i < numInputs; i++) {
        dictInsert("inputs", curp, dict);
        curp += sizeof(int);
        dim0 = *((int *)curp);
        dim1 = *((int *)(curp+sizeof(int)));
        dictInsert("weights", curp, dict);
        curp += 2 * sizeof(int) + dim0 * dim1 * sizeof(float);
    }
    dim0 = *((int *)curp);
    dim1 = *((int *)(curp+sizeof(int)));
    dictInsert("biases", curp, dict);
}

void loadPoolParam(char *curp, dictParam_t &dict) {
    dictInsert("type", curp, dict);
    curp += strlen(curp) + 1;
    dictInsert("name", curp, dict);
    curp += strlen(curp) + 1;
    dictInsert("pool", curp, dict);
    curp += strlen(curp) + 1;
    dictInsert("inputs", curp, dict);
    curp += sizeof(int);
    dictInsert("channels", curp, dict);
    curp += sizeof(int);
    dictInsert("sizeX", curp, dict);
    curp += sizeof(int);
    dictInsert("start", curp, dict);
    curp += sizeof(int);
    dictInsert("stride", curp, dict);
    curp += sizeof(int);
    dictInsert("outputsX", curp, dict);
    curp += sizeof(int);
    dictInsert("imgSize", curp, dict);
}

void loadNormParam(char *curp, dictParam_t &dict) {
    dictInsert("type", curp, dict);
    curp += strlen(curp) + 1;
    dictInsert("name", curp, dict);
    curp += strlen(curp) + 1;
    dictInsert("inputs", curp, dict);
    curp += sizeof(int);
    dictInsert("imgSize", curp, dict);
    curp += sizeof(int);
    dictInsert("channels", curp, dict);
    curp += sizeof(int);
    dictInsert("size", curp, dict);
    curp += sizeof(float);
    dictInsert("scale", curp, dict);
    curp += sizeof(float);
    dictInsert("pow", curp, dict);
}

void loadNeuronParam(char *curp, dictParam_t &dict) {
    dictInsert("type", curp, dict);
    curp += strlen(curp) + 1;
    dictInsert("name", curp, dict);
    curp += strlen(curp) + 1;
    dictInsert("neuron", curp, dict);
    curp += strlen(curp) + 1;
    dictInsert("inputs", curp, dict);
}

void loadSoftmaxParam(char *curp, dictParam_t &dict) {
    dictInsert("type", curp, dict);
    curp += strlen(curp) + 1;
    dictInsert("name", curp, dict);
    curp += strlen(curp) + 1;
    dictInsert("inputs", curp, dict);
    curp += sizeof(int);
    dictInsert("outputs", curp, dict);
}

int loadParam(const char *filePath, listDictParam_t &listDictParam)
{
    FILE *fp = NULL;
    if (NULL == (fp = fopen(filePath, "rb"))) {
        fprintf(stderr, "model read error.\n");
        return -1;
    }
    // init public memory
    fseek(fp, 0, SEEK_END);
    pubMemory = (char *)memalign(16, ftell(fp));
    fseek(fp, 0, SEEK_SET);

    int layersNum, layerLen;
    fread(&layersNum, sizeof(int), 1, fp);
    char *layerp = NULL, *curp = NULL;
    dictParam_t dict;
    layerp = pubMemory;
    for (int i = 0; i < layersNum; i++) {
        fread(&layerLen, sizeof(int), 1, fp);
        if (layerLen != 0) {
            fread(layerp, sizeof(char), layerLen, fp);
        }
        dict.clear();
        curp = layerp;
        if (strcmp(layerp, "data") == 0) {
            loadDataParam(curp, dict);
        } else if (strcmp(layerp, "conv") == 0) {
            loadConvParam(curp, dict);
        } else if (strcmp(layerp, "local") == 0) {
            loadLocalParam(curp, dict);
        } else if (strcmp(layerp, "fc") == 0) {
            loadFcParam(curp, dict);
        } else if (strcmp(layerp, "l2") == 0) {
            loadL2Param(curp, dict);
        } else if (strcmp(layerp, "pool") == 0) {
            loadPoolParam(curp, dict);
        } else if (strcmp(layerp, "neuron") == 0) {
            loadNeuronParam(curp, dict);
        } else if (strcmp(layerp, "cnorm") == 0 || strcmp(layerp, "rnorm") == 0 || strcmp(layerp, "cmrnorm") == 0) {
            loadNormParam(curp, dict);
        } else if (strcmp(layerp, "softmax") == 0) {
            loadSoftmaxParam(curp, dict);
        } else {
            fprintf(stderr, "unregistered layer %s.\n", layerp);
            return -1;
        }
        listDictParam.push_back(dict);

        layerp += layerLen;
    }
    fclose(fp);
    fp = NULL;
    return 0;
}

int releaseParam() {
    if (pubMemory != NULL) {
        free(pubMemory);
        pubMemory = NULL;
    }
    return 0;
}

floatv* getFloatV(vector<void *> &vecSrc) {
    floatv* vec = new floatv();
    for (int i = 0; i < vecSrc.size(); i++) {
        vec->push_back(*((float*)vecSrc[i]));
    }
    return vec;
}

intv* getIntV(vector<void *> &vecSrc) {
    intv* vec = new intv(); 
    for (int i = 0; i < vecSrc.size(); i++) {
        vec->push_back(*((int*)vecSrc[i]));
    }
    return vec;
}


MatrixV* getMatrixV(vector<void *> &vecSrc) {
    MatrixV* vec = new MatrixV();
    for (int i = 0; i < vecSrc.size(); i++) {
        dim_t *dim = (dim_t*)vecSrc[i];
        float *data = (float *)((char*)vecSrc[i] + sizeof(dim_t));
        vec->push_back(new Matrix(data, dim->dim0, dim->dim1));
    }
    return vec;
}

int dictGetInt(dictParam_t &dict, const char* key) {
    assert(dict.find(string(key)) != dict.end());
    return *((int*)dict[string(key)][0]);
}

intv* dictGetIntV(dictParam_t &dict, const char* key) {
    assert(dict.find(string(key)) != dict.end());
    return getIntV(dict[string(key)]);
}

string dictGetString(dictParam_t &dict, const char* key) {
    assert(dict.find(string(key)) != dict.end());
    return string((char*)dict[string(key)][0]);
}

float dictGetFloat(dictParam_t &dict, const char* key) {
    assert(dict.find(string(key)) != dict.end());
    return *((float*)dict[string(key)][0]);
}

floatv* dictGetFloatV(dictParam_t &dict, const char* key) {
    assert(dict.find(string(key)) != dict.end());
    return getFloatV(dict[string(key)]);
}

Matrix* dictGetMatrix(dictParam_t &dict, const char* key) {
    assert(dict.find(string(key)) != dict.end());
    dim_t *dim = (dim_t*)dict[string(key)][0];
    float *data = (float *)((char*)dict[string(key)][0] + sizeof(dim_t));
    return new Matrix(data, dim->dim0, dim->dim1);
}

MatrixV* dictGetMatrixV(dictParam_t &dict, const char* key) {
    assert(dict.find(string(key)) != dict.end());
    return getMatrixV(dict[string(key)]);
}



