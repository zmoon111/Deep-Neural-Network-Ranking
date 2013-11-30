
#include <malloc.h>
#include <xmmintrin.h>
#include <assert.h>
extern "C" {
#include <cblas.h>
}
#include "matrix.h"
#include "matrix_ssemul.h"
#include "exp.h"
#include "conv_util.h"
using namespace std;


/** 
 * @brief arrange images data for convolution operation
 * 
 */
int imgMemoryPrepare(float *in, int *offsetIn, int *offsetOut, int imgNum, int imgSize, int moduleX, 
        int padding, int channels, int filterSize, int moduleStride, float *&out, int &outRow, int &outCol) {
    int exSize, filterRowLength;
    exSize = ((imgSize + 2 * padding + 3)>>2)<<2;
    filterRowLength = filterSize * filterSize * channels;

    float *inbuf = (float *)memalign(16, exSize*exSize*channels*sizeof(float));
    memset(inbuf, 0, exSize*exSize*channels*sizeof(float));

    float *outbuf = (float *)memalign(16, imgNum*moduleX*moduleX*filterRowLength*sizeof(float));
    for (int i = 0; i < imgNum; i++) {

        float *inTmp = in + i * imgSize * imgSize * channels;
        
        for (int j = 0; j < imgSize; j++) {
            memcpy(inbuf+((j+padding)*exSize+padding)*channels, inTmp+j*imgSize*channels, imgSize*channels*sizeof(float));
        }

        float *outbufTmp = outbuf + i * moduleX * moduleX * filterRowLength;
        for (int i = 0; i < moduleX * moduleX * filterSize; i++) {
            memcpy(outbufTmp+offsetOut[i], inbuf+offsetIn[i], filterSize*channels*sizeof(float));
        }
    }

    out = outbuf;
    outRow = imgNum * moduleX * moduleX;
    outCol = filterRowLength;

    free(inbuf);
    inbuf = NULL;

    return 0;
}

/** 
 * @brief arrange images data for convolution operation, for multi-groups filters
 * 
 */
int imgMemoryPrepareGroup(float *in, int *offsetIn, int *offsetOut, int imgNum, int imgSize, int moduleX, 
        int padding, int channels, int filterChannels, int filterSize, int groups, int moduleStride, 
        float *&out, int &outRow, int &outCol) {
    int exSize, filterRowLength;
    exSize = ((imgSize + 2 * padding + 3)>>2)<<2;
    filterRowLength = filterSize * filterSize * filterChannels;

    float *inbuf = (float *)memalign(16, exSize*exSize*channels*sizeof(float));
    memset(inbuf, 0, exSize*exSize*channels*sizeof(float));

    float *outbuf = (float *)memalign(16, imgNum*moduleX*moduleX*filterRowLength*groups*sizeof(float));
    for (int i = 0; i < imgNum; i++) {

        float *inTmp = in + i * imgSize * imgSize * channels;
        
        for (int j = 0; j < imgSize; j++) {
            memcpy(inbuf+((j+padding)*exSize+padding)*channels, inTmp+j*imgSize*channels, imgSize*channels*sizeof(float));
        }
        for (int j = 0; j < groups; j++) {
            float *outbufTmp = outbuf + j*imgNum*moduleX*moduleX*filterRowLength + i*moduleX*moduleX*filterRowLength ;
            for (int k = 0; k < moduleX * moduleX * filterSize * filterSize; k++) {
                memcpy(outbufTmp+offsetOut[k], inbuf+offsetIn[k]+j*filterChannels, filterChannels*sizeof(float));
            }
        }
    }

    out = outbuf;
    outRow = imgNum * moduleX * moduleX;
    outCol = filterRowLength;

    free(inbuf);
    inbuf = NULL;

    return 0;
}
/** 
 * @brief convolution operation
 * 
 * @param images input images
 * @param filters the convolution kernel
 * @param targets result
 * @param offsetIn offset for arrange input images to more suitable data structure
 * @param offsetOut offset for arrage input images to more suitable data structure
 * @param imgSizeX the size of image
 * @param numModulesX the size of convolution output
 * @param paddingStart the padding start, < 0
 * @param moduleStride the distance between successive convolution applications
 * @param numImgColors the channels of input images 
 * @param groups the group number of filters
 * @param scaleTargets
 * @param scaleOutput
 */
void convFilterActsUnroll(Matrix& images, Matrix& filters, Matrix& targets, int *offsetIn, int *offsetOut, 
                   int imgSizeX, int numModulesX, int paddingStart, int moduleStride, int numImgColors, int groups, 
                   float scaleTargets, float scaleOutput) {
    int numFilterColors = numImgColors / groups;
    assert(numImgColors == numFilterColors * groups);
    int numFilters = filters.getNumRows();
    int numFiltersPerGroup = numFilters / groups;
    int numModules = numModulesX * numModulesX;
    int numImages = images.getNumRows();
    int imgPixels = images.getNumCols() / numImgColors;

    assert((numImgColors > 0 && (numImgColors <= 3 || numImgColors % 8 == 0)));
    assert((numFilters % 8) == 0 && (numFiltersPerGroup % 8) == 0);
    assert(images.getNumCols() == imgPixels * numImgColors);
    assert(imgSizeX * imgSizeX == imgPixels);

    int filterPixels = filters.getNumCols() / numFilterColors;
    int filterSize = int(sqrt(filterPixels));
    int filtersCols = filters.getNumCols();
    assert(filterSize * filterSize == filterPixels);
    assert(filtersCols == numFilterColors * filterPixels);

    assert(paddingStart <= 0);
    assert(paddingStart + (numModulesX-1)*moduleStride + filterSize >= imgSizeX);
    assert(moduleStride <= filterSize);
    assert(!images.isTrans());
    assert(!filters.isTrans());

    if (scaleTargets == 0) {
        targets.resize(numImages, numFilters * numModules);
    }
    float *imgData = images.getData();
    float *filterData = filters.getData();
    float *targetData = targets.getData();

    if (scaleTargets == 0) {
        memset(targetData, 0, sizeof(float) * numImages * numFilters * numModules);
    }

    int padding = -paddingStart;
    float *prepareModule = NULL;
    int prepareModuleRow = 0, prepareModuleCol = 0;
    if (groups == 1) {
        imgMemoryPrepare(imgData, offsetIn, offsetOut, numImages, imgSizeX, numModulesX, padding, numImgColors, filterSize, 
                moduleStride, prepareModule, prepareModuleRow, prepareModuleCol);
    }
    else {
        imgMemoryPrepareGroup(imgData, offsetIn, offsetOut, numImages, imgSizeX, numModulesX, padding, numImgColors, numFilterColors,
                filterSize, groups, moduleStride, prepareModule, prepareModuleRow, prepareModuleCol);
    }
    float *targetTmp = (float *)memalign(16, numFilters * prepareModuleRow * sizeof(float));
    int filterGroupStep = numFiltersPerGroup * filterPixels * numFilterColors;
    int prepareModuleGroupStep = prepareModuleRow * prepareModuleCol;
    int targetTmpGroupStep = numFiltersPerGroup * prepareModuleRow;
    for (int i = 0; i < groups; i++) {
        // the condition was determined by experiments
        if (numFiltersPerGroup > BLAS2SSE_THRESH && numImages > BLAS2SSE_THRESH) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, numFiltersPerGroup, prepareModuleRow, prepareModuleCol,
                1, filterData+i*filterGroupStep, filtersCols, prepareModule+i*prepareModuleGroupStep, prepareModuleCol, 0,
                targetTmp+i*targetTmpGroupStep, prepareModuleRow);
        } else {
            mulBlock16SSE(filterData+i*filterGroupStep, prepareModule+i*prepareModuleGroupStep, targetTmp+i*targetTmpGroupStep, numFiltersPerGroup, prepareModuleRow, prepareModuleCol);
        }
    }
    if (scaleOutput != 1) {
        __m128 a, b;
        float *targetTmpPtr = targetTmp;
        b = _mm_set1_ps(scaleOutput);
        for (int i = 0; i < numFilters * prepareModuleRow; i+=4) {
            a = _mm_load_ps(targetTmpPtr);
            a = _mm_mul_ps(a, b);
            _mm_store_ps(targetTmpPtr, a);
            targetTmpPtr += 4;
        }
    }
    if (groups == 1) {
        for (int i = 0; i < numFilters; i++) {
            for (int j = 0; j < prepareModuleRow; j++) {
                targetData[j*numFilters+i] += targetTmp[i*prepareModuleRow+j];
            }
        }
    } else {
        for (int i = 0; i < groups; i++) {
            for (int j = 0; j < numFiltersPerGroup; j++) {
                for (int k = 0; k < prepareModuleRow; k++) {
                    targetData[k*numFilters+i*numFiltersPerGroup+j] += targetTmp[i*targetTmpGroupStep+j*prepareModuleRow+k];
                }
            }
        }
    }

    free(targetTmp);
    targetTmp = NULL;
    free(prepareModule);
    prepareModule = NULL;
}

/** 
 * @brief make max-pooling operation for images
 * 
 * @param images the input images
 * @param targets the output targets, allocate memory within function
 * @param numFilters the num of filters
 * @param subsX the width of region for pooling
 * @param startX the x-coordinate in the input image to start the pooling
 * @param strideX the distance between successive pooling applications 
 * @param outputsX the output values in the x (equivalently, y) dimension this operation will produce
 */
void convLocalPoolMax(Matrix& images, Matrix& targets, int numFilters,
                   int subsX, int startX, int strideX, int outputsX) {


    int numImages = images.getNumRows();
    int imgPixels = images.getNumCols() / numFilters;
    assert(images.getNumCols() == numFilters * imgPixels);
    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);
    assert(numFilters % 8 == 0);
    assert(!images.isTrans());

    int outputs = outputsX * outputsX;
    targets.resize(numImages, numFilters*outputs);

    float *imgData = images.getData();
    float *targetData = targets.getData();

    __m128 a, b;
    int startL, startT, startR, startB;
    float *targetPtr = NULL, *imgPtr = NULL;
    for (int i = 0; i < numImages; i++) {
        targetPtr = targetData+ i * numFilters * outputs;
        imgPtr = imgData + i * numFilters * imgPixels;
        startL = startX; startT = startX; startR = startX + subsX; startB = startX + subsX;

        for (int j = 0; j < outputsX; j++) {
            for (int k = 0; k < outputsX; k++) {
                for (int l = 0; l < numFilters; l+=4) {

                    a = _mm_set1_ps(-2e38);
                    for (int m = startT; m < startB; m++) {
                        for (int n = startL; n < startR; n++) {
                            b = _mm_loadu_ps(imgPtr+(m*imgSize+n)*numFilters+l);
                            a = _mm_max_ps(a, b);
                        }
                    }
                    _mm_storeu_ps(targetPtr+(j*outputsX+k)*numFilters+l, a);
                }
                startL += strideX;
                startR = min(startR+strideX, imgSize); 
            }
            startT += strideX;
            startB = min(startB+strideX, imgSize);
            startL = startX;
            startR = min(startX+subsX, imgSize);
        }
    }
}


/** 
 * @brief make avg-pooling operation for images, note:this is a original version
 * 
 * @param images the input images
 * @param targets the output targets, allocate memory within function
 * @param numFilters the num of filters
 * @param subsX the width of region for pooling
 * @param startX the x-coordinate in the input image to start the pooling
 * @param strideX the distance between successive pooling applications 
 * @param outputsX the output values in the x (equivalently, y) dimension this operation will produce
 */
void convLocalPoolAvg(Matrix& images, Matrix& targets, int numFilters,
                   int subsX, int startX, int strideX, int outputsX) {

    int numImages = images.getNumRows();
    int imgPixels = images.getNumCols() / numFilters;
    assert(images.getNumCols() == numFilters * imgPixels);
    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);
    assert(numFilters % 8 == 0);
    assert(!images.isTrans());

    int outputs = outputsX * outputsX;
    targets.resize(numImages, numFilters*outputs);

    float *imgData = images.getData();
    float *targetData = targets.getData();
    float subsPixels = subsX * subsX;

    __m128 a, b, c;
    c = _mm_set1_ps(subsPixels);
    int startL, startT, startR, startB;
    float *targetPtr = NULL, *imgPtr = NULL;
    for (int i = 0; i < numImages; i++) {
        targetPtr = targetData+ i * numFilters * outputs;
        imgPtr = imgData + i * numFilters * imgPixels;
        startL = startX; startT = startX; startR = startX + subsX; startB = startX + subsX;

        for (int j = 0; j < outputsX; j++) {
            for (int k = 0; k < outputsX; k++) {
                for (int l = 0; l < numFilters; l+=4) {

                    a = _mm_set1_ps(0);
                    for (int m = startT; m < startB; m++) {
                        for (int n = startL; n < startR; n++) {
                            b = _mm_loadu_ps(imgPtr+(m*imgSize+n)*numFilters+l);
                            a = _mm_add_ps(a, b);
                        }
                    }
                    a = _mm_div_ps(a, c);
                    _mm_storeu_ps(targetPtr+(j*outputsX+k)*numFilters+l, a);
                }
                startL += strideX;
                startR = min(startR+strideX, imgSize);
            }
            startT += strideX;
            startB = min(startB+strideX, imgSize);
            startL = startX;
            startR = min(startX+subsX, imgSize);
        }
    }
}

/** 
 * @brief constrast normalization within single channels, note:this is a original version, incompatible with the current data structure
 * 
 * @param images the input images
 * @param meanDiffs images - mean(images), winSize is sizeX
 * @param targets the output targets, allocate memory within function 
 * @param numFilters the num of filters
 * @param sizeX the width of win for normalize
 * @param addScale 
 * @param powScale
 */
void convContrastNorm(Matrix& images, Matrix& meanDiffs, Matrix& targets, int numFilters, int sizeX, float addScale, float powScale) {
    int numImages = images.getNumRows();
    int imgPixels = images.getNumCols() / numFilters;
    assert(images.getNumCols() == numFilters * imgPixels);
    int imgSize = int(sqrt(imgPixels));
    int halfSize = sizeX / 2;
    assert(imgSize * imgSize == imgPixels);
    assert(meanDiffs.isSameDims(images));
    assert(numFilters % 8 == 0);
    assert(!images.isTrans());
    assert(!meanDiffs.isTrans());

    targets.resize(images);

    float *imgData = images.getData();
    float *diffData = meanDiffs.getData();
    float *targetData = targets.getData();

    int exSize = imgSize+1;
    float *integralData = (float *)memalign(16, exSize*exSize*sizeof(float));

    float *targetPtr = NULL, *imgPtr = NULL, *diffPtr = NULL, *integralPtr = NULL, *top = NULL, *left = NULL, *lefttop = NULL;
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < numFilters; j++) {

            targetPtr = targetData+ i * numFilters * imgPixels + j * imgPixels;
            imgPtr = imgData + i * numFilters * imgPixels + j * imgPixels;
            diffPtr = diffData + i * numFilters * imgPixels + j * imgPixels;

            memset(integralData, 0, exSize * exSize * sizeof(float));
            // compute integral image
            for (int m = 1; m <= imgSize; m++) {
                integralPtr = integralData + m * exSize + 1;
                top = integralPtr - exSize;
                left = integralPtr - 1;
                lefttop = top - 1;
                for (int n = 0; n < imgSize; n++) {
                    integralPtr[n] = *top++ + *left++ - *lefttop++ + _square(diffPtr[(m-1)*imgSize+n]); 
                }
            }

            int startL, startT, startR, startB;
            for (int m = 0; m < imgSize; m++) {
                for (int n = 0; n < imgSize; n++) {
                    startL = _max(0, m-halfSize);
                    startR = _min(imgSize, m-halfSize+sizeX);
                    startT = _max(0, n-halfSize);
                    startB = _min(imgSize, n-halfSize+sizeX);
                    targetPtr[m*imgSize+n] = _pow(1.0f+addScale*(integralData[startB*exSize+startR] 
                                - integralData[startB*exSize+startL] - integralData[startT*exSize+startR] 
                                + integralData[startT*exSize+startL]), -powScale) * imgPtr[m*imgSize+n];
                }
            }
        }
    }

    free(integralData);
    integralData = NULL;
}

void convResponseNorm(Matrix& images, Matrix& targets, int numFilters, int sizeX, float addScale, float powScale) {
    convContrastNorm(images, images, targets, numFilters, sizeX, addScale, powScale);
}


/** 
 * @brief constrast normalization cross channels
 * 
 * @param images the input images
 * @param meanDiffs images - mean(images), winSize is sizeX
 * @param targets the output targets, allocate memory within function 
 * @param numFilters the num of filters
 * @param sizeF the width of win for normalize
 * @param addScale 
 * @param powScale
 */
void convContrastNormCrossMap(Matrix& images, Matrix& meanDiffs, Matrix& targets, int numFilters, 
        int sizeF, float addScale, float powScale) {
    int numImages = images.getNumRows();
    int imgPixels = images.getNumCols() / numFilters;
    int numCols = images.getNumCols();
    assert(numCols == numFilters * imgPixels);
    int imgSize = int(sqrt(imgPixels));
    int halfSize = sizeF / 2;
    assert(imgSize * imgSize == imgPixels);
    assert(meanDiffs.isSameDims(images));
    assert(numFilters % 8 == 0);
    assert(!images.isTrans());
    assert(!meanDiffs.isTrans());

    targets.resize(images);
    float *imgData = images.getData();
    float *diffData = meanDiffs.getData();
    float *targetData = targets.getData();
    float *integralData = (float *)memalign(16, (numFilters+sizeF+1) * sizeof(float));
    for (int i = 0; i <= halfSize; i++) {
        integralData[i] = 0;
    }
    float *targetPtr = NULL, *imgPtr = NULL, *diffPtr = NULL;

    __m128 a, b, scale;
    scale = _mm_set1_ps(addScale);
    for (int i = 0; i < numImages; i++) {

        targetPtr = targetData + i * numCols;
        imgPtr = imgData + i * numCols;
        diffPtr = diffData + i * numCols;

        for (int m = 0; m < imgSize; m++) {
            for (int n = 0; n < imgSize; n++) {
                for (int k = 0; k < numFilters; k++) {
                    integralData[k+halfSize+1] = integralData[k+halfSize] + _square(*diffPtr++);
                }
                for (int k = numFilters+halfSize+1; k <= numFilters+sizeF; k++) {
                    integralData[k] = integralData[numFilters+halfSize];
                }
                for (int k = 0; k < numFilters; k+=4) {

                    a = _mm_load_ps(integralData+k);
                    b = _mm_loadu_ps(integralData+k+sizeF);
                    a = _mm_sub_ps(b, a);
                    a = _mm_mul_ps(a, scale);
                    a = _mm_add_ps(a, _mm_set1_ps(1.0));

                    b = _mm_mul_ps(a, a);
                    a = _mm_mul_ps(b, a);
                    a = _mm_div_ps(_mm_set1_ps(1.0), _mm_sqrt_ps(_mm_sqrt_ps(a)));
                    b = _mm_load_ps(imgPtr);
                    a = _mm_mul_ps(a, b);
                    _mm_storeu_ps(targetPtr, a);

                    imgPtr += 4;
                    targetPtr += 4;
                }
            }
        }
    }

    free(integralData);
    integralData = NULL;
}


void convResponseNormCrossMap(Matrix& images, Matrix& targets, int numFilters, int sizeF, float addScale, float powScale) {
    convContrastNormCrossMap(images, images, targets, numFilters, sizeF, addScale, powScale);
}

void convAddBiases(Matrix& biases, Matrix& targets, int numModules, bool sharedBiases) {
    int numImages = targets.getNumRows();
    int numFilters = targets.getNumCols() / numModules;
    assert(numFilters % 8 == 0);
    assert(targets.getNumCols() == numFilters * numModules);
    assert((sharedBiases && biases.getNumCols() == numFilters) || 
            (!sharedBiases && biases.getNumCols() == numFilters * numModules));
    assert(!biases.isTrans());
    assert(!targets.isTrans());

    float *biasesData = biases.getData();
    float *targetData = targets.getData();
    float *biasesPtr = NULL;

    __m128 a, b;
    if (sharedBiases) {
        for (int i = 0; i < numImages; i++) {
            for (int j = 0; j < numModules; j++) {
                biasesPtr = biasesData;
                for (int k = 0; k < numFilters; k+=4) {
                    a = _mm_loadu_ps(targetData);
                    b = _mm_loadu_ps(biasesPtr);

                    a = _mm_add_ps(a, b);
                    _mm_storeu_ps(targetData, a);
                    targetData += 4;
                    biasesPtr += 4;
                }
            }
        }
    } else {
        for (int i = 0; i < numImages; i++) {
            biasesPtr = biasesData;
            for (int j = 0; j < numFilters * numModules; j+=4) {
                a = _mm_loadu_ps(targetData);
                b = _mm_loadu_ps(biasesPtr);

                a = _mm_add_ps(a, b);
                _mm_storeu_ps(targetData, a);
                targetData += 4;
                biasesPtr += 4;
            }
        }
    }
}

void localAddBiases(Matrix& biases, Matrix& targets, int numModules) {
    int numImages = targets.getNumRows();
    int numFilters = targets.getNumCols() / numModules;
    assert(numFilters % 8 == 0);
    assert(targets.getNumCols() == numFilters * numModules);
    assert(biases.getNumCols() == numFilters * numModules);
    assert(!biases.isTrans());
    assert(!targets.isTrans());

    float *biasesData = biases.getData();
    float *targetData = targets.getData();
    float *biasesPtr = NULL;

    __m128 a, b;
    for (int i = 0; i < numImages; i++) {
        biasesPtr = biasesData;
        for (int j = 0; j < numFilters * numModules; j+=4) {
            a = _mm_loadu_ps(targetData);
            b = _mm_loadu_ps(biasesPtr);

            a = _mm_add_ps(a, b);
            _mm_storeu_ps(targetData, a);
            targetData += 4;
            biasesPtr += 4;
        }
    }
}


void localFilterActsUnroll(Matrix& images, Matrix& filters, Matrix& targets, int *offsetIn, int *offsetOut, 
                   int imgSizeX, int numModulesX, int paddingStart, int moduleStride, int numImgColors, 
                   float scaleTargets, float scaleOutput) {
    int numFilterColors = numImgColors;
    int numModules = numModulesX * numModulesX;
    int numFilters = filters.getNumRows() / numModules;
    int numImages = images.getNumRows();
    int imgPixels = images.getNumCols() / numImgColors;

    assert((numImgColors > 0 && (numImgColors <= 3 || numImgColors % 8 == 0)));
    assert((numFilters % 8) == 0);
    assert(images.getNumCols() == imgPixels * numImgColors);
    assert(imgSizeX * imgSizeX == imgPixels);

    int filterPixels = filters.getNumCols() / numFilterColors;
    int filterSize = int(sqrt(filterPixels));
    int filtersCols = filters.getNumCols();
    assert(filterSize * filterSize == filterPixels);
    assert(filtersCols == numFilterColors * filterPixels);

    assert(paddingStart <= 0);
    assert(paddingStart + (numModulesX-1)*moduleStride + filterSize >= imgSizeX);
    assert(moduleStride <= filterSize);
    assert(!images.isTrans());
    assert(!filters.isTrans());

    if (scaleTargets == 0) {
        targets.resize(numImages, numFilters * numModules);
    }
    float *imgData = images.getData();
    float *filterData = filters.getData();
    float *targetData = targets.getData();

    if (scaleTargets == 0) {
        memset(targetData, 0, sizeof(float) * numImages * numFilters * numModules);
    }

    int padding = -paddingStart;
    float *prepareModule = NULL;
    int prepareModuleRow = 0, prepareModuleCol = 0;
    imgMemoryPrepare(imgData, offsetIn, offsetOut, numImages, imgSizeX, numModulesX, padding, numImgColors, filterSize, 
            moduleStride, prepareModule, prepareModuleRow, prepareModuleCol);

    float *targetTmp = (float *)memalign(16, numFilters * prepareModuleRow * sizeof(float));
    for (int i = 0; i < numFilters; i++) {
        float * filterPtr = filterData + i * prepareModuleCol * numModules;
        for (int j = 0; j < numImages; j++) {
            float *imgPtr = prepareModule + j * prepareModuleCol * numModules;
            float *targetTmpPtr = targetTmp + i * prepareModuleRow + j * numModules;
            vecPairProduct16SSE(filterPtr, imgPtr, targetTmpPtr, numModules, prepareModuleCol);
        }
    }
    if (scaleOutput != 1) {
        __m128 a, b;
        float *targetTmpPtr = targetTmp;
        b = _mm_set1_ps(scaleOutput);
        for (int i = 0; i < numFilters * prepareModuleRow; i+=4) {
            a = _mm_load_ps(targetTmpPtr);
            a = _mm_mul_ps(a, b);
            _mm_store_ps(targetTmpPtr, a);
            targetTmpPtr += 4;
        }
    }
    for (int i = 0; i < numFilters; i++) {
        for (int j = 0; j < prepareModuleRow; j++) {
            targetData[j*numFilters+i] += targetTmp[i*prepareModuleRow+j];
        }
    }

    free(targetTmp);
    targetTmp = NULL;
    free(prepareModule);
    prepareModule = NULL;
}


void fcWeightMul(Matrix& input, Matrix& weight, float scaleTargets, float scaleOutput, Matrix& targets) {
    int numImages = input.getNumRows();
    int inputLen = input.getNumCols();
    int outputLen = weight.getNumCols();
    assert(inputLen % 4 == 0);
    assert(weight.isTrans());
    assert(inputLen == weight.getNumRows());

    float *inputData = input.getData();
    float *weightData = weight.getData();

    if (scaleTargets == 0) {
        targets.resize(numImages, outputLen);
    }
    float *targetData = targets.getData();

    if (numImages > BLAS2SSE_THRESH) {
        targets.addProduct(input, weight, 1, scaleTargets);
    } else {
        float *targetTmp = (float *)memalign(16, numImages * outputLen * sizeof(float));
        mulBlock16SSE(inputData, weightData, targetTmp, numImages, outputLen, inputLen);

        if (scaleOutput != 1) {
            __m128 a, b;
            float *targetTmpPtr = targetTmp;
            b = _mm_set1_ps(scaleOutput);
            for (int i = 0; i < (numImages*outputLen)>>2<<2; i+=4) {
                a = _mm_load_ps(targetTmpPtr);
                a = _mm_mul_ps(a, b);
                _mm_store_ps(targetTmpPtr, a);
                targetTmpPtr += 4;
            }
            for (int i = (numImages*outputLen)>>2<<2; i < numImages*outputLen; i++) {
                targetTmp[i] *= scaleOutput;
            }
        }
        if (scaleTargets == 0) {
            memcpy(targetData, targetTmp, numImages*outputLen*sizeof(float));
        } else {
            __m128 a, b, c;
            float *targetPtr = targetData;
            float *targetTmpPtr = targetTmp;
            b = _mm_set1_ps(scaleTargets);
            for (int i = 0; i < (numImages*outputLen)>>2<<2; i+=4) {
                a = _mm_load_ps(targetPtr);
                a = _mm_mul_ps(a, b);
                c = _mm_load_ps(targetTmpPtr);
                a = _mm_add_ps(a, c);
                _mm_store_ps(targetPtr, a);
                targetTmpPtr += 4;
                targetPtr += 4;
            }
            for (int i = (numImages*outputLen)>>2<<2; i < numImages*outputLen; i++) {
                targetData[i] = targetData[i] * scaleTargets + targetTmp[i];
            }
        }
        free(targetTmp);
    }
}

void fcWeightMulSparse(Matrix& input, csc_t* weight, float scaleTargets, float scaleOutput, Matrix& targets) {
    int numImages = input.getNumRows();
    int inputLen = input.getNumCols();
    int outputLen = weight->cols;
    assert(inputLen % 4 == 0);
    assert(inputLen == weight->rows);

    float *inputData = input.getData();

    if (scaleTargets == 0) {
        targets.resize(numImages, outputLen);
    }
    float *targetData = targets.getData();

    float *targetTmp = (float *)memalign(16, numImages * outputLen * sizeof(float));
    rMatMulCscMatSSE8(inputData, weight, targetTmp, numImages, outputLen, inputLen);

    if (scaleOutput != 1) {
        __m128 a, b;
        float *targetTmpPtr = targetTmp;
        b = _mm_set1_ps(scaleOutput);
        for (int i = 0; i < (numImages*outputLen)>>2<<2; i+=4) {
            a = _mm_load_ps(targetTmpPtr);
            a = _mm_mul_ps(a, b);
            _mm_store_ps(targetTmpPtr, a);
            targetTmpPtr += 4;
        }
        for (int i = (numImages*outputLen)>>2<<2; i < numImages*outputLen; i++) {
            targetTmp[i] *= scaleOutput;
        }
    }
    if (scaleTargets == 0) {
        memcpy(targetData, targetTmp, numImages*outputLen*sizeof(float));
    } else {
        __m128 a, b, c;
        float *targetPtr = targetData;
        float *targetTmpPtr = targetTmp;
        b = _mm_set1_ps(scaleTargets);
        for (int i = 0; i < (numImages*outputLen)>>2<<2; i+=4) {
            a = _mm_load_ps(targetPtr);
            a = _mm_mul_ps(a, b);
            c = _mm_load_ps(targetTmpPtr);
            a = _mm_add_ps(a, c);
            _mm_store_ps(targetPtr, a);
            targetTmpPtr += 4;
            targetPtr += 4;
        }
        for (int i = (numImages*outputLen)>>2<<2; i < numImages*outputLen; i++) {
            targetData[i] = targetData[i] * scaleTargets + targetTmp[i];
        }
    }
    free(targetTmp);
}


void fcAddBiases(Matrix& biases, Matrix& targets) {
    int numImages = targets.getNumRows();
    int biasesLen = biases.getNumCols();
    assert(biases.getNumCols() == targets.getNumCols());
    assert(!biases.isTrans());
    assert(!targets.isTrans());

    float *biasesData = biases.getData();
    float *targetData = targets.getData();
    float *biasesPtr = NULL;
    
    int alignedLen = (biasesLen >> 2) << 2;
    int leftLen = biasesLen & 3;
    __m128 a, b;
    for (int i = 0; i < numImages; i++) {
        biasesPtr = biasesData;
        for (int j = 0; j < alignedLen; j+=4) {
            a = _mm_loadu_ps(targetData);
            b = _mm_loadu_ps(biasesPtr);

            a = _mm_add_ps(a, b);
            _mm_storeu_ps(targetData, a);
            targetData += 4;
            biasesPtr += 4;
        }
        for (int j = 0; j < leftLen; j++) {
            *targetData++ += *biasesPtr++;
        }
    }
}

void softmax(Matrix& inputs, Matrix& outputs) {
    int numImages = inputs.getNumRows();
    int outputLen = inputs.getNumCols();
    int alignedLen = (outputLen >> 2) << 2;
    int leftLen = outputLen & 3;
    
    float *inputData = inputs.getData();
    outputs.resize(numImages, outputLen);
    float *outputData = outputs.getData();
    
    __m128 a, b, c;
    float f4[4];
    for (int i = 0; i < numImages; i++) {
        float *inputPtr = inputData + i * outputLen;
        float max = *inputPtr;
        if (alignedLen > 0) {
            a = _mm_loadu_ps(inputPtr);
            for (int j = 0; j < alignedLen; j += 4) {
                b = _mm_loadu_ps(inputPtr);
                a = _mm_max_ps(a, b);
                inputPtr += 4;
            }
            _mm_storeu_ps(f4, a);
            for (int j = 0; j < 4; j++) {
                if (max < f4[j]) {
                    max = f4[j];
                }
            }
        }
        for (int j = 0; j < leftLen; j++) {
            if (max < *inputPtr) {
                max = *inputPtr;
            }
            inputPtr++;
        }
        inputPtr = inputData + i * outputLen;
        float sum = 0;
        if (alignedLen > 0) {
            a = _mm_set1_ps(max);
            c = _mm_setzero_ps();
            for (int j = 0; j < alignedLen; j += 4) {
                b = _mm_loadu_ps(inputPtr);
                b = _mm_sub_ps(b, a);
                b = exp_ps(b);
                _mm_storeu_ps(inputPtr, b);
                c = _mm_add_ps(b, c);
                inputPtr += 4;
            }
            _mm_storeu_ps(f4, c);
            for (int j = 0; j < 4; j++) {
                sum += f4[j];
            }
        }
        for (int j = 0; j < leftLen; j++) {
            *inputPtr = exp(*inputPtr-max);
            sum += *inputPtr;
            inputPtr++;
        }
        inputPtr = inputData + i * outputLen;
        float *outputPtr = outputData + i * outputLen;
        b = _mm_set1_ps(sum);
        for (int j = 0; j < alignedLen; j += 4) {
            a = _mm_loadu_ps(inputPtr);
            a = _mm_div_ps(a, b);
            _mm_storeu_ps(outputPtr, a);
            inputPtr += 4;
            outputPtr += 4;
        }
        for (int j = 0; j < leftLen; j++) {
            *outputPtr++ = *inputPtr++ / sum;
        }
    }

}


