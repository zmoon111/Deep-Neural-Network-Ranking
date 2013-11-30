
#ifndef CONV_UTIL_H_
#define	CONV_UTIL_H_

#include "matrix.h"
#include "matrix_ssemul.h"
#define BLAS2SSE_THRESH 16

void convFilterActsUnroll(Matrix& images, Matrix& filters, Matrix& targetss, int *offsetIn, int *offsetOut,
        int imgSizeX, int numModulesX, int paddingStart, int moduleStride,
        int numImgColors, int groups, float scaletarget, float scaleOutput);

void convLocalPoolMax(Matrix& images, Matrix& targets, int numFilters,
                   int subsX, int startX, int strideX, int outputsX);

void convLocalPoolAvg(Matrix& images, Matrix& targets, int numFilters,
                   int subsX, int startX, int strideX, int outputsX);

void convContrastNorm(Matrix& images, Matrix& meanDiffs, 
        Matrix& targets, int numFilters, int sizeX, float addScale, float powScale);

void convResponseNorm(Matrix& images, Matrix& targets, int numFilters, int sizeX, float addScale, float powScale);

void convContrastNormCrossMap(Matrix& images, Matrix& meanDiffs, 
                    Matrix& targets, int numFilters, int sizeF, float addScale, float powScale);

void convResponseNormCrossMap(Matrix& images, Matrix& targets, int numFilters, int sizeF, float addScale, float powScale);

void convAddBiases(Matrix& biases, Matrix& targets, int numModules, bool sharedBiases);

void fcWeightMul(Matrix& input, Matrix& weight, float scaleTargets, float scaleOutput, Matrix& targets);

void fcWeightMulSparse(Matrix& input, csc_t* weight, float scaleTargets, float scaleOutput, Matrix& targets);

void fcAddBiases(Matrix& biases, Matrix& targets);

void localFilterActsUnroll(Matrix& images, Matrix& filters, Matrix& targetss, int *offsetIn, int *offsetOut,
        int imgSizeX, int numModulesX, int paddingStart, int moduleStride,
        int numImgColors, float scaletarget, float scaleOutput);

void localAddBiases(Matrix& biases, Matrix& targets, int numModules);

void softmax(Matrix& inputs, Matrix& outputs);
#endif	/* CONV_UTIL_H_ */

