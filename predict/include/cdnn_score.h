
#ifndef CDNNSCORE_H_
#define	CDNNSCORE_H_

#include <vector>
#include <string>
using namespace std;

int cdnnInitModel(const char *filePath, void *&model);

int cdnnScore(float *data, void *model, int dataNum, int dataDim, float *probs);

int cdnnFeatExtract(float *data, void *model, int dataNum, int dataDim, vector<string> &outlayer, float *&outFeat, int &outFeatDim);

int cdnnReleaseModel(void **model);

int cdnnGetDataDim(void *model);

int cdnnGetLabelsDim(void *model);


#endif	/* CDNNSCORE_H_ */

