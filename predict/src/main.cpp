#include <assert.h>
#include <sys/time.h>
#include <vector>
#include <malloc.h>
#include "cdnn_score.h"

using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: ./main model_path data_path.\n");
        return -1;
    }
    char *model_path = argv[1];
    char *data_path = argv[2];
    void *model = NULL;
    if (cdnnInitModel(model_path, model) != 0) {
        fprintf(stderr, "model init error.\n");
        return -1;
    }

    int data_dim = cdnnGetDataDim(model);
    int labels_dim = cdnnGetLabelsDim(model);

    FILE *fR = fopen(data_path, "rb");
    if (fR == NULL) {
        return -1;
    }

    fseek(fR, 0, SEEK_END);
    int data_num = ftell(fR) / (data_dim * sizeof(float));
    fseek(fR, 0, SEEK_SET);
    float *data = (float *)malloc(data_num * data_dim * sizeof(float));
    float *probs = (float *)malloc(data_num * labels_dim * sizeof(float));
    for (int i = 0; i < data_num; i++) {
        fread(data+i*data_dim, sizeof(float), data_dim, fR);
    }
    fclose(fR);
    timeval starttime,endtime;
    gettimeofday(&starttime,0);

    cdnnScore(data, model, data_num, data_dim, probs);
    gettimeofday(&endtime,0);
    float timeuse = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
    printf("conv-timeuse:%fms\n", timeuse / 1000.0);
    printf("result is:\n");

    for (int i = 0; i < data_num; i++) {
        for (int j = 0; j < labels_dim; j++) {
            printf("%f ", probs[i*labels_dim+j]);
        }
        printf("\n");
    }

    //---feature extraction---
    float *feat = NULL;
    int featNum = 0;
    vector<string> outLayerName;
    char layerName[][256] = {"pool5", "fc6", "fc7", "fc8"};
    outLayerName.push_back(string(layerName[0]));
    outLayerName.push_back(string(layerName[2]));
    if (0 != cdnnFeatExtract(data, model, data_num, data_dim, outLayerName, feat, featNum)) {
        fprintf(stderr, "feature extract error.\n");
        return -1;
    }
    free(feat);
    //---feature extraction---

    free(data);
    data = NULL;
    free(probs);
    probs = NULL;

    cdnnReleaseModel(&model);
    return 0;
}

