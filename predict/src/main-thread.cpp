#include <assert.h>
#include <sys/time.h>
#include <vector>
#include <malloc.h>
#include <pthread.h>
#include <stdlib.h>
#include "cdnn_score.h"

using namespace std;

int data_num, data_dim, labels_dim;
float *data = NULL;
void *model = NULL;
void *core_func(void *argv) {
    float *probs = (float *)memalign(16, data_num * labels_dim * sizeof(float));
    cdnnScore(data, model, data_num, data_dim, probs);  
    free(probs);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: ./main model_path data_path thread_num.\n");
        return -1;
    }
    char *model_path = argv[1];
    char *data_path = argv[2];
    int thread_num = atoi(argv[3]);

    if (cdnnInitModel(model_path, model) != 0) {
        fprintf(stderr, "model init error.\n");
        return -1;
    }
    
    data_dim = cdnnGetDataDim(model);
    labels_dim = cdnnGetLabelsDim(model);

    FILE *fR = fopen(data_path, "rb");
    if (fR == NULL) {
        return -1;
    }

    fseek(fR, 0, SEEK_END);
    data_num = ftell(fR) / (data_dim * sizeof(float));
    fseek(fR, 0, SEEK_SET);
    data = (float *)memalign(16, data_num * data_dim * sizeof(float));
    for (int i = 0; i < data_num; i++) {
        fread(data+i*data_dim, sizeof(float), data_dim, fR);
    }
    fclose(fR);

    timeval starttime,endtime;
    gettimeofday(&starttime,0);

    pthread_t *thp_core = (pthread_t *)calloc(thread_num, sizeof(pthread_t));

    for (int i = 0; i < thread_num; i++)
    {
        pthread_create(&(thp_core[i]), NULL, core_func, NULL);
    }

    for (int i = 0; i < thread_num; i++)
    {
        pthread_join(thp_core[i], NULL);
    }

    free(thp_core);
    thp_core = NULL;

    gettimeofday(&endtime,0);
    float timeuse = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
    printf("%dthreads: conv-timeuse:%fms\n", thread_num, timeuse / 1000.0);


    free(data);
    data = NULL;

    cdnnReleaseModel(&model);
    return 0;
}

