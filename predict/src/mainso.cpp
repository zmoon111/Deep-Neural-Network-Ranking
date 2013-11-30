/*
 *标准so文件，输出6个粒度的分词结果
 */

#include <stdio.h>

#include <map>
#include <string>
#include <sstream>
#include <iostream>
#include "ul_conf.h"
#include <sys/time.h>
#include "cdnn_score.h"

using namespace std;
#ifdef __cplusplus
extern "C" {

    typedef map <string,string> hmap;

    typedef struct _thread_src{
        int data_num;
        float* data;
        float* pout;
    }thread_src;

    void *model = NULL;
    int data_dim = 0, labels_dim = 0;
     
    char g_model_file[256] = "\0";
    void sub_time(struct timeval *tv1, struct timeval *tv2, struct timeval *ret);

    int main_thread_init(Ul_confdata* conf)
    {
            
        if(ul_getconfstr(conf,"model_path",g_model_file) != 1)
        {
            fprintf(stderr,"read model_dir error!\n");
            return -1;
        }
    
            
        fprintf(stderr,"Model_Dir=%s\n",g_model_file);
        if(cdnnInitModel(model_path, model) != 0)
        {
            fprintf(stderr,"model init failed. Filename=%s\n",g_model_file);
            exit(-1);
        }
        
        data_dim = cdnnGetDataDim(model);
        labelsDim = cdnnGetLabelsDim(model);

        return 0;
    }



    void main_thread_des(void * arg)
    {   
        cdnnReleaseModel(&model);
        return;
    }

    void * thread_resource_init(int count , int arg[])
    {
        thread_src* p = NULL;
        p = (thread_src*)calloc(1, sizeof(thread_src));
        if(p == NULL)
        {
            fprintf(stderr,"error: thread_src* calloc error.\n");
            return NULL;
        }
        memset(p, 0, sizeof(thread_src));

        int data_num = p->data_num;
        p->data = (float *)memalign(16, sizeof(float) * data_num * data_dim);
        p->prob = (float *)memalign(16, sizeof(float) * data_num * labels_dim);
        return p;
    }

    void thread_resource_des(void * arg)
    {
        thread_src* p = NULL;
        p = (thread_src*)arg;

        if(NULL == p)
            return ;
        
        free(p->data);
        free(p->prob);
        if(p)
        {
            free (p);
        }
        p = NULL;

        return;
    }

    void * thread_run(char* line,void * parg, int count, void *rarg[], int * status, struct timeval* t3)
    {
        thread_src* p = NULL;
        p = (thread_src*)parg;
        if(NULL == p)
        {
            return p;
        }

        struct timeval t1;
        struct timeval t2;    
        (*t3).tv_sec = 0;    
        (*t3).tv_usec = 0;

        gettimeofday(&t1, NULL);

        cdnnScore(p->data, model, p->data_num, data_dim, p->probs);  

        gettimeofday(&t2, NULL);
        sub_time(&t2,&t1,&t3[0]);

        return p;
    }

    int filter_query(char* query)
    {
    }   

    int process_result(void * arg, hmap * result,void * selfswitch )
    {
        thread_src* p = NULL;
        p = (thread_src*)arg;


        return 0;
    }


    void sub_time(struct timeval *tv1, struct timeval *tv2, struct timeval *ret) 
    {  
        if (tv1->tv_usec >= tv2->tv_usec)   
        {       
            ret->tv_sec = tv1->tv_sec - tv2->tv_sec;        
            ret->tv_usec = tv1->tv_usec - tv2->tv_usec; 
        }   
        else    
        {       
            ret->tv_sec = tv1->tv_sec - tv2->tv_sec - 1;        
            ret->tv_usec = 1000000 + tv1->tv_usec - tv2->tv_usec;   
        }
    }



#endif
