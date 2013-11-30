
#include <malloc.h>
#include <string.h>
#include <xmmintrin.h>
#include <stdio.h>
#include "matrix_ssemul.h"

static inline ushort min(ushort a, ushort b) {
    return a < b ? a : b;
}

/** 
 * @brief sse matrix mutiplication a*b->c, 16bit aligned
 * 
 * @param A a
 * @param TB b, column storage priority
 * @param C c
 * @param m the row of a
 * @param n the col of b
 * @param k the row of b
 * 
 * @return 0 
 */
int SSEMatrixMul(float *A, float *TB, float *C, int m, int n, int k) {
    V4SF rc;
    __m128 r;
    for (int i = 0; i < m; i++) {
        __m128 *left = (__m128 *)(A + i*k);
        for (int j = 0; j < n; j++) {
            __m128 *right = (__m128 *)(TB + j*k);
            __m128 *tleft = left;
            rc.v = _mm_setzero_ps();

            for (int t = 0; t < k; t += 4) {
                r = _mm_mul_ps(*tleft, *right);
                rc.v = _mm_add_ps(rc.v, r);
                tleft+=4;
                right+=4;
            }
            C[i*n+j] = rc.f[0] + rc.f[1] + rc.f[2] + rc.f[3];
        }
    }

    return 0;
}


int mulBlock16SSE(float *a, float *b, float *c, int h, int w, int d)
{
    bool flag = false;
    int i, j, k, alignd;
    float *aligna, *alignb;
    float s[64] __attribute__((align(16)));// 16*4 float
    
    __m128 ma0, ma1, ma2, ma3;
    __m128 mb0, mb1, mb2, mb3;
    __m128 mc0, mc1, mc2, mc3, mc4, mc5, mc6, mc7;  
    __m128 mc8, mc9, mc10, mc11, mc12, mc13, mc14, mc15;
    __m128 ms0, ms1, ms2, ms3, ms4, ms5, ms6, ms7;
    __m128 ms8, ms9, ms10, ms11, ms12, ms13, ms14, ms15;

    // rearrange memory
    aligna = a;
    alignb = b;
    alignd = d;
    if ((alignd & 0x3) != 0) {
        alignd = ((d + 3)>>2)<<2;
        aligna = (float*)memalign(16, h*alignd*sizeof(float));
        alignb = (float*)memalign(16, w*alignd*sizeof(float));
        j = d*sizeof(float);
        for (i = 0; i < h; i++) {
            k = i*alignd;
            memcpy(aligna+k, a+i*d, j);
            memset(aligna+k+d, 0, (alignd-d)*sizeof(float));
        }
        for (i = 0; i < w; i++) {
            k = i*alignd;
            memcpy(alignb+k, b+i*d, j);
            memset(alignb+k+d, 0, (alignd-d)*sizeof(float));
        }
        flag = true;
    } else if (((size_t)aligna & 0xF) != 0 || ((size_t)alignb & 0xF) != 0) {
        aligna = (float*)memalign(16, h*alignd*sizeof(float));
        alignb = (float*)memalign(16, w*alignd*sizeof(float));
        memcpy(aligna, a, h*alignd*sizeof(float));
        memcpy(alignb, b, w*alignd*sizeof(float));
        flag = true;
    }

    for (i = 0; i < h>>2<<2; i += 4) {
        for (j = 0; j < w>>2<<2; j += 4) {

            float *pa0 = (float*)aligna + i*alignd;
            float *pb0 = (float*)alignb + j*alignd;
            float *pa1 = pa0 + alignd;
            float *pb1 = pb0 + alignd;
            float *pa2 = pa0 + 2 * alignd;
            float *pb2 = pb0 + 2 * alignd;
            float *pa3 = pa0 + 3 * alignd;
            float *pb3 = pb0 + 3 * alignd;
            float *pc = (float*)c + i * w + j;

            memset(s, 0, 64 * sizeof(float));

            ms0 = _mm_xor_ps (ms0, ms0);
            ms1 = _mm_xor_ps (ms1, ms1);
            ms2 = _mm_xor_ps (ms2, ms2);
            ms3 = _mm_xor_ps (ms3, ms3);
            
            ms4 = _mm_xor_ps (ms4, ms4);
            ms5 = _mm_xor_ps (ms5, ms5);
            ms6 = _mm_xor_ps (ms6, ms6);
            ms7 = _mm_xor_ps (ms7, ms7);
            
            ms8 = _mm_xor_ps (ms8, ms8);
            ms9 = _mm_xor_ps (ms9, ms9);
            ms10= _mm_xor_ps (ms10, ms10);
            ms11= _mm_xor_ps (ms11, ms11);
            
            ms12= _mm_xor_ps (ms12, ms12);
            ms13= _mm_xor_ps (ms13, ms13);
            ms14= _mm_xor_ps (ms14, ms14);
            ms15= _mm_xor_ps (ms15, ms15);
        
            for (k = 0; k < alignd; k += 4) {
                ma0 = _mm_load_ps(pa0);
                ma1 = _mm_load_ps(pa1);
                ma2 = _mm_load_ps(pa2);
                ma3 = _mm_load_ps(pa3);
                
                mb0 = _mm_load_ps(pb0);
                mb1 = _mm_load_ps(pb1);
                mb2 = _mm_load_ps(pb2);
                mb3 = _mm_load_ps(pb3);
                
                mc0 = _mm_mul_ps(ma0, mb0);     
                mc1 = _mm_mul_ps(ma0, mb1);
                mc2 = _mm_mul_ps(ma0, mb2);
                mc3 = _mm_mul_ps(ma0, mb3);
                
                ms0 = _mm_add_ps(ms0, mc0); 
                ms1 = _mm_add_ps(ms1, mc1);
                ms2 = _mm_add_ps(ms2, mc2);
                ms3 = _mm_add_ps(ms3, mc3);
                
                mc4 = _mm_mul_ps(ma1, mb0);     
                mc5 = _mm_mul_ps(ma1, mb1);
                mc6 = _mm_mul_ps(ma1, mb2);
                mc7 = _mm_mul_ps(ma1, mb3);
                
                ms4 = _mm_add_ps(ms4, mc4); 
                ms5 = _mm_add_ps(ms5, mc5);
                ms6 = _mm_add_ps(ms6, mc6);
                ms7 = _mm_add_ps(ms7, mc7);
                
                mc8 = _mm_mul_ps(ma2, mb0);     
                mc9 = _mm_mul_ps(ma2, mb1);
                mc10= _mm_mul_ps(ma2, mb2);
                mc11= _mm_mul_ps(ma2, mb3);
                
                ms8 = _mm_add_ps(ms8, mc8); 
                ms9 = _mm_add_ps(ms9, mc9);
                ms10= _mm_add_ps(ms10, mc10);
                ms11= _mm_add_ps(ms11, mc11);
                
                mc12= _mm_mul_ps(ma3, mb0);     
                mc13= _mm_mul_ps(ma3, mb1);
                mc14= _mm_mul_ps(ma3, mb2);
                mc15= _mm_mul_ps(ma3, mb3);
                
                ms12= _mm_add_ps(ms12, mc12);   
                ms13= _mm_add_ps(ms13, mc13);
                ms14= _mm_add_ps(ms14, mc14);
                ms15= _mm_add_ps(ms15, mc15);
                

                pa0+=4; pb0+=4;
                pa1+=4; pb1+=4;
                pa2+=4; pb2+=4;
                pa3+=4; pb3+=4;
            }
            
            _mm_store_ps(s, ms0);   
            _mm_store_ps(s+4, ms1);
            _mm_store_ps(s+8, ms2);
            _mm_store_ps(s+12, ms3);
            _mm_store_ps(s+16, ms4);    
            _mm_store_ps(s+20, ms5);
            _mm_store_ps(s+24, ms6);
            _mm_store_ps(s+28, ms7);
        
            _mm_store_ps(s+32, ms8);    
            _mm_store_ps(s+36, ms9);
            _mm_store_ps(s+40, ms10);
            _mm_store_ps(s+44, ms11);
            _mm_store_ps(s+48, ms12);   
            _mm_store_ps(s+52, ms13);
            _mm_store_ps(s+56, ms14);
            _mm_store_ps(s+60, ms15);


            pc[0]   = s[0]+s[1]+s[2]+s[3];
            pc[1]   = s[4]+s[5]+s[6]+s[7];
            pc[2]   = s[8]+s[9]+s[10]+s[11];
            pc[3]   = s[12]+s[13]+s[14]+s[15];
            
            pc[w]   = s[16]+s[17]+s[18]+s[19];
            pc[w+1] = s[20]+s[21]+s[22]+s[23];
            pc[w+2] = s[24]+s[25]+s[26]+s[27];
            pc[w+3] = s[28]+s[29]+s[30]+s[31];
                            
            pc[2*w]     = s[32+0]+s[32+1]+s[32+2]+s[32+3];
            pc[2*w+1]   = s[32+4]+s[32+5]+s[32+6]+s[32+7];
            pc[2*w+2]   = s[32+8]+s[32+9]+s[32+10]+s[32+11];
            pc[2*w+3]   = s[32+12]+s[32+13]+s[32+14]+s[32+15];
            
            pc[3*w]     = s[32+16]+s[32+17]+s[32+18]+s[32+19];
            pc[3*w+1]   = s[32+20]+s[32+21]+s[32+22]+s[32+23];
            pc[3*w+2]   = s[32+24]+s[32+25]+s[32+26]+s[32+27];
            pc[3*w+3]   = s[32+28]+s[32+29]+s[32+30]+s[32+31];
            
        }
 
        for (j = w>>2<<2; j < w>>1<<1; j += 2) {
            float *pa0 = (float*)aligna + i*alignd;
            float *pb0 = (float*)alignb + j*alignd;
            float *pa1 = pa0 + alignd;
            float *pb1 = pb0 + alignd;
            float *pa2 = pa0 + 2 * alignd;
            float *pa3 = pa0 + 3 * alignd;
            float *pc = (float*)c + i * w + j;

            memset(s, 0, 32 * sizeof(float));

            ms0 = _mm_xor_ps (ms0, ms0);
            ms1 = _mm_xor_ps (ms1, ms1);
            ms2 = _mm_xor_ps (ms2, ms2);
            ms3 = _mm_xor_ps (ms3, ms3);
            
            ms4 = _mm_xor_ps (ms4, ms4);
            ms5 = _mm_xor_ps (ms5, ms5);
            ms6 = _mm_xor_ps (ms6, ms6);
            ms7 = _mm_xor_ps (ms7, ms7);
            
            for (k = 0; k < alignd; k += 4) {
                ma0 = _mm_load_ps(pa0);
                ma1 = _mm_load_ps(pa1);
                ma2 = _mm_load_ps(pa2);
                ma3 = _mm_load_ps(pa3);
                
                mb0 = _mm_load_ps(pb0);
                mb1 = _mm_load_ps(pb1);
                
                mc0 = _mm_mul_ps(ma0, mb0);     
                mc1 = _mm_mul_ps(ma0, mb1);
                
                ms0 = _mm_add_ps(ms0, mc0); 
                ms1 = _mm_add_ps(ms1, mc1);
                
                mc2 = _mm_mul_ps(ma1, mb0);     
                mc3 = _mm_mul_ps(ma1, mb1);
                
                ms2 = _mm_add_ps(ms2, mc2); 
                ms3 = _mm_add_ps(ms3, mc3);
                
                mc4 = _mm_mul_ps(ma2, mb0);     
                mc5 = _mm_mul_ps(ma2, mb1);
                
                ms4 = _mm_add_ps(ms4, mc4); 
                ms5 = _mm_add_ps(ms5, mc5);
                
                mc6 = _mm_mul_ps(ma3, mb0);     
                mc7 = _mm_mul_ps(ma3, mb1);
                
                ms6 = _mm_add_ps(ms6, mc6);   
                ms7 = _mm_add_ps(ms7, mc7);

                pa0+=4; pb0+=4;
                pa1+=4; pb1+=4;
                pa2+=4; pa3+=4;
            }
            
            _mm_store_ps(s, ms0);   
            _mm_store_ps(s+4, ms1);
            _mm_store_ps(s+8, ms2);
            _mm_store_ps(s+12, ms3);
            _mm_store_ps(s+16, ms4);    
            _mm_store_ps(s+20, ms5);
            _mm_store_ps(s+24, ms6);
            _mm_store_ps(s+28, ms7);

            pc[0]   = s[0]+s[1]+s[2]+s[3];
            pc[1]   = s[4]+s[5]+s[6]+s[7];

            pc[w]   = s[8]+s[9]+s[10]+s[11];
            pc[w+1] = s[12]+s[13]+s[14]+s[15];

            pc[2*w]     = s[16]+s[17]+s[18]+s[19];
            pc[2*w+1]   = s[20]+s[21]+s[22]+s[23];

            pc[3*w]     = s[24]+s[25]+s[26]+s[27];
            pc[3*w+1]   = s[28]+s[29]+s[30]+s[31];
        }
        for (j = w>>1<<1; j < w; j++) {
            float *pa0 = (float*)aligna + i*alignd;
            float *pb0 = (float*)alignb + j*alignd;
            float *pa1 = pa0 + alignd;
            float *pa2 = pa0 + 2 * alignd;
            float *pa3 = pa0 + 3 * alignd;
            float *pc = (float*)c + i * w + j;

            memset(s, 0, 16 * sizeof(float));

            ms0 = _mm_xor_ps (ms0, ms0);
            ms1 = _mm_xor_ps (ms1, ms1);
            ms2 = _mm_xor_ps (ms2, ms2);
            ms3 = _mm_xor_ps (ms3, ms3);
            
            for (k = 0; k < alignd; k += 4) {
                ma0 = _mm_load_ps(pa0);
                ma1 = _mm_load_ps(pa1);
                ma2 = _mm_load_ps(pa2);
                ma3 = _mm_load_ps(pa3);
                
                mb0 = _mm_load_ps(pb0);
                
                mc0 = _mm_mul_ps(ma0, mb0);     
                ms0 = _mm_add_ps(ms0, mc0); 
                
                mc1 = _mm_mul_ps(ma1, mb0);     
                ms1 = _mm_add_ps(ms1, mc1); 
                
                mc2 = _mm_mul_ps(ma2, mb0);     
                ms2 = _mm_add_ps(ms2, mc2); 
                
                mc3 = _mm_mul_ps(ma3, mb0);     
                ms3 = _mm_add_ps(ms3, mc3);   

                pa0+=4; pb0+=4;
                pa1+=4; pa2+=4; pa3+=4;
            }
            _mm_store_ps(s, ms0);   
            _mm_store_ps(s+4, ms1);
            _mm_store_ps(s+8, ms2);
            _mm_store_ps(s+12, ms3);

            pc[0]   = s[0]+s[1]+s[2]+s[3];
            pc[w]   = s[4]+s[5]+s[6]+s[7];
            pc[2*w]   = s[8]+s[9]+s[10]+s[11];
            pc[3*w] = s[12]+s[13]+s[14]+s[15];
        }
    }

    for (i = h>>2<<2; i < h>>1<<1; i += 2) {
        for (j = 0; j < w>>2<<2; j += 4) {

            float *pa0 = (float*)aligna + i*alignd;
            float *pb0 = (float*)alignb + j*alignd;
            float *pa1 = pa0 + alignd;
            float *pb1 = pb0 + alignd;
            float *pb2 = pb0 + 2 * alignd;
            float *pb3 = pb0 + 3 * alignd;
            float *pc = (float*)c + i * w + j;

            memset(s, 0, 32 * sizeof(float));

            ms0 = _mm_xor_ps (ms0, ms0);
            ms1 = _mm_xor_ps (ms1, ms1);
            ms2 = _mm_xor_ps (ms2, ms2);
            ms3 = _mm_xor_ps (ms3, ms3);
            
            ms4 = _mm_xor_ps (ms4, ms4);
            ms5 = _mm_xor_ps (ms5, ms5);
            ms6 = _mm_xor_ps (ms6, ms6);
            ms7 = _mm_xor_ps (ms7, ms7);
            
            for (k = 0; k < alignd; k += 4) {
                ma0 = _mm_load_ps(pa0);
                ma1 = _mm_load_ps(pa1);
                
                mb0 = _mm_load_ps(pb0);
                mb1 = _mm_load_ps(pb1);
                mb2 = _mm_load_ps(pb2);
                mb3 = _mm_load_ps(pb3);
                
                mc0 = _mm_mul_ps(ma0, mb0);     
                mc1 = _mm_mul_ps(ma0, mb1);
                ms0 = _mm_add_ps(ms0, mc0); 
                ms1 = _mm_add_ps(ms1, mc1);

                mc2 = _mm_mul_ps(ma0, mb2);
                mc3 = _mm_mul_ps(ma0, mb3);
                ms2 = _mm_add_ps(ms2, mc2);
                ms3 = _mm_add_ps(ms3, mc3);
                
                mc4 = _mm_mul_ps(ma1, mb0);     
                mc5 = _mm_mul_ps(ma1, mb1);
                ms4 = _mm_add_ps(ms4, mc4); 
                ms5 = _mm_add_ps(ms5, mc5);

                mc6 = _mm_mul_ps(ma1, mb2);
                mc7 = _mm_mul_ps(ma1, mb3);
                ms6 = _mm_add_ps(ms6, mc6);
                ms7 = _mm_add_ps(ms7, mc7);
                
                pa0+=4; pa1+=4;
                pb0+=4; pb1+=4;
                pb2+=4; pb3+=4;
            }
            
            _mm_store_ps(s, ms0);   
            _mm_store_ps(s+4, ms1);
            _mm_store_ps(s+8, ms2);
            _mm_store_ps(s+12, ms3);
            _mm_store_ps(s+16, ms4);    
            _mm_store_ps(s+20, ms5);
            _mm_store_ps(s+24, ms6);
            _mm_store_ps(s+28, ms7);
        
            pc[0]   = s[0]+s[1]+s[2]+s[3];
            pc[1]   = s[4]+s[5]+s[6]+s[7];
            pc[2]   = s[8]+s[9]+s[10]+s[11];
            pc[3]   = s[12]+s[13]+s[14]+s[15];
            
            pc[w]   = s[16]+s[17]+s[18]+s[19];
            pc[w+1] = s[20]+s[21]+s[22]+s[23];
            pc[w+2] = s[24]+s[25]+s[26]+s[27];
            pc[w+3] = s[28]+s[29]+s[30]+s[31];
        }
        
        for (j = w>>2<<2; j < w>>1<<1; j += 2) {
            float *pa0 = (float*)aligna + i*alignd;
            float *pb0 = (float*)alignb + j*alignd;
            float *pa1 = pa0 + alignd;
            float *pb1 = pb0 + alignd;
            float *pc = (float*)c + i * w + j;

            memset(s, 0, 16 * sizeof(float));

            ms0 = _mm_xor_ps (ms0, ms0);
            ms1 = _mm_xor_ps (ms1, ms1);
            ms2 = _mm_xor_ps (ms2, ms2);
            ms3 = _mm_xor_ps (ms3, ms3);
            
            for (k = 0; k < alignd; k += 4) {
                ma0 = _mm_load_ps(pa0);
                ma1 = _mm_load_ps(pa1);
                
                mb0 = _mm_load_ps(pb0);
                mb1 = _mm_load_ps(pb1);
                
                mc0 = _mm_mul_ps(ma0, mb0);     
                ms0 = _mm_add_ps(ms0, mc0); 

                mc1 = _mm_mul_ps(ma0, mb1);
                ms1 = _mm_add_ps(ms1, mc1);
                
                mc2 = _mm_mul_ps(ma1, mb0);     
                ms2 = _mm_add_ps(ms2, mc2); 

                mc3 = _mm_mul_ps(ma1, mb1);
                ms3 = _mm_add_ps(ms3, mc3);

                pa0+=4; pb0+=4;
                pa1+=4; pb1+=4;
            }
            
            _mm_store_ps(s, ms0);   
            _mm_store_ps(s+4, ms1);
            _mm_store_ps(s+8, ms2);
            _mm_store_ps(s+12, ms3);

            pc[0]   = s[0]+s[1]+s[2]+s[3];
            pc[1]   = s[4]+s[5]+s[6]+s[7];

            pc[w]   = s[8]+s[9]+s[10]+s[11];
            pc[w+1] = s[12]+s[13]+s[14]+s[15];
        }
        for (j = w>>1<<1; j < w; j++) {
            float *pa0 = (float*)aligna + i*alignd;
            float *pb0 = (float*)alignb + j*alignd;
            float *pa1 = pa0 + alignd;
            float *pc = (float*)c + i * w + j;

            memset(s, 0, 8 * sizeof(float));

            ms0 = _mm_xor_ps (ms0, ms0);
            ms1 = _mm_xor_ps (ms1, ms1);

            for (k = 0; k < alignd; k += 4) {
                ma0 = _mm_load_ps(pa0);
                ma1 = _mm_load_ps(pa1);
                mb0 = _mm_load_ps(pb0);
                
                mc0 = _mm_mul_ps(ma0, mb0);     
                ms0 = _mm_add_ps(ms0, mc0); 
                
                mc1 = _mm_mul_ps(ma1, mb0);     
                ms1 = _mm_add_ps(ms1, mc1); 
                
                pa0+=4; pb0+=4;
                pa1+=4;
            }
            _mm_store_ps(s, ms0);   
            _mm_store_ps(s+4, ms1);

            pc[0]   = s[0]+s[1]+s[2]+s[3];
            pc[w]   = s[4]+s[5]+s[6]+s[7];
        }
    }

    for (i = h>>1<<1; i < h; i++) {
        for (j = 0; j < w>>2<<2; j += 4) {

            float *pa0 = (float*)aligna + i*alignd;
            float *pb0 = (float*)alignb + j*alignd;
            float *pb1 = pb0 + alignd;
            float *pb2 = pb0 + 2 * alignd;
            float *pb3 = pb0 + 3 * alignd;
            float *pc = (float*)c + i * w + j;

            memset(s, 0, 16 * sizeof(float));

            ms0 = _mm_xor_ps (ms0, ms0);
            ms1 = _mm_xor_ps (ms1, ms1);
            ms2 = _mm_xor_ps (ms2, ms2);
            ms3 = _mm_xor_ps (ms3, ms3);

            for (k = 0; k < alignd; k += 4) {
                ma0 = _mm_load_ps(pa0);
                
                mb0 = _mm_load_ps(pb0);
                mb1 = _mm_load_ps(pb1);
                mb2 = _mm_load_ps(pb2);
                mb3 = _mm_load_ps(pb3);

                mc0 = _mm_mul_ps(ma0, mb0);     
                ms0 = _mm_add_ps(ms0, mc0); 

                mc1 = _mm_mul_ps(ma0, mb1);
                ms1 = _mm_add_ps(ms1, mc1);

                mc2 = _mm_mul_ps(ma0, mb2);
                ms2 = _mm_add_ps(ms2, mc2);

                mc3 = _mm_mul_ps(ma0, mb3);
                ms3 = _mm_add_ps(ms3, mc3);
                
                pa0+=4;
                pb0+=4; pb1+=4;
                pb2+=4; pb3+=4;
            }
            _mm_store_ps(s, ms0);   
            _mm_store_ps(s+4, ms1);
            _mm_store_ps(s+8, ms2);
            _mm_store_ps(s+12, ms3);
        
            pc[0]   = s[0]+s[1]+s[2]+s[3];
            pc[1]   = s[4]+s[5]+s[6]+s[7];
            pc[2]   = s[8]+s[9]+s[10]+s[11];
            pc[3]   = s[12]+s[13]+s[14]+s[15];
        }
        for (j = w>>2<<2; j < w>>1<<1; j += 2) {
            float *pa0 = (float*)aligna + i*alignd;
            float *pb0 = (float*)alignb + j*alignd;
            float *pb1 = pb0 + alignd;
            float *pc = (float*)c + i * w + j;

            memset(s, 0, 8 * sizeof(float));

            ms0 = _mm_xor_ps (ms0, ms0);
            ms1 = _mm_xor_ps (ms1, ms1);
            
            for (k = 0; k < alignd; k += 4) {
                ma0 = _mm_load_ps(pa0);
                
                mb0 = _mm_load_ps(pb0);
                mb1 = _mm_load_ps(pb1);
                
                mc0 = _mm_mul_ps(ma0, mb0);     
                ms0 = _mm_add_ps(ms0, mc0); 

                mc1 = _mm_mul_ps(ma0, mb1);
                ms1 = _mm_add_ps(ms1, mc1);
                
                pa0+=4; 
                pb0+=4; pb1+=4;
            }
            
            _mm_store_ps(s, ms0);   
            _mm_store_ps(s+4, ms1);

            pc[0]   = s[0]+s[1]+s[2]+s[3];
            pc[1]   = s[4]+s[5]+s[6]+s[7];
        }
        for (j = w>>1<<1; j < w; j++) {
            float *pa0 = (float*)aligna + i*alignd;
            float *pb0 = (float*)alignb + j*alignd;
            float *pc = (float*)c + i * w + j;

            memset(s, 0, 4 * sizeof(float));

            ms0 = _mm_xor_ps (ms0, ms0);

            for (k = 0; k < alignd; k += 4) {
                ma0 = _mm_load_ps(pa0);
                mb0 = _mm_load_ps(pb0);
                
                mc0 = _mm_mul_ps(ma0, mb0);     
                ms0 = _mm_add_ps(ms0, mc0); 
                
                pa0+=4; pb0+=4;
            }
            _mm_store_ps(s, ms0);   

            pc[0]   = s[0]+s[1]+s[2]+s[3];
        }
    }

    if (flag) {
        free(aligna); aligna = NULL;
        free(alignb); alignb = NULL;
    }

    return 0;
}

int cDense2CscAlign16(ushort rows, ushort cols, float *cMat, csc_t *&cscMat) {
    int entryNum = 0;
    int entryCol = 0;

    for (ushort i = 0; i < cols; i++) {
        entryCol = 0;
        for (ushort j = 0; j < rows; j++) {
            if (cMat[i*rows+j] != 0) {
                entryCol++;
            }
        }
        entryNum += (entryCol + 3) >> 2 << 2;
    }

    cscMat = (csc_t *)memalign(16, sizeof(csc_t));
    cscMat->rows = rows;
    cscMat->cols = cols;
    cscMat->entryNum = entryNum;
    char *mem = (char *)memalign(16, sizeof(int) * (cols+1) + sizeof(short) * entryNum + sizeof(float) * entryNum);
    cscMat->val = (float *)mem;
    cscMat->rptr = (ushort *)(mem + sizeof(float) * entryNum);
    cscMat->cptr = (int *)(cscMat->rptr + entryNum);

    memset(cscMat->rptr, 0, sizeof(short) * entryNum);
    memset(cscMat->val, 0, sizeof(float) * entryNum);

    int num = 0;
    for (ushort i = 0; i < cols; i++) {
        cscMat->cptr[i] = num;
        for (ushort j = 0; j < rows; j++) {
            if (cMat[i*rows+j] != 0) {
                cscMat->val[num] = cMat[i*rows+j];
                cscMat->rptr[num] = j;
                num++;
            }
        }
        num = (num + 3) >> 2 << 2;
    }
    cscMat->cptr[cols] = num;
    return 0;
}

int releaseCscMat(csc_t **cscMatPtr) {
    if (cscMatPtr == NULL || *cscMatPtr == NULL) {
        return 0;
    }
    free((char*)(*cscMatPtr)->val);
    free(*cscMatPtr);
    *cscMatPtr = NULL;
    return 0;
}

int rMatMulCscMatSSE8(float *rMat, csc_t *cscMat, float *resMat, ushort h, ushort w, ushort d) {
    if (w % 2 != 0) {
        fprintf(stderr, "w must be divided by 2.\n");
        return -1;
    }
    if (w != cscMat->cols) {
        fprintf(stderr, "error w in cscMat.\n");
        return -1;
    }
    if (d != cscMat->rows) {
        fprintf(stderr, "error d in cscMat.\n");
        return -1;
    }
    int *cptr = cscMat->cptr;
    ushort *rptr = cscMat->rptr;
    float *valptr = cscMat->val;

    __m128 ma0, ma1, ma2, ma3, ma4, ma5, ma6, ma7;
    __m128 mb0, mb1;
    __m128 mc0, mc1, mc2, mc3, mc4, mc5, mc6, mc7;
    __m128 ms0, ms1, ms2, ms3, ms4, ms5, ms6, ms7;
    float s[32] __attribute__((align(16)));
    for (ushort i = 0; i < h>>2<<2; i+=4) {
        float *pa0 = rMat + i * d;
        float *pa1 = pa0 + d;
        float *pa2 = pa0 + 2 * d;
        float *pa3 = pa0 + 3 * d;
        for (ushort j = 0; j < w; j+=2) {
            float *pb0 = valptr + cptr[j];
            float *pb1 = valptr + cptr[j+1];
            ushort *pi0 = rptr + cptr[j];
            ushort *pi1 = rptr + cptr[j+1];
            float *pc = (float*)resMat + i * w + j;

            ms0 = _mm_xor_ps(ms0, ms0);
            ms1 = _mm_xor_ps(ms1, ms1);
            ms2 = _mm_xor_ps(ms2, ms2);
            ms3 = _mm_xor_ps(ms3, ms3);
            ms4 = _mm_xor_ps(ms4, ms4);
            ms5 = _mm_xor_ps(ms5, ms5);
            ms6 = _mm_xor_ps(ms6, ms6);
            ms7 = _mm_xor_ps(ms7, ms7);
            ushort len1 = cptr[j+1]-cptr[j];
            ushort len2 = cptr[j+2]-cptr[j+1];
            ushort minLoop = min(len1, len2);
            for (ushort k = 0; k < minLoop; k+=4) {
                mb0 = _mm_load_ps(pb0);
                mb1 = _mm_load_ps(pb1);

                ma0 = _mm_set_ps(pa0[*(pi0+3)], pa0[*(pi0+2)], pa0[*(pi0+1)], pa0[*pi0]);
                mc0 = _mm_mul_ps(ma0, mb0);
                ms0 = _mm_add_ps(ms0, mc0);

                ma1 = _mm_set_ps(pa1[*(pi0+3)], pa1[*(pi0+2)], pa1[*(pi0+1)], pa1[*pi0]);
                mc1 = _mm_mul_ps(ma1, mb0);
                ms1 = _mm_add_ps(ms1, mc1);

                ma2 = _mm_set_ps(pa2[*(pi0+3)], pa2[*(pi0+2)], pa2[*(pi0+1)], pa2[*pi0]);
                mc2 = _mm_mul_ps(ma2, mb0);
                ms2 = _mm_add_ps(ms2, mc2);

                ma3 = _mm_set_ps(pa3[*(pi0+3)], pa3[*(pi0+2)], pa3[*(pi0+1)], pa3[*pi0]);
                mc3 = _mm_mul_ps(ma3, mb0);
                ms3 = _mm_add_ps(ms3, mc3);

                ma4 = _mm_set_ps(pa0[*(pi1+3)], pa0[*(pi1+2)], pa0[*(pi1+1)], pa0[*pi1]);
                mc4 = _mm_mul_ps(ma4, mb1);
                ms4 = _mm_add_ps(ms4, mc4);

                ma5 = _mm_set_ps(pa1[*(pi1+3)], pa1[*(pi1+2)], pa1[*(pi1+1)], pa1[*pi1]);
                mc5 = _mm_mul_ps(ma5, mb1);
                ms5 = _mm_add_ps(ms5, mc5);

                ma6 = _mm_set_ps(pa2[*(pi1+3)], pa2[*(pi1+2)], pa2[*(pi1+1)], pa2[*pi1]);
                mc6 = _mm_mul_ps(ma6, mb1);
                ms6 = _mm_add_ps(ms6, mc6);

                ma7 = _mm_set_ps(pa3[*(pi1+3)], pa3[*(pi1+2)], pa3[*(pi1+1)], pa3[*pi1]);
                mc7 = _mm_mul_ps(ma7, mb1);
                ms7 = _mm_add_ps(ms7, mc7);

                pb0 += 4; pb1 += 4;
                pi0 += 4; pi1 += 4;
            }
            for (ushort k = minLoop; k < len1; k += 4) {
                mb0 = _mm_load_ps(pb0);

                ma0 = _mm_set_ps(pa0[*(pi0+3)], pa0[*(pi0+2)], pa0[*(pi0+1)], pa0[*pi0]);
                mc0 = _mm_mul_ps(ma0, mb0);
                ms0 = _mm_add_ps(ms0, mc0);

                ma1 = _mm_set_ps(pa1[*(pi0+3)], pa1[*(pi0+2)], pa1[*(pi0+1)], pa1[*pi0]);
                mc1 = _mm_mul_ps(ma1, mb0);
                ms1 = _mm_add_ps(ms1, mc1);

                ma2 = _mm_set_ps(pa2[*(pi0+3)], pa2[*(pi0+2)], pa2[*(pi0+1)], pa2[*pi0]);
                mc2 = _mm_mul_ps(ma2, mb0);
                ms2 = _mm_add_ps(ms2, mc2);

                ma3 = _mm_set_ps(pa3[*(pi0+3)], pa3[*(pi0+2)], pa3[*(pi0+1)], pa3[*pi0]);
                mc3 = _mm_mul_ps(ma3, mb0);
                ms3 = _mm_add_ps(ms3, mc3);

                pb0 += 4;
                pi0 += 4;
            }
            for (ushort k = minLoop; k < len2; k += 4) {
                mb1 = _mm_load_ps(pb1);

                ma4 = _mm_set_ps(pa0[*(pi1+3)], pa0[*(pi1+2)], pa0[*(pi1+1)], pa0[*pi1]);
                mc4 = _mm_mul_ps(ma4, mb1);
                ms4 = _mm_add_ps(ms4, mc4);

                ma5 = _mm_set_ps(pa1[*(pi1+3)], pa1[*(pi1+2)], pa1[*(pi1+1)], pa1[*pi1]);
                mc5 = _mm_mul_ps(ma5, mb1);
                ms5 = _mm_add_ps(ms5, mc5);

                ma6 = _mm_set_ps(pa2[*(pi1+3)], pa2[*(pi1+2)], pa2[*(pi1+1)], pa2[*pi1]);
                mc6 = _mm_mul_ps(ma6, mb1);
                ms6 = _mm_add_ps(ms6, mc6);

                ma7 = _mm_set_ps(pa3[*(pi1+3)], pa3[*(pi1+2)], pa3[*(pi1+1)], pa3[*pi1]);
                mc7 = _mm_mul_ps(ma7, mb1);
                ms7 = _mm_add_ps(ms7, mc7);

                pb1 += 4;
                pi1 += 4;
            }
            _mm_store_ps(s, ms0);
            _mm_store_ps(s+4, ms1);
            _mm_store_ps(s+8, ms2);
            _mm_store_ps(s+12, ms3);
            _mm_store_ps(s+16, ms4);
            _mm_store_ps(s+20, ms5);
            _mm_store_ps(s+24, ms6);
            _mm_store_ps(s+28, ms7);

            pc[0]   = s[0]+s[1]+s[2]+s[3];
            pc[w]   = s[4]+s[5]+s[6]+s[7];
            pc[2*w]   = s[8]+s[9]+s[10]+s[11];
            pc[3*w]   = s[12]+s[13]+s[14]+s[15];
            pc[1]   = s[16]+s[17]+s[18]+s[19];
            pc[w+1] = s[20]+s[21]+s[22]+s[23];
            pc[2*w+1]   = s[24]+s[25]+s[26]+s[27];
            pc[3*w+1] = s[28]+s[29]+s[30]+s[31];
        }
    }
    for (ushort i = h>>2<<2; i < h>>1<<1; i+=2) {
        float *pa0 = rMat + i * d;
        float *pa1 = pa0 + d;
        for (ushort j = 0; j < w; j+=2) {
            float *pb0 = valptr + cptr[j];
            float *pb1 = valptr + cptr[j+1];
            ushort *pi0 = rptr + cptr[j];
            ushort *pi1 = rptr + cptr[j+1];
            float *pc = (float*)resMat + i * w + j;

            ms0 = _mm_xor_ps(ms0, ms0);
            ms1 = _mm_xor_ps(ms1, ms1);
            ms2 = _mm_xor_ps(ms2, ms2);
            ms3 = _mm_xor_ps(ms3, ms3);
            ushort len1 = cptr[j+1]-cptr[j];
            ushort len2 = cptr[j+2]-cptr[j+1];
            ushort minLoop = min(len1, len2);
            for (ushort k = 0; k < minLoop; k+=4) {
                mb0 = _mm_load_ps(pb0);
                mb1 = _mm_load_ps(pb1);

                ma0 = _mm_set_ps(pa0[*(pi0+3)], pa0[*(pi0+2)], pa0[*(pi0+1)], pa0[*pi0]);
                mc0 = _mm_mul_ps(ma0, mb0);
                ms0 = _mm_add_ps(ms0, mc0);

                ma1 = _mm_set_ps(pa1[*(pi0+3)], pa1[*(pi0+2)], pa1[*(pi0+1)], pa1[*pi0]);
                mc1 = _mm_mul_ps(ma1, mb0);
                ms1 = _mm_add_ps(ms1, mc1);

                ma2 = _mm_set_ps(pa0[*(pi1+3)], pa0[*(pi1+2)], pa0[*(pi1+1)], pa0[*pi1]);
                mc2 = _mm_mul_ps(ma2, mb1);
                ms2 = _mm_add_ps(ms2, mc2);

                ma3 = _mm_set_ps(pa1[*(pi1+3)], pa1[*(pi1+2)], pa1[*(pi1+1)], pa1[*pi1]);
                mc3 = _mm_mul_ps(ma3, mb1);
                ms3 = _mm_add_ps(ms3, mc3);

                pb0 += 4; pb1 += 4;
                pi0 += 4; pi1 += 4;
            }
            for (ushort k = minLoop; k < len1; k += 4) {
                mb0 = _mm_load_ps(pb0);

                ma0 = _mm_set_ps(pa0[*(pi0+3)], pa0[*(pi0+2)], pa0[*(pi0+1)], pa0[*pi0]);
                mc0 = _mm_mul_ps(ma0, mb0);
                ms0 = _mm_add_ps(ms0, mc0);

                ma1 = _mm_set_ps(pa1[*(pi0+3)], pa1[*(pi0+2)], pa1[*(pi0+1)], pa1[*pi0]);
                mc1 = _mm_mul_ps(ma1, mb0);
                ms1 = _mm_add_ps(ms1, mc1);

                pb0 += 4;
                pi0 += 4;
            }
            for (ushort k = minLoop; k < len2; k += 4) {
                mb1 = _mm_load_ps(pb1);

                ma2 = _mm_set_ps(pa0[*(pi1+3)], pa0[*(pi1+2)], pa0[*(pi1+1)], pa0[*pi1]);
                mc2 = _mm_mul_ps(ma2, mb1);
                ms2 = _mm_add_ps(ms2, mc2);

                ma3 = _mm_set_ps(pa1[*(pi1+3)], pa1[*(pi1+2)], pa1[*(pi1+1)], pa1[*pi1]);
                mc3 = _mm_mul_ps(ma3, mb1);
                ms3 = _mm_add_ps(ms3, mc3);

                pb1 += 4;
                pi1 += 4;
            }
            _mm_store_ps(s, ms0);
            _mm_store_ps(s+4, ms1);
            _mm_store_ps(s+8, ms2);
            _mm_store_ps(s+12, ms3);

            pc[0]   = s[0]+s[1]+s[2]+s[3];
            pc[w]   = s[4]+s[5]+s[6]+s[7];
            pc[1]   = s[8]+s[9]+s[10]+s[11];
            pc[w+1] = s[12]+s[13]+s[14]+s[15];
        }
    }
    for (ushort i = h>>1<<1; i < h; i+=1) {
        float *pa0 = rMat + i * d;
        for (ushort j = 0; j < w; j+=2) {
            float *pb0 = valptr + cptr[j];
            float *pb1 = valptr + cptr[j+1];
            ushort *pi0 = rptr + cptr[j];
            ushort *pi1 = rptr + cptr[j+1];
            float *pc = (float*)resMat + i * w + j;

            ms0 = _mm_xor_ps(ms0, ms0);
            ms1 = _mm_xor_ps(ms1, ms1);
            ushort len1 = cptr[j+1]-cptr[j];
            ushort len2 = cptr[j+2]-cptr[j+1];
            ushort minLoop = min(len1, len2);
            for (ushort k = 0; k < minLoop; k+=4) {
                mb0 = _mm_load_ps(pb0);
                mb1 = _mm_load_ps(pb1);

                ma0 = _mm_set_ps(pa0[*(pi0+3)], pa0[*(pi0+2)], pa0[*(pi0+1)], pa0[*pi0]);
                mc0 = _mm_mul_ps(ma0, mb0);
                ms0 = _mm_add_ps(ms0, mc0);

                ma1 = _mm_set_ps(pa0[*(pi1+3)], pa0[*(pi1+2)], pa0[*(pi1+1)], pa0[*pi1]);
                mc1 = _mm_mul_ps(ma1, mb1);
                ms1 = _mm_add_ps(ms1, mc1);

                pb0 += 4; pb1 += 4;
                pi0 += 4; pi1 += 4;
            }
            for (ushort k = minLoop; k < len1; k += 4) {
                mb0 = _mm_load_ps(pb0);

                ma0 = _mm_set_ps(pa0[*(pi0+3)], pa0[*(pi0+2)], pa0[*(pi0+1)], pa0[*pi0]);
                mc0 = _mm_mul_ps(ma0, mb0);
                ms0 = _mm_add_ps(ms0, mc0);

                pb0 += 4;
                pi0 += 4;
            }
            for (ushort k = minLoop; k < len2; k += 4) {
                mb1 = _mm_load_ps(pb1);

                ma1 = _mm_set_ps(pa0[*(pi1+3)], pa0[*(pi1+2)], pa0[*(pi1+1)], pa0[*pi1]);
                mc1 = _mm_mul_ps(ma1, mb1);
                ms1 = _mm_add_ps(ms1, mc1);

                pb1 += 4;
                pi1 += 4;
            }
            _mm_store_ps(s, ms0);
            _mm_store_ps(s+4, ms1);

            pc[0]   = s[0]+s[1]+s[2]+s[3];
            pc[1]   = s[4]+s[5]+s[6]+s[7];
        }
    }

    return 0;
}

int vecPairProduct16SSE(float *a, float *b, float *c, int n, int d)
{
    bool flag = false;
    int i, j, k, alignd;
    float *aligna, *alignb;
    float s[16] __attribute__((align(16)));// 16*4 float
    
    __m128 ma0, ma1, ma2, ma3;
    __m128 mb0, mb1, mb2, mb3;
    __m128 mc0, mc1, mc2, mc3, mc4, mc5, mc6, mc7;  
    __m128 mc8, mc9, mc10, mc11, mc12, mc13, mc14, mc15;
    __m128 ms0, ms1, ms2, ms3, ms4, ms5, ms6, ms7;
    __m128 ms8, ms9, ms10, ms11, ms12, ms13, ms14, ms15;

    // rearrange memory
    aligna = a;
    alignb = b;
    alignd = d;
    if ((alignd & 0x3) != 0) {
        alignd = ((d + 3)>>2)<<2;
        aligna = (float*)memalign(16, n*alignd*sizeof(float));
        alignb = (float*)memalign(16, n*alignd*sizeof(float));
        j = d*sizeof(float);
        for (i = 0; i < n; i++) {
            k = i*alignd;
            memcpy(aligna+k, a+i*d, j);
            memset(aligna+k+d, 0, (alignd-d)*sizeof(float));
        }
        for (i = 0; i < n; i++) {
            k = i*alignd;
            memcpy(alignb+k, b+i*d, j);
            memset(alignb+k+d, 0, (alignd-d)*sizeof(float));
        }
        flag = true;
    } else if (((size_t)aligna & 0xF) != 0 || ((size_t)alignb & 0xF) != 0) {
        aligna = (float*)memalign(16, n*alignd*sizeof(float));
        alignb = (float*)memalign(16, n*alignd*sizeof(float));
        memcpy(aligna, a, n*alignd*sizeof(float));
        memcpy(alignb, b, n*alignd*sizeof(float));
        flag = true;
    }

    for (i = 0; i < n>>2<<2; i += 4) {
        float *pa0 = (float*)aligna + i*alignd;
        float *pb0 = (float*)alignb + i*alignd;
        float *pa1 = pa0 + alignd;
        float *pb1 = pb0 + alignd;
        float *pa2 = pa0 + 2 * alignd;
        float *pb2 = pb0 + 2 * alignd;
        float *pa3 = pa0 + 3 * alignd;
        float *pb3 = pb0 + 3 * alignd;
        float *pc = (float*)c + i;

        memset(s, 0, 16 * sizeof(float));

        ms0 = _mm_xor_ps (ms0, ms0);
        ms1 = _mm_xor_ps (ms1, ms1);
        ms2 = _mm_xor_ps (ms2, ms2);
        ms3 = _mm_xor_ps (ms3, ms3);

        for (k = 0; k < alignd; k += 4) {
            ma0 = _mm_load_ps(pa0);
            mb0 = _mm_load_ps(pb0);
            mc0 = _mm_mul_ps(ma0, mb0);     
            ms0 = _mm_add_ps(ms0, mc0); 

            ma1 = _mm_load_ps(pa1);
            mb1 = _mm_load_ps(pb1);
            mc1 = _mm_mul_ps(ma1, mb1);
            ms1 = _mm_add_ps(ms1, mc1);

            ma2 = _mm_load_ps(pa2);
            mb2 = _mm_load_ps(pb2);
            mc2 = _mm_mul_ps(ma2, mb2);
            ms2 = _mm_add_ps(ms2, mc2);

            ma3 = _mm_load_ps(pa3);
            mb3 = _mm_load_ps(pb3);
            mc3 = _mm_mul_ps(ma3, mb3);
            ms3 = _mm_add_ps(ms3, mc3);


            pa0+=4; pb0+=4;
            pa1+=4; pb1+=4;
            pa2+=4; pb2+=4;
            pa3+=4; pb3+=4;
        }

        _mm_store_ps(s, ms0);   
        _mm_store_ps(s+4, ms1);
        _mm_store_ps(s+8, ms2);
        _mm_store_ps(s+12, ms3);

        pc[0]   = s[0]+s[1]+s[2]+s[3];
        pc[1]   = s[4]+s[5]+s[6]+s[7];
        pc[2]   = s[8]+s[9]+s[10]+s[11];
        pc[3]   = s[12]+s[13]+s[14]+s[15];
    }

    for (i = n>>2<<2; i < n>>1<<1; i += 2) {
        float *pa0 = (float*)aligna + i*alignd;
        float *pb0 = (float*)alignb + i*alignd;
        float *pa1 = pa0 + alignd;
        float *pb1 = pb0 + alignd;
        float *pc = (float*)c + i;

        memset(s, 0, 8 * sizeof(float));

        ms0 = _mm_xor_ps (ms0, ms0);
        ms1 = _mm_xor_ps (ms1, ms1);

        for (k = 0; k < alignd; k += 4) {
            ma0 = _mm_load_ps(pa0);
            mb0 = _mm_load_ps(pb0);
            mc0 = _mm_mul_ps(ma0, mb0);     
            ms0 = _mm_add_ps(ms0, mc0); 

            ma1 = _mm_load_ps(pa1);
            mb1 = _mm_load_ps(pb1);
            mc1 = _mm_mul_ps(ma1, mb1);
            ms1 = _mm_add_ps(ms1, mc1);

            pa0+=4; pa1+=4;
            pb0+=4; pb1+=4;
        }

        _mm_store_ps(s, ms0);   
        _mm_store_ps(s+4, ms1);

        pc[0]   = s[0]+s[1]+s[2]+s[3];
        pc[1]   = s[4]+s[5]+s[6]+s[7];
    }

    for (i = n>>1<<1; i < n; i++) {
        float *pa0 = (float*)aligna + i*alignd;
        float *pb0 = (float*)alignb + i*alignd;
        float *pc = (float*)c + i;

        memset(s, 0, 4 * sizeof(float));

        ms0 = _mm_xor_ps (ms0, ms0);

        for (k = 0; k < alignd; k += 4) {
            ma0 = _mm_load_ps(pa0);
            mb0 = _mm_load_ps(pb0);
            mc0 = _mm_mul_ps(ma0, mb0);     
            ms0 = _mm_add_ps(ms0, mc0); 

            pa0+=4;
            pb0+=4;
        }
        _mm_store_ps(s, ms0);   

        pc[0]   = s[0]+s[1]+s[2]+s[3];
    }

    if (flag) {
        free(aligna); aligna = NULL;
        free(alignb); alignb = NULL;
    }

    return 0;
}


