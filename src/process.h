#ifndef __PROCESS_H__
#define __PROCESS_H__

typedef float* FloatPtr;

extern "C" void process(float* im, int dimy, int dimx, int sbin, 
    float* feat, int hy, int hx, int hz);

extern "C" void process_all_scales(float* im, int dimy, int dimx, int sbin, 
    FloatPtr* featArr, int maxoct, int interval, int* eleSize/*, float* res_img*/);

extern "C" void debug_resize(float* im, int height, int width, int res_dimy, int res_dimx, float* res_img);

#endif

