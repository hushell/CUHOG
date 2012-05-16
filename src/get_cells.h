#ifndef __GRADIENTS_H__
#define __GRADIENTS_H__

int voc_prepare_image(const float* h_pImg, int width, int height);                                      

extern __host__ int voc_compute_gradients( int width, int height, int sbin,
                                int blocks_0, int blocks_1,
								int padX, int padY, float* d_pHist);

int voc_prepare_image2(const float* h_pImg, int width, int height, int chan);
int voc_destroy_image2();
int voc_prepare_image3(float* h_pImg, int width, int height);
int voc_destroy_image3();

extern __host__ void voc_resize_image(int width, int height, int res_wid, int res_hei, int oct);
extern __host__ void voc_set_octref(int ref_dimx, int ref_dimy, int oct);
extern __host__ void voc_debug_resize_image(int width, int height, int res_wid, int res_hei, int oct, float* res_img);

#endif
