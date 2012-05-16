#ifndef __VOC_HOG_H__
#define __VOC_HOG_H__

extern int voc_hog_transfer_image(float* h_pImg, int width, int height);

extern void voc_hog_resize_image(int width, int height, int res_wid, int res_hei, int oct);

extern void voc_hog_debug_resize_image(int width, int height, int res_wid, int res_hei, int oct, float* res_img);

extern void voc_hog_set_octref(int ref_dimx, int ref_dimy, int oct);

extern int voc_hog_initialize();

extern int voc_hog_finalize();

extern int voc_hog_get_descriptor(int width, int height, int bPad,
						int out_dim, float scale,
						int sbin, float* h_pDescriptor);

extern int voc_hog_release_image();


#endif
