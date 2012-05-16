#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <assert.h>

#include "voc_hog.h"
#include "global.h"
#include "process.h"

using namespace std;

void process(float* im, int dimy, int dimx, int sbin, 
    float* feat, int hy, int hx, int hz)
{
    //vocHOGManager hog; // constructor and deconstructor

    int blocks[2];
    blocks[0] = (int)round((float)dimy/(float)sbin);
    blocks[1] = (int)round((float)dimx/(float)sbin);

    int out[3];
    out[0] = max(blocks[0]-2, 0);
    out[1] = max(blocks[1]-2, 0);
    out[2] = 31;
    if (hy!=out[0] || hx!=out[1] || hz!=out[2])
    {
        printf("Error in hog shape\n");
        return;
    }

    if( voc_hog_transfer_image(im, dimx, dimy) ) {
		return;
	}

    voc_hog_resize_image(dimx, dimy, dimx, dimy, 0);

    const int dim = out[0]*out[1]*out[2];
	//float* descriptor = (float*)malloc(sizeof(float) * dim);
	memset(feat,0,sizeof(float) * dim);

    bool pPadding = 0;
    int offset = pPadding ? 0 : 1;

	if(	voc_hog_get_descriptor(dimx, dimy, offset, dim, 1.f, 
        sbin, feat) ) {
		//free(descriptor);
		return;
	}
}

// Change to use 32 floats 
// featArrSize = maxoct * interval
void process_all_scales(float* im, int dimy, int dimx, int sbin, 
    FloatPtr* featArr, int maxoct, int interval, int* eleSize/*, float* res_img*/)
{
    //vocHOGManager hog; // constructor and deconstructor

    bool pPadding = 0;
    int offset = pPadding ? 0 : 1;

    //int interval = 10;
    //int maxoct = (int)log2(min(dimy, dimx) / (float)(sbin)) - 1;
    float scale = 1.0f;
    int res_dimx, res_dimy;
    //int width = dimx;
    //int height = dimy;
    res_dimx = (int)round((float)dimx*scale);
    res_dimy = (int)round((float)dimy*scale);
    int width = res_dimx;
    int height = res_dimy;

    voc_hog_initialize();

    if( voc_hog_transfer_image(im, dimx, dimy) ) {
		//free(descriptor);
		return;
	}
    
//    float* temp_img = (float*)malloc(sizeof(float)*4*width*height);
//    int flag = 1;

    for (int o = 0; o < maxoct; ++o)
	{
        //voc_hog_set_octref(res_dimx, res_dimy, o);
		for (int i = 0; i < interval; ++i)
		{   
            //printf("--------------------------------------\n");
			//printf("octave = %d, scale = %f\n", o, scale);
			//printf("width = %d, height = %d\n", res_dimx, res_dimy);

		    int index = o*interval + i;
            // every octave keeps the same size for scaling
            // first call with scale 1.0 because of the texture binding 
            //voc_hog_resize_image(width, height, res_dimx, res_dimy);

            
            //voc_hog_debug_resize_image(dimx, dimy, res_dimx, res_dimy, temp_img);
            //voc_hog_debug_resize_image(width, height, res_dimx, res_dimy, o, temp_img);
            voc_hog_resize_image(width, height, res_dimx, res_dimy, o);
/*
            if (flag == 22){
                printf("copying to host...ref_width = %d, ref_height = %d, res_dimy = %d, res_dimx = %d\n", width, height, res_dimy, res_dimx);
                memcpy(res_img, temp_img, sizeof(float)*4*res_dimx*res_dimy);
                //memcpy(res_img, temp_img, sizeof(float)*4*width*height);
                //flag = 0;
            }
            flag++;
*/            
            
        	if(	voc_hog_get_descriptor(
                    res_dimx, res_dimy, // every processing with a new size
                    offset, eleSize[index], scale, 
                    sbin, *(featArr+index) ) ){
        		return;
            }

/*
            printf("eleSize = %d\n", *(eleSize + index));
            
            for (int j = 0; j < *(eleSize + index); j++){
                *(*(featArr+index)+j) = (float)(index+j);

            }
*/

            scale = pow(2, -1.0*(i+1)/interval);
            //scale = pow(2, -1.0*(index+1)/interval);

            res_dimx = (int)round((float)width*scale);
			res_dimy = (int)round((float)height*scale);
            //res_dimx = (int)round((float)dimx*scale);
			//res_dimy = (int)round((float)dimy*scale);

            
        } // end for interv
        
        width = res_dimx;
        height = res_dimy;
        //voc_hog_set_octref(res_dimx, res_dimy);
        //voc_hog_set_octref(dimx, dimy);
	}
    voc_hog_release_image();
//    free(temp_img);
    voc_hog_finalize();
}

void debug_resize(float* im, int height, int width, int res_dimy, int res_dimx, float* res_img)
{
    if( voc_hog_transfer_image(im, width, height) ) {
		//free(descriptor);
		return;
	}
    voc_hog_resize_image(width, height, width, height, 0);

    //res_dimx = (int)round((float)width*scale);
    //res_dimy = (int)round((float)height*scale);
    voc_hog_debug_resize_image(width, height, res_dimx, res_dimy, 0, res_img);
    voc_hog_release_image();
}
