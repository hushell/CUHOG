#include <cuda.h>
#include <cutil_inline.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "global.h"
#include "get_cells.h"
#include "get_features.h"
#include "voc_hog.h"
#include "timer.h"

#define MAX_ALLOC 300
#define MAX_BINS 18
#define MAX_DIMS 32

float* d_pBlocks = NULL;
float2* d_pGradMag = NULL;

float* d_pHist = NULL;
float* d_pNorm = NULL;
float* d_pOut = NULL;

bool bImagePrepared = false;

// VOC
// ------------------------------------------------------------------

int voc_hog_transfer_image(ftype* h_pImg, int width, int height)
{
    //Timer tt;
    //startTimer(&tt);
	//printf("In voc_hog_transfer_image: first pixel = %f, %f, %f, %f\n", *(im), *(im+1), *(im+2), *(im+3));

	if(bImagePrepared)
		voc_hog_release_image();

    if( voc_prepare_image2(h_pImg, width, height, 8) ) {
    //if( voc_prepare_image3(h_pImg, width, height) ) {
		printf("prepare_image failed\n");
		return -1;
	}

    //stopTimer(&tt);
    //printf("time in voc_prepare_image = %f\n", getTimerValue(&tt));
    /*
	int maxoct = (int)log2(min(height, width) / (8.0)) - 1;

    cudaMalloc((void**)&d_pOut, MAX_ALLOC * MAX_ALLOC * MAX_DIMS * sizeof(float));
		ONFAIL("d_pOut malloc failed");
	cudaMemset(d_pOut, 0, MAX_ALLOC * MAX_ALLOC * MAX_DIMS * sizeof(float));
		ONFAIL("d_pOut memset failed");
    */

    bImagePrepared = true;
	return 0;
}

void voc_hog_resize_image(int width, int height, int res_wid, int res_hei, int oct)
{
#ifdef DEBUG_TIME_EACH_STEP
    Timer tt;
    startTimer(&tt);
#endif
    voc_resize_image(width, height, res_wid, res_hei, oct);
#ifdef DEBUG_TIME_EACH_STEP
    stopTimer(&tt);
    printf("time in voc_hog_resize_image = %f\n", getTimerValue(&tt));
#endif

    cutilSafeCall(cudaMemset(d_pHist, 0, MAX_ALLOC * MAX_ALLOC * MAX_BINS * sizeof(float)));

    cutilSafeCall(cudaMemset(d_pNorm, 0, MAX_ALLOC * MAX_ALLOC * sizeof(float)));

    cutilSafeCall(cudaMemset(d_pOut, 0, MAX_ALLOC * MAX_ALLOC * MAX_DIMS * sizeof(float))); 
}

void voc_hog_set_octref(int ref_dimx, int ref_dimy, int oct)
{
    voc_set_octref(ref_dimx, ref_dimy, oct);
}

void voc_hog_debug_resize_image(int width, int height, int res_wid, int res_hei, int oct, float* res_img)
{
    Timer tt;
    startTimer(&tt);
    voc_debug_resize_image(width, height, res_wid, res_hei, oct, res_img);
    stopTimer(&tt);
    printf("time in voc_hog_resize_image = %f\n", getTimerValue(&tt));

    cutilSafeCall(cudaMemset(d_pHist, 0, MAX_ALLOC * MAX_ALLOC * MAX_BINS * sizeof(float)));

    cutilSafeCall(cudaMemset(d_pNorm, 0, MAX_ALLOC * MAX_ALLOC * sizeof(float)));

    cutilSafeCall(cudaMemset(d_pOut, 0, MAX_ALLOC * MAX_ALLOC * MAX_DIMS * sizeof(float)));
}


int voc_hog_get_descriptor(int width, int height, int bPad,
						int out_dim, float scale,
						int sbin, float* h_pDescriptor)
{

	if(!h_pDescriptor) return -10;

	int padX = 0;
	int padY = 0;

    // no padding but in case
    bPad = false;
	if( bPad ) {
		padX = HOG_PADDING_X;
		padY = HOG_PADDING_Y;
	}

	//int w = (int)(width/scale);
	//int h = (int)(height/scale);

	int paddedWidth = padX * 2 + (int)(width);
	int paddedHeight= padY * 2 + (int)(height);

    int blocks[2];
    blocks[0] = (int)round((ftype)height/(ftype)sbin);
    blocks[1] = (int)round((ftype)width/(ftype)sbin);

	//int res = voc_compute_gradients(paddedWidth, paddedHeight, 
      //      sbin, blocks[0], blocks[1], padX, padY, d_pHist);
	//Timer tt;
	//startTimer(&tt);
	
	//Timer ttg;
	//startTimer(&ttg);
	int res = voc_compute_gradients(paddedWidth, paddedHeight, 
            sbin, blocks[0], blocks[1], padX, padY, d_pHist);
	//stopTimer(&ttg);
	//printf("time in compute_gradients = %f\n", getTimerValue(&ttg));
	if( res ) {
		printf("w x h : %d x %d \t s: %.4f\n", paddedWidth, paddedHeight, scale);
		printf("compute_gradients failed: %d\n", res);
		return -3;
	}

	//Timer tte;
	//startTimer(&tte);
    res = voc_compute_block_energy(blocks[0], blocks[1], d_pHist, d_pNorm);
	//stopTimer(&tte);
	//printf("time in compute_block_energy = %f\n", getTimerValue(&tte));
    if( res ) {
		//printf("w x h : %d x %d \t s: %.4f\n", paddedWidth, paddedHeight, scale);
		printf("compute_block_energy failed: %d\n", res);
		return -3;
	}

	//Timer ttf;
	//startTimer(&ttf);
    voc_compute_features(blocks[0], blocks[1], d_pHist, d_pNorm, d_pOut);
	//stopTimer(&ttf);
	//printf("time in compute_features = %f\n", getTimerValue(&ttf));


	//Timer ttt;
	//startTimer(&ttt);
    cudaMemcpy(h_pDescriptor, d_pOut, sizeof(float) * out_dim,
				cudaMemcpyDeviceToHost);
	//stopTimer(&ttt);
	//printf("time in trasfering = %f\n", getTimerValue(&ttt));

	//stopTimer(&tt);
	//printf("time each scale = %f\n", getTimerValue(&tt));

    return 0;
}

int voc_hog_initialize()
{

// ------------------------------------------------------------------------
//	check cuda device
/*
	int deviceCount = 0;
	if( cudaGetDeviceCount( &deviceCount) ) {
		printf("cudaGetDeviceCount failed\n");
		printf("CUDA driver and runtime version may be mismatched!\n");
	}

	if( deviceCount == 0 ) {
		printf("sorry no CUDA capable device found.\n");
		return -1;
	}

	int dev;

    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0) {
			// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                printf("There is no device supporting CUDA.\n");
            else if (deviceCount == 1)
                printf("There is 1 device supporting CUDA\n");
            else
                printf("There are %d devices supporting CUDA\n", deviceCount);
        }
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	}

	int driverVersion = 0, runtimeVersion = 0;
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf("driver: %d\nruntime: %d\n", driverVersion / 1000, runtimeVersion / 1000);
*/

// ------------------------------------------------------------------------
//	prepare weights
	//prepareGaussWeights();
	//prepareBilinearWeights();
// ------------------------------------------------------------------------
//	malloc all memory that will be needed during processing
	// cell hists
	cudaMalloc((void**)&d_pHist, MAX_ALLOC * MAX_ALLOC * MAX_BINS * sizeof(float));
		ONFAIL("d_pHist malloc failed");
	cudaMemset(d_pHist, 0, MAX_ALLOC * MAX_ALLOC * MAX_BINS * sizeof(float));
		ONFAIL("d_pHist memset failed");

	// blocks outputs
	//const int nBlocks = MAX_IMAGE_DIMENSION/8 * MAX_IMAGE_DIMENSION/8 ;	// WE ASSUME MAXIMUM IMAGE SIZE OF 1280x1280
	//const int blocksMemorySize = nBlocks * HOG_BLOCK_CELLS_X * HOG_BLOCK_CELLS_Y * NBINS * sizeof(float);
	cudaMalloc((void**)&d_pNorm, MAX_ALLOC * MAX_ALLOC * sizeof(float));
		ONFAIL("d_pNorm malloc failed");
    cudaMemset(d_pNorm, 0, MAX_ALLOC * MAX_ALLOC * sizeof(float));
		ONFAIL("d_pNorm memset failed");

    cudaMalloc((void**)&d_pOut, MAX_ALLOC * MAX_ALLOC * MAX_DIMS * sizeof(float));
		ONFAIL("d_pOut malloc failed");
	cudaMemset(d_pOut, 0, MAX_ALLOC * MAX_ALLOC * MAX_DIMS * sizeof(float));
		ONFAIL("d_pOut memset failed");

	return 0;
}

int voc_hog_finalize()
{

	cudaFree(d_pHist); d_pHist = NULL;
		ONFAIL("cudaFree failed: d_pHist");
	cudaFree(d_pNorm); d_pNorm = NULL;
		ONFAIL("cudaFree failed: d_pNorm");

    cudaFree(d_pOut); d_pOut = NULL;
		ONFAIL("cudaFree failed: d_pOut");
	//if( svm_finalize() || blocks_finalize() )
		//return -1;

	return 0;
}

int voc_hog_release_image()
{
    if (bImagePrepared == false)
    {
        printf("There is no image be prepared in GPU!\n");
        return -1;
    }
    bImagePrepared = false;
	return voc_destroy_image2();
}

