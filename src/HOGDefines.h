// 
// This file is derived from fastHOG, some utility functions are used in CUHOG
//

#ifndef __HOG_DEFINES__
#define __HOG_DEFINES__

#define UNROLL_LOOPS

#ifdef _WIN32
	#pragma comment( lib, "C:\\CUDA\\lib\\cuda.lib" )
	#pragma comment( lib, "C:\\CUDA\\lib\\cudart.lib" )
	#pragma comment( lib, "C:\\CUDA\\SDK\\common\\lib\\cutil32.lib" )
#endif

#ifndef CUDA_PIXEL
#define CUDA_PIXEL unsigned char
#endif

#ifndef CUDA_FLOAT
#define CUDA_FLOAT float
#endif

#ifndef CUDA_DT_PIXEL
#define CUDA_DT_PIXEL float
#endif

#ifndef CUDA_DT_PIXEL_INT
#define CUDA_DT_PIXEL_INT int
#endif

#ifndef THREAD_SIZE_W
#define THREAD_SIZE_W 16
#endif

#ifndef THREAD_SIZE_H
#define THREAD_SIZE_H 16
#endif

#ifndef BLOCK_SIZE_H
#define BLOCK_SIZE_H 16
#endif

#ifndef BLOCK_SIZE_W
#define BLOCK_SIZE_W 16
#endif

#ifndef MAX_HISTOGRAM_NO_BINS
#define MAX_HISTOGRAM_NO_BINS 9
#endif

#ifndef MAX_CELL_SIZE_Y
#define MAX_CELL_SIZE_Y 8
#endif

#ifndef MAX_CELL_SIZE_X
#define MAX_CELL_SIZE_X 8
#endif

#ifndef MAX_BLOCK_SIZE_X
#define MAX_BLOCK_SIZE_X 2
#endif

#ifndef MAX_BLOCK_SIZE_Y
#define MAX_BLOCK_SIZE_Y 2
#endif

#ifndef MAX_BLOCKS_PER_WINDOW_X
#define MAX_BLOCKS_PER_WINDOW_X 7
#endif

#ifndef MAX_BLOCKS_PER_WINDOW_Y
#define MAX_BLOCKS_PER_WINDOW_Y 15
#endif

#ifndef EXECUTYIN512THREADS
#define EXECUTYIN512THREADS(counter, startPoint, func, params) \
	startPoint = 0;\
	if (counter / 512 > 0) \
	{ \
		while (counter / 512 > 0) \
		{ \
		func<<<1, 512>>> ## params; \
			startPoint += 512; \
			counter -= 512; \
		} \
		if (counter != 0) \
		func<<<1, counter>>> ## params; \
	} \
	else \
	func<<<1, counter>>> ## params;
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef MAX_BLOCKS_PER_DIM
#define MAX_BLOCKS_PER_DIM	65536
#endif

#ifndef IMUL
#define IMUL(a, b) __mul24(a, b)
#endif

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

#ifndef DEGTORAD
#define DEGTORAD 0.017453292519943295769236907684886
#endif

#ifndef RADTODEG
#define RADTODEG 57.2957795
#endif

#endif
