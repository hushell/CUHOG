#ifndef __GLOBAL_H__
#define __GLOBAL_H__


#define HOG_BLOCK_CELLS_X	2
#define HOG_BLOCK_CELLS_Y	2
#define HOG_BLOCK_WIDTH 	(8*HOG_BLOCK_CELLS_X)
#define HOG_BLOCK_HEIGHT	(8*HOG_BLOCK_CELLS_Y)
#define HOG_CELL_SIZE		8		// we assume 8x8 cells!

#define HOG_PADDING_X		16		// padding in pixels to add to each side
#define HOG_PADDING_Y		16		// padding in pixels to add to top&bottom


extern float	HOG_START_SCALE;
extern float 	HOG_SCALE_STEP;

const float SIGMA	=		(0.5 * HOG_BLOCK_WIDTH);		// gaussian window size for block histograms
const int NBINS = 9;
const float HOG_TRAINING_SCALE_STEP = 1.05f;//1.2f // the original procedure uses 1.2 steps
const int MAXIMUM_HARD_EXAMPLES = 50000;

// FLAGS
#define ENABLE_GAMMA_COMPRESSION	// enable sqrt gamma compression

#define PRINT_PROFILING_TIMINGS		0 // show profiling timings for each frame
#define PRINT_VERBOSE_INFO			0 // show detection information
#define PRINT_DEBUG_INFO			0 // show verbose debug information at each scale level

// DEBUG FLAGS
#define DEBUG_PRINT_PROGRESS					0
#define DEBUG_PRINT_SCALE_CONSTRAINTS 			0

// -----------------------------------------------
#define VERBOSE_CUDA_FAILS

#ifdef VERBOSE_CUDA_FAILS
#define ONFAIL(S) { cudaError_t e = cudaGetLastError(); \
					if(e) { printf("%s:%s:"S":%s\n", __FILE__, __FUNCTION__ , cudaGetErrorString(e));\
					return -2; } }
#else
#define ONFAIL(S)
#endif

#define eps 0.0001f
typedef float ftype;

/*
extern float uu[9] = {
        1.0000, 
		0.9397, 
		0.7660, 
		0.500, 
		0.1736, 
		-0.1736, 
		-0.5000, 
		-0.7660, 
		-0.9397};
extern float vv[9] = {
        0.0000, 
		0.3420, 
		0.6428, 
		0.8660, 
		0.9848, 
		0.9848, 
		0.8660, 
		0.6428, 
		0.3420};
*/
    
#define min(x,y) \
    (x <= y ? x : y)

#define max(x,y) \
    (x <= y ? y : x)


//#define DEBUG_TIME_EACH_STEP




#endif
