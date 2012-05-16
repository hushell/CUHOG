//
// A test program for gpu_trace.h library
//
// Compile with:
//
//   Trace mode:
//
//     nvcc test.cu -o test -D__ENABLE_TRACE__ -arch compute_13
//
//   Normal mode:
//
//     nvcc test.cu -o test
//
// Please note that -arch compute_13 is used because of __trace() ing double type data
// If you only use float ones, you may remove it
//
// Tested on Ubuntu 8.10 x86-64, CUDA 2.1
//

#include <stdio.h>

#include "cuda_runtime_api.h"

//
// Change some defaults

//
// Minimum thread index to trace
#define __TRACE_MIN_THREAD__	1

//
// Maximum thread index to trace
#define __TRACE_MAX_THREAD__	2

//
// Size of msg field
#define __TRACE_MSG_SIZE__		16

#include "gpu_trace.h"

__global__ void test __traceable__ (int dummy)
{
	int x = threadIdx.x;

	__trace("Test", "int", x);
	__trace("Test", "unsigned int", static_cast <unsigned int> (x));
	__trace("Test", "long int", static_cast <long int> (x));
	__trace("Test", "unsigned long int", static_cast <unsigned long int> (x));
	__trace("Test", "float", static_cast <float> (x));
	__trace("Test", "double", static_cast <double> (x));
	
	for (int i = 0; i < x; i++)
		__trace_exp("Loop", 3 + 2 * i);
}

int main()
{
	INITIALIZE_TRACE_DATA();
		
	test <<<10, 10>>> __traceable_call__ (0);
	
	cudaError_t ErrorCode = cudaGetLastError();
	if (ErrorCode != cudaSuccess)
		printf("*** Kernel did not launch, %s ***\n", cudaGetErrorString(ErrorCode));

	ErrorCode = cudaThreadSynchronize();
	if (ErrorCode != cudaSuccess)
		printf("*** Kernel exited while executing, %s ***\n", cudaGetErrorString(ErrorCode));
	
	FINALIZE_TRACE_DATA();
	PRINT_TRACE_DATA(stdout);
	
	return 0;
}
