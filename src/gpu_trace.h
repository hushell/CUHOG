//
// GPU Trace version 0.01
//
// A trace library for CUDA gpu codes
//
// Revision history
//
//		0.01: Initial release
//		0.02: Some minor corrections; added __trace_exp()
//

#ifndef __GPU_TRACE__H__
#define __GPU_TRACE__H__

// Uncomment following line, so by default tracing will be enabled, if gpu_trace.h is included
//#define __ENABLE_TRACE__

// Macros

//
// __ENABLE_TRACE__
// Enables tracing functionality

#ifdef __ENABLE_TRACE__

//
// Include stdio.h to use fprintf and FILE

#include <stdio.h>

//
// __LINEAR_THREAD_INDEX__
// Linear thread index

#define __LINEAR_THREAD_INDEX__()		((blockDim.x * blockDim.y * blockDim.z) * (gridDim.x * blockIdx.y + blockIdx.x) + \
										 (blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x))

//
// Default values
// To change these values, #define them before including gpu_trace.h

//
// __TRACE_MIN_THREAD__
// Minimum thread index to trace

#ifndef __TRACE_MIN_THREAD__
	#define __TRACE_MIN_THREAD__ 0
#endif

//
// __TRACE_MAX_THREAD__
// Maximum thread index to trace

#ifndef __TRACE_MAX_THREAD__
	#define __TRACE_MAX_THREAD__ 64
#endif

//
// __TRACE_MAX_PACKETS__
// Minimum number of trace packets kept

#ifndef __TRACE_MAX_PACKETS__
	#define __TRACE_MAX_PACKETS__ 1000
#endif

//
// __TRACE_TAG_SIZE__
// Size of tag field in __trace_packet

#ifndef __TRACE_TAG_SIZE__
	#define __TRACE_TAG_SIZE__ 8
#endif

//
// __TRACE_MSG_SIZE__
// Size of msg field in __trace_packet

#ifndef __TRACE_MSG_SIZE__
	#define __TRACE_MSG_SIZE__ 16
#endif	

//
// __traceable__
// Used to define traceable functions

#define __traceable__(...)			(__trace_data *__global_trace_data, __VA_ARGS__)

//
// __traceable_call__
// Used to call traceable functions

#define __traceable_call__(...)		(__global_trace_data, __VA_ARGS__)

//
// __trace
// Used to trace values in traceable functions

#define __trace(tag, msg, x)		__trace__(__global_trace_data, tag, msg, x)

//
// __trace_exp
// Used to trace expressions in traceable functions

#define __trace_exp(tag, x)			__trace__(__global_trace_data, tag, #x, x)

//
// __gpu_strcpy
// Device mode strcpy, when defined as a __device__ function, it does not work as expected. A NVCC 2.1 bug?

#define __gpu_strcpy(dst, src, buf_len)			\
{												\
	char *__src = src, *__dst = dst;			\
	for (int i = 0; i < buf_len - 1; i++)		\
		*__dst++ = *__src++;					\
												\
	*__dst = 0;									\
}

//
// __REGISTER_TRACE_TYPE__
// Register appropriate __trace__() for data given type

#define __REGISTER_TRACE_TYPE__(var_type, var_id, var_type_id)												\
	__device__ void __trace__(__trace_data *__global_trace_data, char *__tag, char *__msg, var_type var)	\
	{																										\
		const int TIdx = __LINEAR_THREAD_INDEX__() - __TRACE_MIN_THREAD__;									\
																											\
		if (TIdx >= 0 && TIdx <= __TRACE_MAX_THREAD__ - __TRACE_MIN_THREAD__)								\
			if (__global_trace_data[TIdx].size < __TRACE_MAX_PACKETS__)										\
			{																								\
				const unsigned int Idx = __global_trace_data[TIdx].size++;									\
				__gpu_strcpy(__global_trace_data[TIdx].data[Idx].tag, __tag, __TRACE_TAG_SIZE__);			\
				__gpu_strcpy(__global_trace_data[TIdx].data[Idx].msg, __msg, __TRACE_MSG_SIZE__);			\
				__global_trace_data[TIdx].data[Idx].type = var_type_id;										\
				__global_trace_data[TIdx].data[Idx].var_id = var;											\
			}																								\
	}

//
// __INITIALIZE_TRACE_DATA__
// Used to initialize trace data

#define INITIALIZE_TRACE_DATA()		__InitializeTraceData()

//
// __FINALIZE_TRACE_DATA__
// Used to finalize trace data

#define FINALIZE_TRACE_DATA()		__FinalizeTraceData()

//
// __PRINT_TRACE_DATA__
// Used to print trace data

#define PRINT_TRACE_DATA(f)			__PrintTraceData(f)

//
// __trace_var_type
// Stored variable type

enum __trace_var_type
{
	__int,
	__unsigned_int,
	__long_int,
	__unsigned_long_int,
	__float,
	__double,
};

//
// __trace_packet
// Smallest packet of trace data

struct __trace_packet
{
	char tag[__TRACE_TAG_SIZE__], msg[__TRACE_MSG_SIZE__];

	__trace_var_type type;

	union
	{
		int _int;
		unsigned int _unsigned_int;
		long int _long_int;
		unsigned long int _unsigned_long_int;
		float _float;
		double _double;
	};
};

//
// __trace_data
// Set of __trace_packet's for a single thread

struct __trace_data
{
	unsigned int size;
	__trace_packet data[__TRACE_MAX_PACKETS__];
};

//
// Register various trace data types

__REGISTER_TRACE_TYPE__(int, _int, __int)
__REGISTER_TRACE_TYPE__(unsigned int, _unsigned_int, __unsigned_int)
__REGISTER_TRACE_TYPE__(long int, _long_int, __long_int)
__REGISTER_TRACE_TYPE__(unsigned long int, _unsigned_long_int, __unsigned_long_int)
__REGISTER_TRACE_TYPE__(float, _float, __float)
__REGISTER_TRACE_TYPE__(double, _double, __double)

//
// __global_trace_data
// Global trace data for all threads

__trace_data __host_global_trace_data[__TRACE_MAX_THREAD__ - __TRACE_MIN_THREAD__ + 1], *__global_trace_data;

//
// __InitializeTraceData
// Initilizes trace data

void __InitializeTraceData()
{
	memset(__host_global_trace_data, 0, sizeof(__host_global_trace_data));
	
	for (int __i = 0; __i < __TRACE_MAX_THREAD__ - __TRACE_MIN_THREAD__ + 1; __i++)
		__host_global_trace_data[__i].size = 0;
				
	cudaMalloc((void **) &__global_trace_data, sizeof(__host_global_trace_data));
	cudaMemcpy(__global_trace_data, __host_global_trace_data, sizeof(__host_global_trace_data), cudaMemcpyHostToDevice);
}

//
// __FinalizeTraceData
// Finalizes trace data

void __FinalizeTraceData()
{
	cudaMemcpy(__host_global_trace_data, __global_trace_data, sizeof(__host_global_trace_data), cudaMemcpyDeviceToHost);
	cudaFree(__global_trace_data);
}

//
// __PrintTraceData
// Prints trace data

void __PrintTraceData(FILE *f)
{
	fprintf(f, "\nGPU Trace: collected trace data:\n\n");

	for (int i = 0; i < __TRACE_MAX_THREAD__ - __TRACE_MIN_THREAD__ + 1; i++)
	{
		fprintf(f, "== Thread %d: %u trace packets ================================\n", i, __host_global_trace_data[i].size);

		for (int j = 0; j < __host_global_trace_data[i].size; j++)
		{
			fprintf(f, "    [%-*s][%-*s]", __TRACE_TAG_SIZE__ - 1, __host_global_trace_data[i].data[j].tag, __TRACE_MSG_SIZE__ - 1, __host_global_trace_data[i].data[j].msg);
			
			switch (__host_global_trace_data[i].data[j].type)
			{
				case __int: fprintf(f, "[int: %d]\n", __host_global_trace_data[i].data[j]._int); break;

				case __unsigned_int: fprintf(f, "[unsigned int: %u]\n", __host_global_trace_data[i].data[j]._unsigned_int); break;

				case __long_int: fprintf(f, "[long int: %ld]\n", __host_global_trace_data[i].data[j]._long_int); break;

				case __unsigned_long_int: fprintf(f, "[unsigned long int: %lu]\n", __host_global_trace_data[i].data[j]._unsigned_long_int); break;

				case __float: fprintf(f, "[float: %g]\n", __host_global_trace_data[i].data[j]._float); break;

				case __double: fprintf(f, "[double: %g]\n", __host_global_trace_data[i].data[j]._double); break;
			}
		}

		fprintf(f, "\n");
	}
}

#else

//
// Defining some of the above macros when tracing is disabled to trivial one

#define __traceable__(...)			(__VA_ARGS__)
#define __traceable_call__(...)		(__VA_ARGS__)

#define __trace(tag, msg, x)
#define __trace_exp(tag, x)

#define INITIALIZE_TRACE_DATA()
#define FINALIZE_TRACE_DATA()
#define PRINT_TRACE_DATA(f)

#endif  // __ENABLE_TRACE__

#endif  // __GPU_TRACE__H__
