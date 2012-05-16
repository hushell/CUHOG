#include <stdio.h>
#include "global.h"
#include <cuda.h>
#include "cuda_runtime_api.h"
#include "gpu_trace.h"
#include "timer.h"

cudaArray* d_pGaussWeights = NULL;
texture<float, 2, cudaReadModeElementType> t_gaussian_weights;
// float4 -x UL - y UR - z BL - w BR -- one lookup table for each cell in the block
cudaArray* d_pBilinearWeights = NULL;
texture<float4, 2, cudaReadModeElementType> t_bilinear_weights;


__global__ void d_voc_compute_block_energy(int blocks_0, int blocks_1, 
                                                float* d_pHists, float* d_pNorms)
{
    const int posx	= blockDim.x * blockIdx.x + threadIdx.x;
	const int posy	= blockDim.y * blockIdx.y + threadIdx.y;

    if (posx < blocks_1 && posy < blocks_0)
    {
        int o;
        const int pos = posx*blocks_0 + posy;
        float* dst = d_pNorms + pos;
        const int bin_step = blocks_0*blocks_1;
        float sum = 0.0f;
        for (o=0; o<9; ++o){
            float* src1 = d_pHists + pos + o*bin_step; 
            float* src2 = d_pHists + pos + (o+9)*bin_step;
            
            sum += (*src1+*src2) * (*src1+*src2);
        }
        atomicAdd(dst,sum);
    }
}

__host__ int voc_compute_block_energy(int blocks_0, int blocks_1, float* d_pHists, float* d_pNorms)
{
    dim3 grid;
    grid.x = (int)ceil((blocks_1+7) / 8);
    grid.y = (int)ceil((blocks_0+7) / 8);
    dim3 threads;
	threads.x = 8; 
    threads.y = 8;

    //printf("grid.x = %d, grid.y = %d, blocks_0 = %d, blocks_1 = %d\n", grid.x, grid.y, blocks_0, blocks_1);
#ifdef DEBUG_TIME_EACH_STEP
	Timer tt;
	startTimer(&tt);
#endif

    d_voc_compute_block_energy<<< grid , threads >>>(blocks_0, blocks_1, d_pHists, d_pNorms);
        ONFAIL("compute_blocks kernel failed");

#ifdef DEBUG_TIME_EACH_STEP
	stopTimer(&tt);
	printf("time in voc_compute_block_energy = %f\n", getTimerValue(&tt));
#endif

//#define DEBUG_voc_compute_block_energy
#ifdef DEBUG_voc_compute_block_energy

	float *h_pNorm = (float*)malloc(blocks_0*blocks_1*sizeof(float));
	if(!h_pNorm) {
		printf("h_pNorm: malloc failed \n");
		return -1;
	}
	// copy results
	cudaMemcpy(h_pNorm, d_pNorms, blocks_0*blocks_1*sizeof(float), cudaMemcpyDeviceToHost);

    int out[3];
    out[0] = max(blocks_0-2, 0);
    out[1] = max(blocks_1-2, 0);
    out[2] = 32;

    char filname[50];
    sprintf(filname, "energy_blocks_gpu_%d_%d.txt", out[1], out[0]);

	// write complete output to file
	FILE* fp = fopen(filname, "w");
	if(!fp) 
        printf("failed to open output file: fmag\n");

    int ci, cj;
    for (ci = 0; ci < blocks_0; ci++)
    {
      for (cj = 0; cj < blocks_1; cj++)
      {
          fprintf(fp, "(i=%d,j=%d)\n", ci, cj);
          //for (cb = 0; cb < 18; cb++)
          {
              fprintf(fp, "%f ", *(h_pNorm + cj * blocks_0 + ci));
          }
          fprintf(fp, "\n");
      }

    }

	fclose(fp);
    free(h_pNorm);
#endif

    return 0;
}

__global__ void d_voc_compute_features /*__traceable__*/ (int out_0, int out_1, 
                                                int blocks_0, int blocks_1,
                                                float* d_pHists, float* d_pNorms, 
                                                float* d_pOut)
{
    const int posx	= blockDim.x * blockIdx.x + threadIdx.x;
	const int posy	= blockDim.y * blockIdx.y + threadIdx.y;
    volatile __shared__ float s_norm[10][10]; // s_norm[blockDim.x+1][blockDim.y+1]

    //__trace("pos", "int", posx);
	//__trace("pos", "int", posy);

    if (posx < blocks_1 && posy < blocks_0)
    {
        s_norm[threadIdx.x][threadIdx.y] = *(d_pNorms + (posx)*blocks_0 + posy);

        //__syncthreads();

        // !!!potential 2-bank conflicts
        if (threadIdx.x == blockDim.x - 1 && (posx+2) < blocks_1)
        {
            s_norm[threadIdx.x + 1][threadIdx.y] = *(d_pNorms + (posx+1)*blocks_0 + posy);
            s_norm[threadIdx.x + 2][threadIdx.y] = *(d_pNorms + (posx+2)*blocks_0 + posy);
            //__syncthreads();
        }

        if (threadIdx.y == blockDim.y - 1 && (posy+2) < blocks_0)
        {
            s_norm[threadIdx.x][threadIdx.y + 1] = *(d_pNorms + (posx)*blocks_0 + posy+1);
            s_norm[threadIdx.x][threadIdx.y + 2] = *(d_pNorms + (posx)*blocks_0 + posy+2);
            //__syncthreads();
        }

        if (threadIdx.y == blockDim.y - 1 && threadIdx.x == blockDim.x - 1 && (posy+2) < blocks_0 && (posx+2) < blocks_1)
        {
            s_norm[threadIdx.x + 1][threadIdx.y + 1] = *(d_pNorms + (posx+1)*blocks_0 + posy+1);
            s_norm[threadIdx.x + 1][threadIdx.y + 2] = *(d_pNorms + (posx+1)*blocks_0 + posy+2);
            s_norm[threadIdx.x + 2][threadIdx.y + 1] = *(d_pNorms + (posx+2)*blocks_0 + posy+1);
            s_norm[threadIdx.x + 2][threadIdx.y + 2] = *(d_pNorms + (posx+2)*blocks_0 + posy+2);
            //__syncthreads();
        }
        __syncthreads();
    }
    //__syncthreads();

    if (posx < out_1 && posy < out_0)
    {
        float* dst = d_pOut + posx*out_0 + posy;
        float* src = 0; 
        int px, py;
        float n1, n2, n3, n4;
        int bx = blockDim.x * blockIdx.x;
        int by = blockDim.y * blockIdx.y;

        float h1, h2, h3, h4;
        float t1 = 0; float t2 = 0; float t3 = 0; float t4 = 0;
        int o;

        px = posx - bx + 1;
        py = posy - by + 1;
        n1 = 1.0f / sqrtf(s_norm[px][py] + s_norm[px][py+1] + s_norm[px+1][py] + s_norm[px+1][py+1] + eps);

        /*
        __trace("T", "float", s_norm[px][py]);
        __trace("T", "float", s_norm[px][py+1]);
        __trace("T", "float", s_norm[px+1][py]);
        __trace("T", "float", s_norm[px+1][py+1]);
        */

        px = posx - bx + 1;
        py = posy - by;
        n2 = 1.0f / sqrtf(s_norm[px][py] + s_norm[px][py+1] + s_norm[px+1][py] + s_norm[px+1][py+1] + eps);

        /*
        __trace("T", "float", s_norm[px][py]);
        __trace("T", "float", s_norm[px][py+1]);
        __trace("T", "float", s_norm[px+1][py]);
        __trace("T", "float", s_norm[px+1][py+1]);
        */

        px = posx - bx;
        py = posy - by + 1;
        n3 = 1.0f / sqrtf(s_norm[px][py] + s_norm[px][py+1] + s_norm[px+1][py] + s_norm[px+1][py+1] + eps);

        /*
        __trace("T", "float", s_norm[px][py]);
        __trace("T", "float", s_norm[px][py+1]);
        __trace("T", "float", s_norm[px+1][py]);
        __trace("T", "float", s_norm[px+1][py+1]);
        */

        px = posx - bx;
        py = posy - by;
        n4 = 1.0f / sqrtf(s_norm[px][py] + s_norm[px][py+1] + s_norm[px+1][py] + s_norm[px+1][py+1] + eps);

        /*
        __trace("T", "float", s_norm[px][py]);
        __trace("T", "float", s_norm[px][py+1]);
        __trace("T", "float", s_norm[px+1][py]);
        __trace("T", "float", s_norm[px+1][py+1]);
        */

        src = d_pHists + (posx+1)*blocks_0 + (posy+1);
        for (o = 0; o < 18; o++)
        {
            float src_val = *src;
            h1 = (src_val * n1 <= 0.2f ? src_val * n1 : 0.2f);
            h2 = (src_val * n2 <= 0.2f ? src_val * n2 : 0.2f);
            h3 = (src_val * n3 <= 0.2f ? src_val * n3 : 0.2f);
            h4 = (src_val * n4 <= 0.2f ? src_val * n4 : 0.2f);

            *dst = 0.5f * (h1+h2+h3+h4);
            
            t1 += h1;
            t2 += h2;
            t3 += h3;
            t4 += h4;
            
            dst += out_0 * out_1;
            src += blocks_0 * blocks_1;
        }

        src = d_pHists + (posx+1)*blocks_0 + (posy+1);
        for (o = 0; o < 9; o++)
        {
            float sum_val = *src + *(src + 9 * blocks_0 * blocks_1);
            h1 = (sum_val * n1 <= 0.2f ? sum_val * n1 : 0.2f);
            h2 = (sum_val * n2 <= 0.2f ? sum_val * n2 : 0.2f);
            h3 = (sum_val * n3 <= 0.2f ? sum_val * n3 : 0.2f);
            h4 = (sum_val * n4 <= 0.2f ? sum_val * n4 : 0.2f);

            *dst = 0.5f * (h1+h2+h3+h4);

            dst += out_0 * out_1;
            src += blocks_0 * blocks_1;
        }

        *dst = 0.2357f * t1;
        dst += out_0 * out_1;
        *dst = 0.2357f * t2;
        dst += out_0 * out_1;
        *dst = 0.2357f * t3;
        dst += out_0 * out_1;
        *dst = 0.2357f * t4;
        dst += out_0 * out_1;

    }
}

__host__ int voc_compute_features(int blocks_0, int blocks_1, 
                                        float* d_pHists, float* d_pNorms, 
                                        float* d_pOut)
{
    dim3 grid;
    grid.x = (int)ceil((blocks_1+7) / 8);
    grid.y = (int)ceil((blocks_0+7) / 8);
    dim3 threads;
	threads.x = 8; 
    threads.y = 8;

    int out[3];
    out[0] = max(blocks_0-2, 0);
    out[1] = max(blocks_1-2, 0);
    out[2] = 32;
    
    //INITIALIZE_TRACE_DATA();

#ifdef DEBUG_TIME_EACH_STEP
	Timer tt;
	startTimer(&tt);
#endif

    d_voc_compute_features<<< grid , threads >>> /*__traceable_call__*/ (out[0], out[1], 
            blocks_0, blocks_1, d_pHists, d_pNorms, d_pOut);
        ONFAIL("compute_blocks kernel failed");

#ifdef DEBUG_TIME_EACH_STEP
	stopTimer(&tt);
	printf("time in voc_compute_features = %f\n", getTimerValue(&tt));
#endif
    /*
    cudaError_t ErrorCode = cudaGetLastError();
	if (ErrorCode != cudaSuccess)
		printf("*** Kernel did not launch, %s ***\n", cudaGetErrorString(ErrorCode));

	ErrorCode = cudaThreadSynchronize();
	if (ErrorCode != cudaSuccess)
		printf("*** Kernel exited while executing, %s ***\n", cudaGetErrorString(ErrorCode));
	
	FINALIZE_TRACE_DATA();
	PRINT_TRACE_DATA(stdout);
    */

//#define DEBUG_voc_compute_features
#ifdef DEBUG_voc_compute_features

	float *h_pOut = (float*)malloc(out[0]*out[1]*out[2]*sizeof(float));
	if(!h_pOut) {
		printf("h_pNorm: malloc failed \n");
		return -1;
	}
	// copy results
	cudaMemcpy(h_pOut, d_pOut, out[0]*out[1]*out[2]*sizeof(float), cudaMemcpyDeviceToHost);

	// write complete output to file
	FILE* fp = fopen("cell_feats.txt", "w");
	if(!fp) 
        printf("failed to open output file: fmag\n");

    int ci, cj, cb;
    for (ci = 0; ci < out[0]; ci++)
    {
      for (cj = 0; cj < out[1]; cj++)
      {
          fprintf(fp, "(i=%d,j=%d)\n", ci, cj);
          for (cb = 0; cb < 32; cb++)
          {
              fprintf(fp, "%f ", *(h_pOut + cj * out[0] + ci + cb*out[0]*out[1]));
          }
          fprintf(fp, "\n");
      }

    }

	fclose(fp);
    free(h_pOut);
#endif

    return 0;

}


