#include <cuda.h>
#include <cutil_inline.h>
#include <stdio.h>
#include <time.h>

#include "global.h"
#include "cuda_runtime_api.h"
#include "gpu_trace.h"
#include "get_cells.h"
#include "timer.h"
#include "HOGUtils.h"

static cudaArray* normalized_image_array = NULL;
//float* d_pImage = 0;
float4* d_pImage = 0;
float4* d_pImage4 = 0;
float2* d_pGradients = 0;
texture<uchar4, 2, cudaReadModeNormalizedFloat> t_normalized_image_texture;
__device__ __constant__ float uu[9];
__device__ __constant__ float vv[9];
float* d_uu;
float* d_vv;

float4* d_RescaledImage = 0;
// It seems a texture can not be bind several times with different cudaArray
//cudaArray *imageArray;
cudaArray *imageArr;
cudaArray** imageArray = 0;
texture<float4, 2, cudaReadModeElementType> tex;
cudaChannelFormatDesc channelDescDownscale;
bool isAlloc = false;
int gmaxoct = 1; 

//
// Change some defaults
// Maximum thread index to trace
//#define __TRACE_MAX_THREAD__	260
//
// Size of msg field
//#define __TRACE_MSG_SIZE__		16

/*
__device__ __constant__ float uu[9] = {
        1.0000, 
		0.9397, 
		0.7660, 
		0.500, 
		0.1736, 
		-0.1736, 
		-0.5000, 
		-0.7660, 
		-0.9397};
__device__ __constant__ float vv[9] = {
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

// VOC
// -------------------------------------------------------------

// width and height is the rescaled size 
// only need to keep the original size in texture
__global__ void d_resize_bicubic(float4 *outputFloat, /*float4* d_pImage,*/ 
                                 int width, int height, float invScaleX, float invScaleY)
{
	int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	int i = __umul24(y, width) + x;

	float u = x*invScaleX;
	float v = y*invScaleY;

	if (x < width && y < height)
	{
		float4 cF;

        /*
		if (invScaleX == 1.0f && invScaleY == 1.0f)
		{
			cF = d_pImage[x + y * width];
			//cF.w = 0;
		}
		else
        */
		//{
			cF = tex2D(tex, u, v);
			//cF.w = 0;
            //cF.x = sqrtf(cF.x); cF.y = sqrtf(cF.y); cF.z = sqrtf(cF.z);
		//}

		//cF.x = sqrtf(cF.x); cF.y = sqrtf(cF.y); cF.z = sqrtf(cF.z); cF.w = 0;
		outputFloat[i] = cF;
	}
}





__global__ void d_voc_compute_hists(int width, int height, 
                                int block_0, int block_1, int sbin,
								int min_x, int min_y, int max_x, int max_y,
								float* d_pHist)

{
    //volatile __shared__ float uu[9];
    //volatile __shared__ float vv[9];

    const int posx	= blockDim.x * blockIdx.x + threadIdx.x + min_x;// pixel pos within padded image
	const int posy	= blockDim.y * blockIdx.y + threadIdx.y + min_y;// pixel pos within padded image

	/*
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        uu[0] =     1.0000f;
    	uu[1] = 	0.9397f;
        uu[2] = 	0.7660f;
    	uu[3] = 	0.500f;
    	uu[4] = 	0.1736f; 
    	uu[5] = 	-0.1736f; 
    	uu[6] = 	-0.5000f;
    	uu[7] = 	-0.7660f;
    	uu[8] = 	-0.9397f;

        vv[0] =     0.0000f;
    	vv[1] = 	0.3420f;
        vv[2] = 	0.6428f;
    	vv[3] = 	0.8660f;
    	vv[4] = 	0.9848f; 
    	vv[5] = 	0.9848f; 
    	vv[6] = 	0.8660f;
    	vv[7] = 	0.6428f;
    	vv[8] = 	0.3420f;
    }
	*/

	//__syncthreads();

    const float xstep = 1.f / (width-2);
	const float ystep = 1.f / (height-2);
	const float xoff = xstep / 2.f;
	const float yoff = ystep / 2.f;

	float4	pixel_up;
	float4	pixel_down;
	float4	pixel_left;
	float4	pixel_right;

    // for linear interpolation to 4 histograms around pixel
    ftype xp = ((ftype)posx+0.5f)/(ftype)sbin - 0.5f;
    ftype yp = ((ftype)posy+0.5f)/(ftype)sbin - 0.5f;
    int ixp = (int)floor(xp);
    int iyp = (int)floor(yp);
    ftype vx0 = xp-ixp;
    ftype vy0 = yp-iyp;
    ftype vx1 = 1.0f-vx0;
    ftype vy1 = 1.0f-vy0;

	if((posx < max_x) && (posx > 0) 
        && (posy < max_y) && (posy > 0)) {
        
		// the indizes can be <0 and >1 ! this implicitely pads the image
		pixel_down = tex2D(t_normalized_image_texture,
								(posx) * xstep + xoff,
								(posy+1) * ystep + yoff);
		pixel_up = tex2D(t_normalized_image_texture,
								(posx) * xstep + xoff,
								(posy-1) * ystep + yoff);
		pixel_left = tex2D(t_normalized_image_texture,
								(posx-1) * xstep + xoff,
								(posy) * ystep + yoff);
		pixel_right = tex2D(t_normalized_image_texture,
								(posx+1) * xstep + xoff,
								(posy) * ystep + yoff);

#ifdef ENABLE_GAMMA_COMPRESSION
		pixel_up.x = sqrtf(	pixel_up.x);
		pixel_up.y = sqrtf(	pixel_up.y);
		pixel_up.z = sqrtf(	pixel_up.z);
		pixel_up.w = sqrtf(	pixel_up.w);
		pixel_down.x = sqrtf(pixel_down.x);
		pixel_down.y = sqrtf(pixel_down.y);
		pixel_down.z = sqrtf(pixel_down.z);
		pixel_down.w = sqrtf(pixel_down.w);
		pixel_left.x = sqrtf(pixel_left.x);
		pixel_left.y = sqrtf(pixel_left.y);
		pixel_left.z = sqrtf(pixel_left.z);
		pixel_left.w = sqrtf(pixel_left.w);
		pixel_right.x = sqrtf(pixel_right.x);
		pixel_right.y = sqrtf(pixel_right.y);
		pixel_right.z = sqrtf(pixel_right.z);
		pixel_right.w = sqrtf(pixel_right.w);
#endif
		// compute gradient direction and magnitude
		//float3 grad_dx, grad_dy;

        ftype dy, dy2, dy3, dx, dx2, dx3, v, v2, v3;
        
		dy = (pixel_right.x - pixel_left.x);
		dy2 = (pixel_right.y - pixel_left.y);
		dy3 = (pixel_right.z - pixel_left.z);

		dx = (pixel_down.x - pixel_up.x);
		dx2 = (pixel_down.y - pixel_up.y);
		dx3 = (pixel_down.z - pixel_up.z);

		//float3 mag;
		//mag.x = grad_dx.x * grad_dx.x + grad_dy.x * grad_dy.x;
		//mag.y = grad_dx.y * grad_dx.y + grad_dy.y * grad_dy.y;
		//mag.z = grad_dx.z * grad_dx.z + grad_dy.z * grad_dy.z;
		v = dx*dx + dy*dy;
        v2 = dx2*dx2 + dy2*dy2;
        v3 = dx3*dx3 + dy3*dy3;


        // pick channel with strongest gradient
        if (v2 > v) {
    	    v = v2;
    	    dx = dx2;
    	    dy = dy2;
        } 
        if (v3 > v) {
    	    v = v3;
    	    dx = dx3;
    	    dy = dy3;
        }

        v = sqrtf(v);

        float dot = 0;
        float best_dot = 0;
        int best_o = 0, o = 0;
        
        for (o = 0; o < 9; o++) {
    	    dot = uu[o]*dx + vv[o]*dy;
    	    if (dot > best_dot) {
    	        best_dot = dot;
    	        best_o = o;
    	    } else if (-dot > best_dot) {
    	        best_dot = -dot;
    	        best_o = o+9;
    	    }
        }

		//d_pGradMag[pixelIdx].x = direction;
		//d_pGradMag[pixelIdx].y = magnitude;

        
        // TODO: cache these operations into shared memory
        if (ixp >= 0 && iyp >= 0) {
	        *(d_pHist+ ixp*block_0 + iyp + best_o*block_0*block_1) += 
                vx1*vy1*v;
        }

        if (ixp+1 < block_1 && iyp >= 0) {
	        *(d_pHist + (ixp+1)*block_0 + iyp + best_o*block_0*block_1) += 
                vx0*vy1*v;
        }

        if (ixp >= 0 && iyp+1 < block_0) {
	        *(d_pHist + ixp*block_0 + (iyp+1) + best_o*block_0*block_1) += 
	            vx1*vy0*v;
        }

        if (ixp+1 < block_1 && iyp+1 < block_0) {
	        *(d_pHist + (ixp+1)*block_0 + (iyp+1) + best_o*block_0*block_1) += 
	            vx0*vy0*v;
        }
	}
}

__global__ void d_voc_compute_hists2 /*__traceable__*/ (int dimx, int dimy, 
                                int block_0, int block_1, int sbin,
								int min_x, int min_y, int max_x, int max_y,
								float4* d_pImage, float* d_pHist)

{
    //__shared__ float s_Cells[(2+1) * (2+1) * 18];

    const int posx	= blockDim.x * blockIdx.x + threadIdx.x + min_x;// pixel pos within padded image
	const int posy	= blockDim.y * blockIdx.y + threadIdx.y + min_y;// pixel pos within padded image

	//__trace("Test", "int", posx);
	//__trace("Test", "int", posy);

	/*
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        for (int k=0; k < (2+1) * (2+1) * 18; ++k)
            s_Cells[k] = 0;
    }
    __syncthreads();
	*/

	/*
	float px1, px2, px3, px4;
	px1 = *(d_pImage + posy*4*dimx + 4*posx); 
	px2 = *(d_pImage+posy*4*dimx + 4*posx+1); 
	px3 = *(d_pImage+posy*4*dimx + 4*posx+2); 
	px4 = *(d_pImage+posy*4*dimx + 4*posx+3);
	__trace("grad", "float", px1); __trace("grad", "float", px2); __trace("grad", "float", px3); __trace("grad", "float", px4);
	*/
	
	//__trace("grad", "float", uu[0]);
	//__trace("grad", "float", uu[1]);
	//__trace("grad", "float", vv[2]);
	//__trace("grad", "float", vv[3]);

	/*
    if (posx == 0 && posy == 0)
    {
        uu[0] =     1.0000f;
    	uu[1] = 	0.9397f;
        uu[2] = 	0.7660f;
    	uu[3] = 	0.500f;
    	uu[4] = 	0.1736f; 
    	uu[5] = 	-0.1736f; 
    	uu[6] = 	-0.5000f;
    	uu[7] = 	-0.7660f;
    	uu[8] = 	-0.9397f;

        vv[0] =     0.0000f;
    	vv[1] = 	0.3420f;
        vv[2] = 	0.6428f;
    	vv[3] = 	0.8660f;
    	vv[4] = 	0.9848f; 
    	vv[5] = 	0.9848f; 
    	vv[6] = 	0.8660f;
    	vv[7] = 	0.6428f;
    	vv[8] = 	0.3420f;
    }

	__syncthreads();
	*/

	//const float xstep = 1.f / (width-2);
	//const float ystep = 1.f / (height-2);
	//const float xoff = xstep / 2.f;
	//const float yoff = ystep / 2.f;

	float4	pixel_up;
	float4	pixel_down;
	float4	pixel_left;
	float4	pixel_right;

	if((posx < max_x) && (posx > 0) 
        && (posy < max_y) && (posy > 0)) {

        //float* pixel_pos = d_pImage + posy * 4 * dimx + 4 * posx;
        float4* pixel_pos = d_pImage + posy*dimx + posx;
        //float4* pixel_pos = d_pImage + posx*dimy + posy;
        /*
        pixel_left.x = *(pixel_pos - 4);
        pixel_left.y = *(pixel_pos - 3);
        pixel_left.z = *(pixel_pos - 2);

        pixel_right.x = *(pixel_pos + 4);
        pixel_right.y = *(pixel_pos + 5);
        pixel_right.z = *(pixel_pos + 6);

        pixel_down.x = *(pixel_pos + 4 * dimx);
        pixel_down.y = *(pixel_pos + 4 * dimx + 1);
        pixel_down.z = *(pixel_pos + 4 * dimx + 2);

        pixel_up.x = *(pixel_pos - 4 * dimx);
        pixel_up.y = *(pixel_pos - 4 * dimx + 1);
        pixel_up.z = *(pixel_pos - 4 * dimx + 2);
        */
		//__trace("left", "float", pixel_left.x);
		//__trace("right", "float", pixel_right.x);
		//__trace("up", "float", pixel_up.x);
		//__trace("down", "float", pixel_down.x);
          
        pixel_down = *(pixel_pos + dimx);
        pixel_up = *(pixel_pos - dimx);
        pixel_left = *(pixel_pos - 1);
        pixel_right = *(pixel_pos + 1);
		

        ftype dy, dy2, dy3, dx, dx2, dx3, v, v2, v3;
        /*
		dy = (pixel_right.x - pixel_left.x);
		dy2 = (pixel_right.y - pixel_left.y);
		dy3 = (pixel_right.z - pixel_left.z);

		dx = (pixel_down.x - pixel_up.x);
		dx2 = (pixel_down.y - pixel_up.y);
		dx3 = (pixel_down.z - pixel_up.z);
        */
        
        
        // because actually the image is column-major
        dx = (pixel_right.x - pixel_left.x);
		dx2 = (pixel_right.y - pixel_left.y);
		dx3 = (pixel_right.z - pixel_left.z);

		dy = (pixel_down.x - pixel_up.x);
		dy2 = (pixel_down.y - pixel_up.y);
		dy3 = (pixel_down.z - pixel_up.z);
        
		
		v = dx*dx + dy*dy;
        v2 = dx2*dx2 + dy2*dy2;
        v3 = dx3*dx3 + dy3*dy3;

		//__trace("grad", "float", v);
		//__trace("grad", "float", v2);
		//__trace("grad", "float", v3);

        // pick channel with strongest gradient
        if (v2 > v) {
    	    v = v2;
    	    dx = dx2;
    	    dy = dy2;
        } 
        if (v3 > v) {
    	    v = v3;
    	    dx = dx3;
    	    dy = dy3;
        }

        v = sqrtf(v);

		//__trace("v", "float", v);

        float dot = 0;
        float best_dot = 0;
        int best_o = 0, o = 0;
        
        for (o = 0; o < 9; o++) {
    	    dot = uu[o]*dx + vv[o]*dy;
    	    if (dot > best_dot) {
    	        best_dot = dot;
    	        best_o = o;
    	    } else if (-dot > best_dot) {
    	        best_dot = -dot;
    	        best_o = o+9;
    	    }
        }

		//__trace("best_dot", "float", best_dot);
		//__trace("best_o", "int", best_o);

		//d_pGradMag[pixelIdx].x = direction;
		//d_pGradMag[pixelIdx].y = magnitude;

		// for linear interpolation to 4 histograms around pixel
		ftype xp = ((ftype)posx+0.5f)/(ftype)sbin - 0.5f;
		ftype yp = ((ftype)posy+0.5f)/(ftype)sbin - 0.5f;
		int ixp = (int)floor(xp);
		int iyp = (int)floor(yp);
		ftype vx0 = xp-ixp;
		ftype vy0 = yp-iyp;
		ftype vx1 = 1.0f-vx0;
		ftype vy1 = 1.0f-vy0;

		//__trace("iyp", "int", iyp);
		//__trace("ixp", "int", ixp);

        ftype cxp = ((ftype)(threadIdx.x)+0.5f)/(ftype)sbin - 0.5f;
        ftype cyp = ((ftype)(threadIdx.y)+0.5f)/(ftype)sbin - 0.5f;
        int icxp = (int)floor(cxp) + 1;
		int icyp = (int)floor(cyp) + 1;
        
        // TODO: cache these operations into shared memory
        if (ixp >= 0 && iyp >= 0) {
	        //*(d_pHist+ ixp*block_0 + iyp + best_o*block_0*block_1) += vx1*vy1*v;
			atomicAdd(d_pHist+ ixp*block_0 + iyp + best_o*block_0*block_1, vx1*vy1*v);
            //atomicAdd(s_Cells+ icxp*3 + icyp + best_o*3*3, vx1*vy1*v);
			//__trace("1", "float", vx1*vy1*v);
			//__syncthreads();
        }
		
        if (ixp+1 < block_1 && iyp >= 0) {
	        //*(d_pHist + (ixp+1)*block_0 + iyp + best_o*block_0*block_1) += vx0*vy1*v;
			atomicAdd(d_pHist + (ixp+1)*block_0 + iyp + best_o*block_0*block_1, vx0*vy1*v);
            //atomicAdd(s_Cells+ (icxp+1)*3 + icyp + best_o*3*3, vx0*vy1*v);
			//__trace("2", "float", vx0*vy1*v);
			//__syncthreads();
        }

        if (ixp >= 0 && iyp+1 < block_0) {
	        //*(d_pHist + ixp*block_0 + (iyp+1) + best_o*block_0*block_1) += vx1*vy0*v;
			atomicAdd(d_pHist + ixp*block_0 + (iyp+1) + best_o*block_0*block_1, vx1*vy0*v);
            //atomicAdd(s_Cells+ icxp*3 + (icyp+1) + best_o*3*3, vx1*vy0*v);
			//__trace("3", "float", vx1*vy0*v);
			//__syncthreads();
        }

        if (ixp+1 < block_1 && iyp+1 < block_0) {
	        //*(d_pHist + (ixp+1)*block_0 + (iyp+1) + best_o*block_0*block_1) += vx0*vy0*v;
			atomicAdd(d_pHist + (ixp+1)*block_0 + (iyp+1) + best_o*block_0*block_1, vx0*vy0*v);
            //atomicAdd(s_Cells+ (icxp+1)*3 + (icyp+1) + best_o*3*3, vx0*vy0*v);
			//__trace("4", "float", *(d_pHist + (ixp+1)*block_0 + (iyp+1) + best_o*block_0*block_1));
			//__trace("4", "float", vx0*vy0*v);
			//__syncthreads();
        }
		/*
        __syncthreads();
        
        if (threadIdx.x == 15 && threadIdx.y == 15)
        {
            for (o = 0; o < 18; ++o)
            {
                if ((ixp-1)>=0 && (iyp-1)>=0)
                    atomicAdd(d_pHist + (ixp-1)*block_0 + (iyp-1) + o*block_0*block_1,
                        s_Cells[(0)*3 + (0) + best_o*3*3]);
                if ((ixp-1)>=0)
                    atomicAdd(d_pHist + (ixp-1)*block_0 + (iyp) + o*block_0*block_1,
                        s_Cells[(0)*3 + (1) + best_o*3*3]);
                if ((ixp-1)>=0 && (iyp+1)<block_0)
                    atomicAdd(d_pHist + (ixp-1)*block_0 + (iyp+1) + o*block_0*block_1,
                        s_Cells[(0)*3 + (2) + best_o*3*3]);
                if ((iyp-1)>=0)
                    atomicAdd(d_pHist + (ixp)*block_0 + (iyp-1) + o*block_0*block_1,
                        s_Cells[(1)*3 + (0) + best_o*3*3]);
                
                atomicAdd(d_pHist + (ixp)*block_0 + (iyp) + o*block_0*block_1,
                    s_Cells[(1)*3 + (1) + best_o*3*3]);
                
                if ((iyp+1)<block_0)
                    atomicAdd(d_pHist + (ixp)*block_0 + (iyp+1) + o*block_0*block_1,
                        s_Cells[(1)*3 + (2) + best_o*3*3]);
                if ((ixp+1)<block_1 && (iyp-1)>=0)
                    atomicAdd(d_pHist + (ixp+1)*block_0 + (iyp-1) + o*block_0*block_1,
                        s_Cells[(2)*3 + (0) + best_o*3*3]);
                if ((ixp+1)<block_1)
                    atomicAdd(d_pHist + (ixp+1)*block_0 + (iyp) + o*block_0*block_1,
                        s_Cells[(2)*3 + (1) + best_o*3*3]);
                if ((ixp+1)<block_1 && (iyp+1) < block_0)
                    atomicAdd(d_pHist + (ixp+1)*block_0 + (iyp+1) + o*block_0*block_1,
                        s_Cells[(2)*3 + (2) + best_o*3*3]);
            }
        }
		*/
		
	}
}


int voc_prepare_image(const ftype* h_pImg, int width, int height)
{
    int ret = -1;

	float* h_uu = (float*)malloc(sizeof(float)*9);
    float* h_vv = (float*)malloc(sizeof(float)*9);
    h_uu[0] =   1.0000f;
    h_uu[1] = 	0.9397f;
    h_uu[2] = 	0.7660f;
    h_uu[3] = 	0.500f;
    h_uu[4] = 	0.1736f; 
    h_uu[5] = 	-0.1736f; 
    h_uu[6] = 	-0.5000f;
    h_uu[7] = 	-0.7660f;
    h_uu[8] = 	-0.9397f;

    h_vv[0] =   0.0000f;
    h_vv[1] = 	0.3420f;
    h_vv[2] = 	0.6428f;
    h_vv[3] = 	0.8660f;
    h_vv[4] = 	0.9848f; 
    h_vv[5] = 	0.9848f; 
    h_vv[6] = 	0.8660f;
    h_vv[7] = 	0.6428f;
    h_vv[8] = 	0.3420f;

    cudaMemcpyToSymbol(uu, h_uu, sizeof(float)*9);
    cudaMemcpyToSymbol(vv, h_vv, sizeof(float)*9);

	free(h_uu);
    free(h_vv);

	// allocate array - copy image there - bind it to texture
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	cudaMallocArray( &normalized_image_array, &channelDesc, width, height);
		ONFAIL("malloc array\n");

	cudaMemcpyToArray(normalized_image_array, 0, 0, h_pImg, width * height * sizeof(float4), cudaMemcpyHostToDevice);
		ONFAIL("memcpy to array\n");

	t_normalized_image_texture.addressMode[0] = cudaAddressModeClamp;
	t_normalized_image_texture.addressMode[1] = cudaAddressModeClamp;
	t_normalized_image_texture.filterMode = cudaFilterModeLinear;
	t_normalized_image_texture.normalized = true;

	cudaBindTextureToArray( t_normalized_image_texture, normalized_image_array, channelDesc);
		ONFAIL("bind tex to array\n");

    ret = 0;

	return ret;
}

int voc_prepare_image2(const ftype* h_pImg, int width, int height, int sbin)
{   
	float* h_uu = (float*)malloc(sizeof(float)*9);
    float* h_vv = (float*)malloc(sizeof(float)*9);
    h_uu[0] =   1.0000f;
    h_uu[1] = 	0.9397f;
    h_uu[2] = 	0.7660f;
    h_uu[3] = 	0.500f;
    h_uu[4] = 	0.1736f; 
    h_uu[5] = 	-0.1736f; 
    h_uu[6] = 	-0.5000f;
    h_uu[7] = 	-0.7660f;
    h_uu[8] = 	-0.9397f;

    h_vv[0] =   0.0000f;
    h_vv[1] = 	0.3420f;
    h_vv[2] = 	0.6428f;
    h_vv[3] = 	0.8660f;
    h_vv[4] = 	0.9848f; 
    h_vv[5] = 	0.9848f; 
    h_vv[6] = 	0.8660f;
    h_vv[7] = 	0.6428f;
    h_vv[8] = 	0.3420f;


    cudaMemcpyToSymbol(uu, h_uu, sizeof(float)*9);
    cudaMemcpyToSymbol(vv, h_vv, sizeof(float)*9);

    free(h_uu);
    free(h_vv);

    // original image
	cudaMalloc((void**)&d_pImage, width * height * sizeof(float4));
	cudaMemcpy(d_pImage, (float4*)h_pImg, width * height * sizeof(float4), cudaMemcpyHostToDevice);
    
    // rescaled image
    cudaMalloc((void**)&d_RescaledImage, width * height * sizeof(float4));
    
    
    // texture
    channelDescDownscale = cudaCreateChannelDesc<float4>();
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false;
    
    // texture array
    isAlloc = true;
    int maxoct = (int)log2(min(height, width) / (float)(sbin)) - 1;
    //printf("maxoct is %d\n", maxoct);
    gmaxoct = maxoct;
    imageArray = (cudaArray**)malloc(sizeof(cudaArray*) * maxoct);

    int iwid, ihei;
    float scale = 1.0f;
    dim3 hThreadSize, hBlockSize;
	hThreadSize = dim3(16, 16);
    hBlockSize = dim3(iDivUp(width, hThreadSize.x), iDivUp(height, hThreadSize.y));

    int lastwid = width, lasthei = height;

    for (int o = 0; o < maxoct; o++)
    {
        iwid = (int)round((float)width*scale);
        ihei = (int)round((float)height*scale);
        //printf("preparing reference images... iwid = %d, ihei = %d\n", iwid, ihei);
        
        cudaChannelFormatDesc chDes = cudaCreateChannelDesc<float4>();
        cutilSafeCall(cudaMallocArray(&(imageArray[o]), &chDes, iwid, ihei) );

        if (o == 0)
            cutilSafeCall(cudaMemcpyToArray(imageArray[o], 0, 0, d_pImage, sizeof(float4)*width*height, cudaMemcpyDeviceToDevice));
        else{
            //voc_resize_image(width, height, iwid, ihei, 0); // or o-1
            voc_resize_image(lastwid, lasthei, iwid, ihei, o-1);
            cutilSafeCall(cudaMemcpyToArray(imageArray[o], 0, 0, d_RescaledImage, sizeof(float4)*iwid*ihei, cudaMemcpyDeviceToDevice));
        }
        scale /= 2.0f;
        lastwid = iwid;
        lasthei = ihei;
    }
    //imageArr = imageArray[0];


/*
    cutilSafeCall(cudaMallocArray(&imageArray, &channelDescDownscale, 
        width, height) );
	cutilSafeCall(cudaMemcpyToArray(imageArray, 0, 0, d_pImage, 
        sizeof(float4) * width * height, cudaMemcpyDeviceToDevice));
*/	
    return 0;

}

int voc_prepare_image3(float* h_pImg, int width, int height)
{
	printf("In voc_prepare_image3: first pixel = %f, %f, %f, %f\n", *(h_pImg), *(h_pImg+1), *(h_pImg+2), *(h_pImg+3));
	float4* tmp = (float4*)h_pImg;
	printf("In voc_prepare_image3: first pixel = %f, %f, %f, %f\n", tmp[0].x, tmp[0].y, tmp[0].z, tmp[0].w);

    cudaMalloc((void**) &d_pImage4, sizeof(float4) * width * height);
    cudaMemcpy(d_pImage4, tmp, sizeof(float4) * width * height, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_pGradients, sizeof(float2) * width * height);
    cudaMemset(d_pGradients, 0, sizeof(float2) * width * height);
    return 0;
}

int voc_destroy_image2()
{
	if( d_pImage ) {
		cudaFree(d_pImage);
		d_pImage = 0;
	}
/*
    if (isAlloc){ 
        cutilSafeCall(cudaFreeArray(imageArray));
        isAlloc = false;
    }
*/
    //int maxoct = (int)log2(min(height, width) / (float)(sbin)) - 1;
    int maxoct = gmaxoct;
    for (int o = 0; o < maxoct; o++)
        cutilSafeCall(cudaFreeArray(imageArray[o]));
    free(imageArray);

    if( d_RescaledImage) {
		cudaFree(d_RescaledImage);
		d_RescaledImage = 0;
	}
	return 0;
}

int voc_destroy_image3()
{
    if( d_pImage4 ) {
		cudaFree(d_pImage4);
		d_pImage4 = 0;
	}
    if( d_pGradients ) {
		cudaFree(d_pGradients);
		d_pGradients = 0;
	}
	return 0;
}



// here width and height are rescaled, every octave uses the same referenced texture 
__host__ void voc_resize_image(int width, int height, int res_wid, int res_hei, int oct)
{
    dim3 hThreadSize, hBlockSize;

	hThreadSize = dim3(16, 16);
    hBlockSize = dim3(iDivUp(width, hThreadSize.x), iDivUp(height, hThreadSize.y));

	// Binding every time?
    cudaArray* imageArr = imageArray[oct];
    cutilSafeCall(cudaBindTextureToArray(tex, imageArr, channelDescDownscale));

    // should we reallocate the d_RescaledImage?
    //cutilSafeCall(cudaFree(d_RescaledImage));
    //cutilSafeCall(cudaMalloc((void**)&d_RescaledImage, res_wid * res_hei * sizeof(float4)));
    cutilSafeCall(cudaMemset(d_RescaledImage, 0, res_wid * res_hei * sizeof(float4)));
    
    float invScalX = (float)width / (float)res_wid;
    float invScalY = (float)height / (float)res_hei;
    
    if (invScalX == 1.0f && invScalY == 1.0f && oct == 0)
    {
        cutilSafeCall(cudaMemcpy(d_RescaledImage, d_pImage, sizeof(float4)*res_wid*res_hei, cudaMemcpyDeviceToDevice));
        //printf("invX=%f,invY=%f,res_wid=%d,res_hei=%d\n", invScalX, invScalY, res_wid, res_hei);
    }
    else
    {
	    //d_resize_bicubic<<<hBlockSize, hThreadSize>>>(d_RescaledImage, d_pImage, res_wid, res_hei, invScalX, invScalY);
	    d_resize_bicubic<<<hBlockSize, hThreadSize>>>(d_RescaledImage, res_wid, res_hei, invScalX, invScalY);
    }

    cutilSafeCall(cudaUnbindTexture(tex));
    
}

__host__ void voc_debug_resize_image(int width, int height, int res_wid, int res_hei, int oct, float* res_img)
{
    dim3 hThreadSize, hBlockSize;

	hThreadSize = dim3(16, 16);
    hBlockSize = dim3(iDivUp(width, hThreadSize.x), iDivUp(height, hThreadSize.y));

	// Binding every time?
    cudaArray* imageArr = imageArray[oct];
    cudaChannelFormatDesc chDes = cudaCreateChannelDesc<float4>();

    cutilSafeCall(cudaBindTextureToArray(tex, imageArr, chDes));

    // should we reallocate the d_RescaledImage?
    //cutilSafeCall(cudaMemset(d_RescaledImage, 0, width * height * sizeof(float4)));
    cutilSafeCall(cudaMemset(d_RescaledImage, 0, res_wid * res_hei * sizeof(float4)));
    float invScalX = (float)width / (float)res_wid;
    float invScalY = (float)height / (float)res_hei;
	//d_resize_bicubic<<<hBlockSize, hThreadSize>>>(d_RescaledImage, d_pImage, res_wid, res_hei, invScalX, invScalY);

    if (invScalX == 1.0f && invScalY == 1.0f && oct == 0)
    {
        cutilSafeCall(cudaMemcpy(d_RescaledImage, d_pImage, sizeof(float4)*res_wid*res_hei, cudaMemcpyDeviceToDevice));
        //printf("invX=%f,invY=%f,res_wid=%d,res_hei=%d\n", invScalX, invScalY, res_wid, res_hei);
    }
    else
    {
	//d_resize_bicubic<<<hBlockSize, hThreadSize>>>(d_RescaledImage, d_pImage, res_wid, res_hei, invScalX, invScalY);
	    d_resize_bicubic<<<hBlockSize, hThreadSize>>>(d_RescaledImage, res_wid, res_hei, invScalX, invScalY);
    }
    cutilSafeCall(cudaUnbindTexture(tex));

    cutilSafeCall(cudaMemcpy(res_img, (float*)d_RescaledImage, sizeof(float) * res_wid * res_hei * 4, cudaMemcpyDeviceToHost));
    //cutilSafeCall(cudaMemcpyFromArray(res_img, imageArr, 0, 0, width*height*sizeof(float4), cudaMemcpyDeviceToHost));
}


__host__ void voc_set_octref(int ref_dimx, int ref_dimy, int oct)
{

/*
    if (isAlloc)
			cutilSafeCall(cudaFreeArray(imageArray));
	cutilSafeCall(cudaMallocArray(&imageArray, &channelDescDownscale, ref_dimx, ref_dimy) );
	cutilSafeCall(cudaMemcpyToArray(imageArray, 0, 0, d_RescaledImage, 
        sizeof(float4) * ref_dimx * ref_dimy, cudaMemcpyDeviceToDevice));
*/

    //cutilSafeCall(cudaMemcpyToArray(imageArray, 0, 0, d_pImage, 
        //sizeof(float4) * ref_dimx * ref_dimy, cudaMemcpyDeviceToDevice));

    //imageArr = imageArray[oct];

}

__host__ int voc_compute_gradients( int width, int height, int sbin,
                                int blocks_0, int blocks_1,
								int padX, int padY, float* d_pHist)
{
	// start kernel to compute gradient directions & magnitudes
	const int TX = 16;
	const int TY = 16;
	dim3 threads(TX, TY);
	dim3 grid( (int)ceil(width/((float)TX)), (int)ceil(height/((float)TY)) );

    //int blocks[2];
    //blocks[0] = (int)round((ftype)height/(ftype)sbin);
    //blocks[1] = (int)round((ftype)width/(ftype)sbin);

    int visible[2];
    visible[0] = blocks_0*sbin; // y
    visible[1] = blocks_1*sbin; // x

    int x_ubound, y_ubound, x_lbound, y_lbound;
    x_lbound = 0;
    y_lbound = 0;
    x_ubound = (visible[1]-1) <= (width-1) ? (visible[1]-1) : (width-1);
    y_ubound = (visible[0]-1) <= (height-1) ? (visible[0]-1) : (height-1);

//	printf("\ncompute_gradients:\n");
//	printf("img dim: %d x %d\n", paddedWidth, paddedHeight);
//	printf("grid: %d, %d, %d\n", grid.x, grid.y, grsid.z);

    /*
	float* h_uu = (float*)malloc(sizeof(float)*9);
    float* h_vv = (float*)malloc(sizeof(float)*9);
    h_uu[0] =   1.0000f;
    h_uu[1] = 	0.9397f;
    h_uu[2] = 	0.7660f;
    h_uu[3] = 	0.500f;
    h_uu[4] = 	0.1736f; 
    h_uu[5] = 	-0.1736f; 
    h_uu[6] = 	-0.5000f;
    h_uu[7] = 	-0.7660f;
    h_uu[8] = 	-0.9397f;

    h_vv[0] =   0.0000f;
    h_vv[1] = 	0.3420f;
    h_vv[2] = 	0.6428f;
    h_vv[3] = 	0.8660f;
    h_vv[4] = 	0.9848f; 
    h_vv[5] = 	0.9848f; 
    h_vv[6] = 	0.8660f;
    h_vv[7] = 	0.6428f;
    h_vv[8] = 	0.3420f;


    cudaMemcpyToSymbol(uu, h_uu, sizeof(float)*9);
    cudaMemcpyToSymbol(vv, h_vv, sizeof(float)*9);
    */

	/*
	cudaMalloc((void**)&d_uu, sizeof(float)*9);
	cudaMalloc((void**)&d_vv, sizeof(float)*9);
	cudaMemcpy(d_uu, h_uu, sizeof(float)*9, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vv, h_vv, sizeof(float)*9, cudaMemcpyHostToDevice);
	*/

	//cudaPrintfInit(sizeof(int)*1000*1000);
	//cudaPrintfInit();

	//INITIALIZE_TRACE_DATA();

#ifdef DEBUG_TIME_EACH_STEP
	Timer tt;
	startTimer(&tt);
#endif
    
	d_voc_compute_hists2<<< grid , threads >>> /*__traceable_call__*/ (width, height,
	                                            blocks_0,blocks_1, sbin,
												x_lbound, y_lbound, 
												x_ubound, y_ubound, 
												/*d_pImage*/d_RescaledImage, d_pHist);
		ONFAIL("d_voc_compute_hists failed\n");

#ifdef DEBUG_TIME_EACH_STEP
	stopTimer(&tt);
	printf("time in voc_compute_gradients = %f\n", getTimerValue(&tt));
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

	//cudaPrintfDisplay(stdout, true);
	//cudaPrintfEnd();
        
	//free(h_uu);
	//free(h_vv);

    //ComputeColorGradients(d_pImage4, d_pGradients, width, height);
        
//#define DEBUG_DUMP_GRADIENTS
#ifdef DEBUG_DUMP_GRADIENTS
    
	float *h_pGradMag = (float*) malloc(blocks_0*blocks_1*18*sizeof(float));
	if(!h_pGradMag) {
		printf("h_pGradMag: malloc failed \n");
		return -1;
	}
	// copy results
	cudaMemcpy(h_pGradMag, d_pHist, sizeof(float)*blocks_0*blocks_1*18, cudaMemcpyDeviceToHost);

    int out[3];
    out[0] = max(blocks_0-2, 0);
    out[1] = max(blocks_1-2, 0);
    out[2] = 32;

	char filname[50];
    printf("writing cell_hists_gpu_res_%d_%d.txt.......\n", out[1], out[0]);
    sprintf(filname, "cell_hists_gpu_res_%d_%d.txt", out[1], out[0]);

	// write complete output to file
	FILE* fp = fopen(filname, "w");
	if(!fp) printf("failed to open output file: fmag\n");

    int ci, cj, cb;
    for (ci = 0; ci < blocks_0; ci++)
    {
      for (cj = 0; cj < blocks_1; cj++)
      {
          fprintf(fp, "(i=%d,j=%d)\n", ci, cj);
          for (cb = 0; cb < 18; cb++)
          {
              fprintf(fp, "%f ", *(h_pGradMag + cj * blocks_0 + ci + cb*blocks_0*blocks_1));
          }
          fprintf(fp, "\n");
      }

    }

	fclose(fp);
    free(h_pGradMag);
    
    
#endif

	return 0;
}

