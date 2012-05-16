#include <math.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <QImage>

#include "process.h"
#include "global.h"
#include "timer.h"

using namespace std;

struct alphainfo {
    int si, di;
    float alpha;
};

// copy src into dst using precomputed interpolation values
void alphacopy(float *src, float *dst, struct alphainfo *ofs, int n) {
    struct alphainfo *end = ofs + n;
    while (ofs != end) {
        dst[ofs->di] += ofs->alpha * src[ofs->si];
        ofs++;
    }
}

// resize along each column
// result is transposed, so we can apply it twice for a complete resize
void resize1dtran(float *src, int sheight, float *dst, int dheight, int width, int chan) {
    float scale = (float)dheight/(float)sheight;
    float invscale = (float)sheight/(float)dheight;
    
    // we cache the interpolation values since they can be 
    // shared among different columns
    int len = (int)ceil(dheight*invscale) + 2*dheight;
    struct alphainfo ofs[len];
    int k = 0;
    int dy,sy;
    for (dy = 0; dy < dheight; dy++) {
        float fsy1 = dy * invscale;
        float fsy2 = fsy1 + invscale;
        int sy1 = (int)ceil(fsy1);
        int sy2 = (int)floor(fsy2);       

        if (sy1 - fsy1 > 1e-3) {
            assert(k < len);
            assert(sy1 >= 0);
            ofs[k].di = dy*width;
            ofs[k].si = sy1-1;
            ofs[k++].alpha = (sy1 - fsy1) * scale;
        }
        //printf("stage1 \n");

        for (sy = sy1; sy < sy2; sy++) {
            assert(k < len);
            assert(sy < sheight);
            ofs[k].di = dy*width;
            ofs[k].si = sy;
            ofs[k++].alpha = scale;
        }

        //printf("stage2 \n");

        if (fsy2 - sy2 > 1e-3) {
            assert(k < len);
            assert(sy2 < sheight);
            ofs[k].di = dy*width;
            ofs[k].si = sy2;
            ofs[k++].alpha = (fsy2 - sy2) * scale;
        }
    }

    // resize each column of each color channel
    bzero(dst, chan*width*dheight*sizeof(float));
    int c,x;
    for (c = 0; c < chan; c++) {
        for (x = 0; x < width; x++) {
            float *s = src + c*width*sheight + x*sheight;
            float *d = dst + c*width*dheight + x;
            alphacopy(s, d, ofs, k);
        }
    }
}

// The rescaled image has the same channels as the original image
void resize_im(float* src, int sh, int sw, int sc, float* dst, int res_dimy, int res_dimx)
{
	//int res_dimx = (int)round((float)sw*scale);
	//int res_dimy = (int)round((float)sh*scale);

	//float* rescaledim = (float*)malloc(sizeof(float) * sc*res_dimx*res_dimy);
	float* rescaledim = dst;
	float* tempim = (float*)malloc(sizeof(float) * sc*sw*res_dimy);
	resize1dtran(src, sh, tempim, res_dimy, sw, sc);
	resize1dtran(tempim, sw, rescaledim, res_dimx, res_dimy, sc);

	free(tempim);
}

int main(int argc, char** argv)
{
	QImage img;
	string example("../person_and_bike_006.png");
	img.load(example.c_str());
	if(img.isNull()) {
		printf("loading failed!\n");
		return -1;
	}

	printf("Color Depth = %d\n", img.depth());
	printf("Bytes Count = %d\n", img.byteCount());

	int height    = img.height();
    int width     = img.width();
	int bytePerLine = img.bytesPerLine();

	printf("height = %d, width = %d, byteperline = %d\n", height, width, bytePerLine);

	/*
	QImage copy_image(width, height, QImage::Format_RGB32);
 	QRgb value;
	//ftype* data = (ftype*)malloc(sizeof(ftype)*3*height*width);
	unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char)*3*height*width);
	unsigned char* img_data = img.bits();
	for (int i = 0; i < height; ++i)
		for (int j = 0; j < width; ++j)
	{
		*(data+i*3*width+3*j + 0) = *(img_data+i*bytePerLine+j*4 + 0);
		*(data+i*3*width+3*j + 1) = *(img_data+i*bytePerLine+j*4 + 1);
		*(data+i*3*width+3*j + 2) = *(img_data+i*bytePerLine+j*4 + 2);
		//BGR
		value = qRgb(*(data+i*3*width+3*j + 2), *(data+i*3*width+3*j + 1), *(data+i*3*width+3*j + 0));
		copy_image.setPixel(j, i, value);
	}

	copy_image.save("./copy_image.png", "PNG");
	
	return 0;
	*/

	ftype* data = (ftype*)malloc(sizeof(ftype)*4*height*width);
	//unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char)*3*height*width);
	unsigned char* img_data = img.bits();
	printf("In main: first pixel = %d, %d, %d, %d\n", *(img_data), *(img_data+1), *(img_data+2), *(img_data+3));
	
	for (int i = 0; i < height; ++i)
		for (int j = 0; j < width; ++j)
	{
		*(data+i * 4*width + 4*j + 0) = *(img_data+i*bytePerLine+j*4 + 0)*1.0f;
		*(data+i * 4*width + 4*j + 1) = *(img_data+i*bytePerLine+j*4 + 1)*1.0f;
		*(data+i * 4*width + 4*j + 2) = *(img_data+i*bytePerLine+j*4 + 2)*1.0f;
		*(data+i * 4*width + 4*j + 3) = *(img_data+i*bytePerLine+j*4 + 3)*1.0f;
		//BGR
		//value = qRgb(*(data+i*3*width+3*j + 2), *(data+i*3*width+3*j + 1), *(data+i*3*width+3*j + 0));
		//copy_image.setPixel(j, i, value);
	}
	
	int maxoct = (int)log2(min(height, width) / (8.0)) - 1;
	printf("maxoct = %d\n", maxoct);

	int blocks[2];
	int hx, hy, hz;
	int interv = 10;
	int res_dimx = width, res_dimy = height;
	float scale = 1.0f;
	float* descriptor;
	ftype* rescaledim = (ftype*)malloc(sizeof(ftype)*4*height*width);
    float* oData = (ftype*)malloc(sizeof(ftype)*4*height*width);

    // first resizing
    memcpy(rescaledim, data, sizeof(ftype)*4*height*width);
    memcpy(oData, data, sizeof(ftype)*4*height*width);

    float** featArr = (float**)malloc(sizeof(float*) * maxoct*interv);
    int* eleSize = (int*)malloc(sizeof(int) * maxoct*interv);
    
    int o, i;

	Timer tt;
	startTimer(&tt);

	for (o = 0; o < maxoct; ++o)
	{
		for (i = 0; i < interv; ++i)
		{
			//res_dimx = (int)round((float)width*scale);
			//res_dimy = (int)round((float)height*scale);
			//width = res_dimx;
			//height = res_dimy;

			//printf("--------------------------------------\n");
			//printf("scale = %f\n", scale);
			//printf("width = %d, height = %d\n", res_dimx, res_dimy);

			blocks[0] = (int)round((float)res_dimy/8.0);
    		blocks[1] = (int)round((float)res_dimx/8.0);	

			hy = max(blocks[0]-2, 0);
    		hx = max(blocks[1]-2, 0);
    		hz = 31;

			descriptor = (float*)malloc(sizeof(float) * hx*hy*hz);
			
            // rescaledim and res_dimy, res_dimx are corresponding 
			//process(rescaledim, res_dimy, res_dimx, 8, descriptor, hy, hx, hz);

            // get dim and allocate mem
            *(eleSize + o*interv + i) = hx*hy*hz;
            *(featArr + o*interv + i) = (float*)malloc(sizeof(float) * hx*hy*hz);

#ifdef DEBUG_process
            int out[3];
            out[0] = max(blocks[0]-2, 0);
            out[1] = max(blocks[1]-2, 0);
            out[2] = 31;
            
            char filname[50];
            sprintf(filname, "cell_feats_%d_%d.txt", o, i);
            
            //FILE* fp = fopen("cell_feats.txt", "w");
            FILE* fp = fopen(filname, "w");
	        if(!fp) 
                printf("failed to open output file: fmag\n");

            int ci, cj, cb;
            for (ci = 0; ci < out[0]; ci++)
            {
              for (cj = 0; cj < out[1]; cj++)
              {
                  fprintf(fp, "(i=%d,j=%d)\n", ci, cj);
                  for (cb = 0; cb < 31; cb++)
                  {
                      fprintf(fp, "%f ", *(descriptor + cj * out[0] + ci + cb*out[0]*out[1]));
                  }
                  fprintf(fp, "\n");
              }

            }

	        fclose(fp);
#endif

			scale = pow(2, -1.0*(i+1)/interv);
			
			res_dimx = (int)round((float)width*scale);
			res_dimy = (int)round((float)height*scale);
			resize_im(oData, height, width, 4, rescaledim, res_dimy, res_dimx);

			free(descriptor);
		}
		width = res_dimx;
		height = res_dimy;
        memcpy(oData, rescaledim, sizeof(float)*res_dimx*res_dimy);
	}

	stopTimer(&tt);
	//printf("Features = %f\n", getTimerValue(&tt));

    printf("--------------------------------------\n");
    printf("Finished size initialization\n");
    printf("--------------------------------------\n");

    startTimer(&tt);
    process_all_scales(data, img.height(), img.width(), 8, featArr, maxoct, interv, eleSize);
    stopTimer(&tt);
	printf("Features = %f\n", getTimerValue(&tt));


    for (o = 0; o < maxoct; ++o)
	{
		for (i = 0; i < interv; ++i)
		{
            free(*(featArr + o*interv + i));
        }
    }
    free(featArr);
    free(eleSize);

    free(data);
    free(oData);
    free(rescaledim);

	return 0;
	
	
}
