#ifndef __BLOCKS_H__
#define __BLOCKS_H__

extern "C++" __host__ int voc_compute_block_energy(int blocks_0, int blocks_1,
                        float* d_pHists, float* d_pNorms);

extern "C++" __host__ int voc_compute_features(int blocks_0, int blocks_1, 
                                        float* d_pHists, float* d_pNorms, 
                                        float* d_pOut);


#endif
