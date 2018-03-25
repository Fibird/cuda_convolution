#include <cuda_runtime.h>
#include <stdio.h>
#include "waveformat/waveformat.h"

#define TILE_SIZE 1024
#define MAX_MASK_WIDTH 5

typedef short element;

__constant__ short M[MAX_MASK_WIDTH];

__global__ void convolution_1D_basic_kernel(element *N, element *P, int Mask_Width, int Width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ element N_ds[TILE_SIZE + MAX_MASK_WIDTH - 1];

    int n = MAX_MASK_WIDTH / 2;

    int halo_index_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x >= blockDim.x - n)
    {
        N_ds[threadIdx.x - (blockDim.x - n)] = 
        (halo_index_left < 0) ? 0 : N[halo_index_left];
    }

    N_ds[n + threadIdx.x] = N[blockIdx.x * blockDim.x + threadIdx.x];

    int halo_index_right = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x < n)
    {
        N_ds[n + blockDim.x + threadIdx.x] = 
        (halo_index_right >= Width) ? 0 : N[halo_index_right];
    }

    __syncthreads();

    element Pvalue = 0;
    for (int j = 0; j < Mask_Width; j++)
    {
        Pvalue += N_ds[threadIdx.x + j] * M[j];
    }
    P[i] = Pvalue / MAX_MASK_WIDTH; 
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Please specify file name!\n");
        exit(EXIT_FAILURE);
    }
    short *data = NULL;
    short *data_clear = NULL;
    short *dev_data, *dev_data_clear;

    const short h_M[MAX_MASK_WIDTH] = {1, 1, 1, 1, 1};
    cudaMemcpyToSymbol(M, h_M, MAX_MASK_WIDTH * sizeof(short));

    int size;
    waveFormat fmt;
    FILE *f = NULL;
    
    f = fopen(argv[1], "rb");
    if (!f)
    {
        printf("Open input file failed!\n");
        return -1;
    }
    fmt = readWaveHeader(f);
    size = fmt.data_size;
    data = (short*)malloc(sizeof(short) * size);
    data_clear = (short*)malloc(sizeof(short) * size);

    fseek(f, 44L, SEEK_SET);
    fread(data, sizeof(short), size, f);
    if (f)
    {
        fclose(f);
        f = NULL;
    }
    
    cudaMalloc((void**)&dev_data, sizeof(short) * size);
    cudaMalloc((void**)&dev_data_clear, sizeof(short) * size);

    cudaMemcpy(dev_data, data, sizeof(short) * size, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, 1);
    dim3 grid((block.x - 1 + size) / block.x, 1);

    convolution_1D_basic_kernel<<<grid, block>>>(dev_data, dev_data_clear, MAX_MASK_WIDTH, size); 
    
    cudaMemcpy(data_clear, dev_data_clear, sizeof(short) * size, cudaMemcpyDeviceToHost);

    f = fopen("audios/gpu_rst.wav", "wb+");
    if (!f)
    {
        printf("Open output file failed!\n");
        return -1;
    }
    writeWaveHeader(fmt, f);
    fseek(f, 44L, SEEK_SET);
    fwrite(data_clear, sizeof(short), size, f);

    fclose(f);
    f = NULL;
    
    free(data);
    free(data_clear);

    cudaFree(dev_data);
    cudaFree(dev_data_clear);
    return 0;    
}
