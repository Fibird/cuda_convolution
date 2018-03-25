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
    __shared__ element N_ds[TILE_SIZE];

    N_ds[threadIdx.x] = N[i];

    __syncthreads();
    
    int this_tile_start_point = blockIdx.x * blockDim.x;
    int next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
    int N_start_point = i - (Mask_Width / 2);
    float Pvalue = 0;

    for (int j = 0; j < Mask_Width; j++)
    {
        int N_index = N_start_point + j;
        if (N_index >= 0 && N_index <= Width)
        {
            if (N_index >= this_tile_start_point 
                && N_index <= next_tile_start_point)
                {
                    Pvalue += N_ds[threadIdx.x + j - Mask_Width] * M[j];
                }
                else
                {
                    Pvalue += N[N_index] * M[j];
                }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Please specify file name!\n");
        exit(EXIT_FAILURE);
    }
    element *data = NULL;
    element *data_clear = NULL;
    element *dev_data, *dev_data_clear;

    const element h_M[MAX_MASK_WIDTH] = {1, 1, 1, 1, 1};
    cudaMemcpyToSymbol(M, h_M, MAX_MASK_WIDTH * sizeof(element));

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
    data = (element*)malloc(sizeof(element) * size);
    data_clear = (element*)malloc(sizeof(element) * size);

    fseek(f, 44L, SEEK_SET);
    fread(data, sizeof(element), size, f);
    if (f)
    {
        fclose(f);
        f = NULL;
    }
    
    cudaMalloc((void**)&dev_data, sizeof(element) * size);
    cudaMalloc((void**)&dev_data_clear, sizeof(element) * size);

    cudaMemcpy(dev_data, data, sizeof(element) * size, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, 1);
    dim3 grid((block.x - 1 + size) / block.x, 1);

    convolution_1D_basic_kernel<<<grid, block>>>(dev_data, dev_data_clear, MAX_MASK_WIDTH, size); 
    
    cudaMemcpy(data_clear, dev_data_clear, sizeof(element) * size, cudaMemcpyDeviceToHost);

    f = fopen("audios/gpu_rst.wav", "wb+");
    if (!f)
    {
        printf("Open output file failed!\n");
        return -1;
    }
    writeWaveHeader(fmt, f);
    fseek(f, 44L, SEEK_SET);
    fwrite(data_clear, sizeof(element), size, f);

    fclose(f);
    f = NULL;
    
    free(data);
    free(data_clear);

    cudaFree(dev_data);
    cudaFree(dev_data_clear);
    return 0;    
}
