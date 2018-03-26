#include <cuda_runtime.h>
#include <stdio.h>
#include "waveformat/waveformat.h"

#define TILE_SIZE 1024
#define MAX_MASK_WIDTH 5

typedef short element;

__global__ void convolution_1D_basic_kernel(element *N, element *P, element *M, int Mask_Width, int Width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N_start_point = i - (Mask_Width / 2);

    element Pvalue = 0;
    for (int j = 0; j < Mask_Width; j++)
    {
        if (N_start_point + j >= 0 && N_start_point + j < Width)
            Pvalue += N[N_start_point + j] * M[j];
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
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    element *data = NULL;
    element *data_clear = NULL;
    element *dev_data, *dev_data_clear;
    element *dev_M;

    const element h_M[MAX_MASK_WIDTH] = {1, 1, 1, 1, 1};
    cudaMalloc((void**)&dev_M, sizeof(element) * MAX_MASK_WIDTH);

    cudaMemcpy(dev_M, h_M, MAX_MASK_WIDTH * sizeof(element), cudaMemcpyHostToDevice);

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
    cudaEventRecord(start, 0);
    convolution_1D_basic_kernel<<<grid, block>>>(dev_data, dev_data_clear, dev_M, MAX_MASK_WIDTH, size); 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
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
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("%s elapsed %f ms\n", argv[0], elapsedTime);
    
    free(data);
    free(data_clear);

    cudaFree(dev_data);
    cudaFree(dev_data_clear);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;    
}
