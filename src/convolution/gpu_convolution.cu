#include "gpu_convolution.h"
#include <cuda_runtime.h>

__global__ void _conv2D(const element *signal, element *result, unsigned width, unsigned height, int ks, int ts_per_dm)
{
    int radius = ks / 2;
    // use dynamic size shared memory
    extern __shared__ element cache[];
    int sh_cols = ts_per_dm + radius * 2;
    int bk_cols = ts_per_dm;    int bk_rows = ts_per_dm;

	int gl_ix = threadIdx.x + blockDim.x * blockIdx.x;
	int gl_iy = threadIdx.y + blockDim.y * blockIdx.y;
    int ll_ix = threadIdx.x + radius;
    int ll_iy = threadIdx.y + radius;

	// Reads input elements into shared memory
	cache[ll_iy * sh_cols + ll_ix] = signal[gl_iy * width + gl_ix];
    // Marginal elements in cache
	if (threadIdx.x < radius)
	{
        int id = gl_iy * width + gl_ix - radius;
        cache[ll_iy * sh_cols + ll_ix - radius] = gl_ix < radius ? 0 : signal[id];
        id = gl_iy * width + gl_ix + bk_cols;
        cache[ll_iy * sh_cols + ll_ix + bk_cols] = gl_ix + bk_cols >= width ? 0 : signal[id];
	}
	if (threadIdx.y < radius)
	{
        int id = (gl_iy - radius) * width + gl_ix; 
        cache[(ll_iy - radius) * sh_cols + ll_ix] = gl_iy < radius ? 0 : signal[id];
        id = (gl_iy + bk_rows) * width + gl_ix;
        cache[(ll_iy + bk_rows) * sh_cols + ll_ix] = gl_iy + bk_rows >= height ? 0 :signal[id];
	}
    if (threadIdx.x < radius && threadIdx.y < radius)
    {
        int id = (gl_iy - radius) * width + gl_ix - radius;
        cache[(ll_iy - radius) * sh_cols + ll_ix - radius] = gl_iy < radius ? 0 : signal[id];
        id = (gl_iy - radius) * width + gl_ix + bk_cols;
        cache[(ll_iy - radius) * sh_cols + ll_ix + bk_cols] = gl_iy < radius ? 0 : signal[id];
        id = (gl_iy + bk_rows) * width + gl_ix + bk_cols;
        cache[(ll_iy + bk_rows) * sh_cols + ll_ix + bk_cols] = gl_iy + bk_rows >= height ? : signal[id];
        id = (gl_iy + bk_rows) * width + gl_ix - radius;
        cache[(ll_iy + bk_rows) * sh_cols + ll_ix - radius] = gl_iy + bk_rows >= height ? 0 : signal[id];
    }
	__syncthreads();

    // Get kernel element 
    element value = 0;
    for (int i = 0; i < ks; ++i)
	    for (int j = 0; j < ks; ++j)
	    	value  += cache[(ll_iy - radius + i) * sh_cols + ll_ix - radius + j] * Mask[i * ks + j];

	// Gets result 
    result[gl_iy * width + gl_ix] = value;
}

void conv2D(const cv::Mat &src, cv::Mat &dst, const cv::Mat &mask)
{
    if (!src.data)
        return;
    if (!src.isContinuous())
        return;

    unsigned width = src.size().width;
    unsigned height = src.size().height;
    int img_type = src.type();
    int ks = mask.size().width;
    
    element *dev_src, *dev_dst;
    element *dst_data;

    dst_data = (element*)malloc(sizeof(element) * width * height);
    CHECK(cudaMalloc((void**)&dev_src, sizeof(element) * width * height));
    CHECK(cudaMalloc((void**)&dev_dst, sizeof(element) * width * height));

    // transfer data to be filtered
    CHECK(cudaMemcpy(dev_src, (element*)src.data, sizeof(element) * width * height, cudaMemcpyHostToDevice));
    // copy mask data to constant memcoy - Mask
    CHECK(cudaMemcpyToSymbol(Mask, (element*)mask.data, sizeof(element) * ks * ks));

    // Set up execution configuration
    int ts_per_dm = 32;
    dim3 block(ts_per_dm, ts_per_dm);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    unsigned shared_size = (ts_per_dm + ks - 1) * (ts_per_dm + ks - 1) * sizeof(element);

    _conv2D<<<grid, block, shared_size>>>(dev_src, dev_dst, width, height, ks, ts_per_dm);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(dst_data, dev_dst, sizeof(element) * width * height, cudaMemcpyDeviceToHost));

    dst = cv::Mat(height, width, img_type, dst_data);
}
