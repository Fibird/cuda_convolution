#ifndef GPU_CONVOLUTION_H
#define GPU_CONVOLUTION_H
#include <opencv2/core/core.hpp>


#define MAX_SPACE 1024

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

// Signal/image element type
// Must be float or double
typedef float element;

// constant memory used to save covolution kernel
__constant__ element Mask[MAX_SPACE];

void conv2D(const cv::Mat &src, cv::Mat &dst, const cv::Mat &mask);

#endif
