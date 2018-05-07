#include "cpu_convolution.h"

void _conv2D(element *src, element *dst, element *kernel, unsigned width, unsigned height, int ks)
{
    int radius = ks / 2;
    for (unsigned i = 0; i < height; ++i)
    {
        for (unsigned j = 0; j < width; ++j)
        {
            element value = 0;
            for (unsigned y = 0; y < ks; y++)
            {
                for (unsigned x = 0; x < ks; x++)
                {
                    unsigned ll_x = j - radius + x;
                    unsigned ll_y = i - radius + y;
                    if (!(ll_x < 0 || ll_x >= width || 
                        ll_y < 0 || ll_y >= height))
                        value += src[ll_y * width + ll_x] * kernel[y * ks + x];
                }
            }
            dst[i * width + j] = value;
        }
    }
}

void conv2D(cv::Mat &src, cv::Mat &dst, cv::Mat kernel)
{
    unsigned width = src.size().width;
    unsigned height = src.size().height;
    int ks = kernel.size().width;

    if (!src.data || !kernel.data)
        return;
    if (!src.isContinuous())
        return;

    dst = cv::Mat(src.size(), src.type());
    _conv2D((element*)src.data, (element*)dst.data, (element*)kernel.data, width, height, ks);
}
