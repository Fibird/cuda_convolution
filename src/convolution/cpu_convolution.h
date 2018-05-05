#ifndef CPU_CONVOLUTION_H
#define CPU_CONVOLUTION_H
#include <opencv2/core/core.hpp>

typedef float element;

void conv2D(cv::Mat &src, cv::Mat &dst, cv::Mat kernel);

#endif
