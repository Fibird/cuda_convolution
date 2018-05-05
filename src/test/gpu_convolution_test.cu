#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "gpu_convolution.h"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cout << "Please specify name of file!\n" << endl;
        return -1;
    }
    Mat image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    int ks = 3;
    ks = atoi(argv[2]);
    image.convertTo(image, CV_32FC1);
    Mat result;
    Mat kernel = Mat::eye(ks, ks, CV_32FC1);
    
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // execute median filter and time it
	cudaEventRecord(start, 0);

	conv2D(image, result, ks, kernel);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << elapsedTime << " ms"<< endl;
    result.convertTo(result, CV_8UC1);
    imwrite("result/restor.jpg", result);
	return 0;
}
