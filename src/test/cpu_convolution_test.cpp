#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ctime>
#include "cpu_convolution.h"

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
    if (argc == 3)
        ks = atoi(argv[2]);

    image.convertTo(image, CV_32FC1);
    Mat result;
    Mat kernel = Mat::eye(ks, ks, CV_32FC1);
    
	float elapsedTime;
    clock_t start, stop;
    
    start = clock();
	conv2D(image, result, kernel);
    stop = clock();
    
    elapsedTime = (double) (stop - start) * 1000 / CLOCKS_PER_SEC;

	cout << elapsedTime << " ms"<< endl;
    result.convertTo(result, CV_8UC1);
    imwrite("result/cpu_conv.jpg", result);
	return 0;
}
