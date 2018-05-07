#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <ctime>
#include "cpu_convolution.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cout << "Usage: program <image> [<kernel size>]\n" << endl;
        return -1;
    }
    Mat image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    int ks = 3;
    if (argc == 3)
        ks = atoi(argv[2]);

    image.convertTo(image, CV_32FC1);
    Mat result;
    Mat ker_real;
	Mat ker_ref = Mat::zeros(ks * 2 + 1, ks * 2 + 1, CV_32FC1);
	ker_ref(Rect(0, ks, ks * 2 + 1, 1)) = Mat::ones(1, ks * 2 + 1, CV_32FC1);
    int i = 0;
    Mat rot_mat = getRotationMatrix2D(Point2f((float)ks, (float)ks), (float)i * 180.0 / (float)8, 1.0);
    // Get new kernel from ker_ref
	warpAffine(ker_ref, ker_real, rot_mat, ker_ref.size());
    
	float elapsedTime;
    clock_t start, stop;
    
    start = clock();
	conv2D(image, result, ker_real);
    stop = clock();
    
    elapsedTime = (double) (stop - start) * 1000 / CLOCKS_PER_SEC;

	cout << elapsedTime << " ms"<< endl;
    result.convertTo(result, CV_8UC1);
    imwrite("result/cpu_conv.png", result);
	return 0;
}
